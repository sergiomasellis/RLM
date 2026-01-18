"""
Recursive Language Model (RLM) Implementation

Based on the paper "Recursive Language Models" by Zhang, Kraska, and Khattab.
This implementation uses LangGraph to create a recursive problem-solving framework
where an LLM can decompose complex problems into subproblems and solve them hierarchically.

Key Components:
1. Problem Decomposition - LLM breaks down complex tasks into subtasks
2. Recursive Execution - Subtasks can themselves be decomposed recursively
3. Context Management - Results are summarized and passed up the recursion tree
4. Result Aggregation - Final answer synthesized from subproblem solutions
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator
import json
import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL = "allenai/olmo-3-32b-think"
LM_STUDIO_BASE_URL = "http://10.5.0.2:1234/v1"
MAX_RECURSION_DEPTH = 3
MIN_SUBPROBLEMS = 2
MAX_SUBPROBLEMS = 4


# ============================================================================
# Structured Output Schemas
# ============================================================================

class DecompositionDecision(BaseModel):
    """Decision on whether to decompose a problem."""
    should_decompose: bool = Field(
        description="True if the problem should be broken into subproblems, False if it can be solved directly"
    )
    reasoning: str = Field(
        description="Brief explanation of why decomposition is or isn't needed"
    )


class SubProblem(BaseModel):
    """A subproblem derived from decomposition."""
    id: int = Field(description="Unique identifier for this subproblem")
    description: str = Field(description="Clear description of what needs to be solved")
    dependencies: list[int] = Field(
        default_factory=list,
        description="IDs of subproblems that must be solved before this one"
    )


class DecompositionResult(BaseModel):
    """Result of decomposing a problem into subproblems."""
    subproblems: list[SubProblem] = Field(
        description="List of subproblems to solve"
    )
    aggregation_strategy: str = Field(
        description="How to combine subproblem solutions into final answer"
    )


class DirectSolution(BaseModel):
    """Direct solution to an atomic problem."""
    answer: str = Field(description="The solution to the problem")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the solution (0-1)"
    )
    reasoning: str = Field(description="Step-by-step reasoning for the solution")


class AggregatedResult(BaseModel):
    """Final aggregated result from subproblem solutions."""
    final_answer: str = Field(description="The synthesized final answer")
    summary: str = Field(description="Brief summary of how subproblems contributed")


# ============================================================================
# State Definition
# ============================================================================

class SubProblemState(TypedDict):
    """State for a single subproblem in the recursion tree."""
    id: int
    description: str
    dependencies: list[int]
    solution: str | None
    depth: int


class RLMState(TypedDict):
    """Main state for the Recursive Language Model graph."""
    # The original/current problem to solve
    problem: str

    # Current recursion depth
    depth: int

    # Maximum allowed depth
    max_depth: int

    # Whether to decompose or solve directly
    should_decompose: bool

    # Subproblems if decomposed
    subproblems: list[SubProblemState]

    # Solutions to subproblems (id -> solution)
    subproblem_solutions: dict[int, str]

    # Strategy for aggregating solutions
    aggregation_strategy: str

    # Final solution
    solution: str

    # Execution trace for debugging
    trace: Annotated[list[str], operator.add]


# ============================================================================
# LLM Setup
# ============================================================================

def get_llm(model: str = DEFAULT_MODEL) -> ChatOpenAI:
    """Initialize the LM Studio LLM via OpenAI-compatible API."""
    return ChatOpenAI(
        model=model,
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio",  # LM Studio doesn't require a real key
        temperature=0.7,
    )


# ============================================================================
# Node Functions
# ============================================================================

def analyze_problem(state: RLMState) -> dict:
    """
    Analyze the problem and decide whether to decompose or solve directly.

    Decomposition criteria:
    - Problem complexity (multiple steps/parts)
    - Current depth vs max depth
    - Whether subproblems would be more tractable
    """
    llm = get_llm()

    # Force direct solution at max depth
    if state["depth"] >= state["max_depth"]:
        return {
            "should_decompose": False,
            "trace": [f"[Depth {state['depth']}] At max depth, solving directly"]
        }

    system_msg = f"""You are an expert problem analyzer. Determine whether to decompose into subproblems or solve directly.

DECOMPOSE when: multiple distinct parts, requires multi-step reasoning, synthesizes different domains.
SOLVE DIRECTLY when: simple, self-contained, straightforward calculation.

Current depth: {state["depth"]}/{state["max_depth"]}. At max depth, MUST solve directly.

IMPORTANT: Respond with ONLY valid JSON, no other text:
{{"should_decompose": true, "reasoning": "why decompose"}}
or
{{"should_decompose": false, "reasoning": "why solve directly"}}"""

    human_msg = f"Problem: {state['problem']}\n\nShould this be decomposed into subproblems?"

    response = llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=human_msg)
    ])

    # Parse the response
    try:
        # Extract JSON from response
        content = response.content
        # Handle potential markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        should_decompose = result.get("should_decompose", False)
        reasoning = result.get("reasoning", "No reasoning provided")
    except (json.JSONDecodeError, IndexError):
        # Default to solving directly if parsing fails
        should_decompose = False
        reasoning = "Could not parse decomposition decision, solving directly"

    return {
        "should_decompose": should_decompose,
        "trace": [f"[Depth {state['depth']}] Analysis: {reasoning} -> {'decompose' if should_decompose else 'solve directly'}"]
    }


def decompose_problem(state: RLMState) -> dict:
    """
    Decompose the problem into smaller, manageable subproblems.
    """
    llm = get_llm()

    system_msg = """You are an expert at breaking down complex problems into simpler subproblems.

Rules:
1. Create 2-4 subproblems that together solve the original problem
2. Each subproblem should be simpler than the original
3. Identify dependencies between subproblems (which must be solved first)
4. Provide a clear strategy for combining the solutions

Respond with JSON:
{
    "subproblems": [
        {"id": 1, "description": "subproblem description", "dependencies": []},
        {"id": 2, "description": "another subproblem", "dependencies": [1]}
    ],
    "aggregation_strategy": "How to combine solutions into final answer"
}"""

    human_msg = f"Problem to decompose: {state['problem']}"

    response = llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=human_msg)
    ])

    # Parse the decomposition
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())

        subproblems = [
            SubProblemState(
                id=sp["id"],
                description=sp["description"],
                dependencies=sp.get("dependencies", []),
                solution=None,
                depth=state["depth"] + 1
            )
            for sp in result.get("subproblems", [])
        ]
        aggregation_strategy = result.get("aggregation_strategy", "Combine all solutions")

    except (json.JSONDecodeError, IndexError, KeyError) as e:
        # Fallback: create a simple decomposition
        subproblems = [
            SubProblemState(
                id=1,
                description=f"Analyze: {state['problem']}",
                dependencies=[],
                solution=None,
                depth=state["depth"] + 1
            ),
            SubProblemState(
                id=2,
                description=f"Synthesize answer for: {state['problem']}",
                dependencies=[1],
                solution=None,
                depth=state["depth"] + 1
            )
        ]
        aggregation_strategy = "Use analysis to form final answer"

    subproblem_descriptions = [f"  {sp['id']}: {sp['description']}" for sp in subproblems]

    return {
        "subproblems": subproblems,
        "aggregation_strategy": aggregation_strategy,
        "trace": [f"[Depth {state['depth']}] Decomposed into {len(subproblems)} subproblems:\n" + "\n".join(subproblem_descriptions)]
    }


def solve_subproblems(state: RLMState) -> dict:
    """
    Recursively solve each subproblem.

    This is the core recursive mechanism - each subproblem is solved
    by invoking the RLM graph recursively.
    """
    solutions = {}
    trace_entries = []

    # Sort subproblems by dependencies (topological sort)
    solved_ids = set()
    remaining = list(state["subproblems"])

    while remaining:
        # Find subproblems whose dependencies are satisfied
        ready = [sp for sp in remaining if all(d in solved_ids for d in sp["dependencies"])]

        if not ready:
            # Circular dependency or error - solve remaining in order
            ready = remaining[:1]

        for subproblem in ready:
            # Build context from solved dependencies
            context = ""
            if subproblem["dependencies"]:
                dep_solutions = [f"Solution to subproblem {d}: {solutions.get(d, 'Not yet solved')}"
                               for d in subproblem["dependencies"]]
                context = "Context from previous solutions:\n" + "\n".join(dep_solutions) + "\n\n"

            # Recursively solve this subproblem
            sub_state: RLMState = {
                "problem": context + subproblem["description"],
                "depth": subproblem["depth"],
                "max_depth": state["max_depth"],
                "should_decompose": False,
                "subproblems": [],
                "subproblem_solutions": {},
                "aggregation_strategy": "",
                "solution": "",
                "trace": []
            }

            # Invoke the recursive graph
            result = rlm_graph.invoke(sub_state)

            solutions[subproblem["id"]] = result["solution"]
            solved_ids.add(subproblem["id"])
            trace_entries.extend(result["trace"])
            trace_entries.append(f"[Depth {subproblem['depth']}] Solved subproblem {subproblem['id']}: {result['solution'][:100]}...")

        remaining = [sp for sp in remaining if sp["id"] not in solved_ids]

    return {
        "subproblem_solutions": solutions,
        "trace": trace_entries
    }


def aggregate_solutions(state: RLMState) -> dict:
    """
    Aggregate subproblem solutions into a final answer.
    """
    llm = get_llm()

    # Format subproblem solutions
    solutions_text = "\n".join([
        f"Subproblem {sp['id']} ({sp['description']}):\n{state['subproblem_solutions'].get(sp['id'], 'No solution')}\n"
        for sp in state["subproblems"]
    ])

    system_msg = """You are an expert at synthesizing information from multiple sources into a coherent answer.

Given solutions to subproblems, create a final comprehensive answer to the original problem.
Follow the aggregation strategy provided.

Be concise but complete. Ensure the final answer directly addresses the original question."""

    human_msg = f"""Original Problem: {state["problem"]}

Aggregation Strategy: {state["aggregation_strategy"]}

Subproblem Solutions:
{solutions_text}

Provide the final synthesized answer:"""

    response = llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=human_msg)
    ])

    return {
        "solution": response.content,
        "trace": [f"[Depth {state['depth']}] Aggregated {len(state['subproblems'])} solutions into final answer"]
    }


def solve_directly(state: RLMState) -> dict:
    """
    Solve the problem directly without decomposition.
    """
    llm = get_llm()

    system_msg = """You are a helpful assistant that solves problems step by step.
Think through the problem carefully and provide a clear, accurate answer.
If the problem includes context from previous solutions, use that information."""

    response = llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=state["problem"])
    ])

    return {
        "solution": response.content,
        "trace": [f"[Depth {state['depth']}] Solved directly: {response.content[:100]}..."]
    }


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_analysis(state: RLMState) -> Literal["decompose", "solve_directly"]:
    """Route based on decomposition decision."""
    if state["should_decompose"]:
        return "decompose"
    return "solve_directly"


# ============================================================================
# Graph Construction
# ============================================================================

def build_rlm_graph() -> StateGraph:
    """Build the Recursive Language Model graph."""
    builder = StateGraph(RLMState)

    # Add nodes
    builder.add_node("analyze", analyze_problem)
    builder.add_node("decompose", decompose_problem)
    builder.add_node("solve_subproblems", solve_subproblems)
    builder.add_node("aggregate", aggregate_solutions)
    builder.add_node("solve_directly", solve_directly)

    # Add edges
    builder.add_edge(START, "analyze")
    builder.add_conditional_edges("analyze", route_after_analysis)
    builder.add_edge("decompose", "solve_subproblems")
    builder.add_edge("solve_subproblems", "aggregate")
    builder.add_edge("aggregate", END)
    builder.add_edge("solve_directly", END)

    return builder.compile()


# Global graph instance (needed for recursion)
rlm_graph = build_rlm_graph()


# ============================================================================
# Main Interface
# ============================================================================

def solve(
    problem: str,
    max_depth: int = MAX_RECURSION_DEPTH,
    verbose: bool = True
) -> str:
    """
    Solve a problem using the Recursive Language Model approach.

    Args:
        problem: The problem to solve
        max_depth: Maximum recursion depth (default: 3)
        verbose: Whether to print execution trace

    Returns:
        The solution to the problem
    """
    initial_state: RLMState = {
        "problem": problem,
        "depth": 0,
        "max_depth": max_depth,
        "should_decompose": False,
        "subproblems": [],
        "subproblem_solutions": {},
        "aggregation_strategy": "",
        "solution": "",
        "trace": []
    }

    print(f"\n{'='*60}")
    print("RECURSIVE LANGUAGE MODEL")
    print(f"{'='*60}")
    print(f"Problem: {problem}")
    print(f"Max Depth: {max_depth}")
    print(f"{'='*60}\n")

    result = rlm_graph.invoke(initial_state)

    if verbose:
        print("\n--- Execution Trace ---")
        for entry in result["trace"]:
            print(entry)
        print("--- End Trace ---\n")

    print(f"\n{'='*60}")
    print("SOLUTION")
    print(f"{'='*60}")
    print(result["solution"])
    print(f"{'='*60}\n")

    return result["solution"]


# ============================================================================
# Demo
# ============================================================================

def main():
    """Demo the Recursive Language Model with example problems."""

    # Example 1: Multi-step reasoning problem
    problem1 = """
    A company has 3 departments: Engineering (40 people), Sales (25 people), and Marketing (15 people).
    Each Engineering employee produces $50,000 in value per year.
    Each Sales employee produces $80,000 in value per year.
    Each Marketing employee produces $30,000 in value per year.

    The company wants to grow by 20% in total value next year.
    They can only hire in one department.
    Which department should they hire for, and how many people do they need?
    """

    # Example 2: Multi-hop reasoning
    problem2 = """
    Consider a simple ecosystem with the following relationships:
    - Rabbits eat grass
    - Foxes eat rabbits
    - Eagles eat both rabbits and small foxes
    - Grass needs sunlight and water to grow

    If there's a drought that reduces grass by 50%, what will be the cascading effects
    on each species in the ecosystem? Provide a detailed analysis.
    """

    # Example 3: Code analysis (simpler)
    problem3 = """
    Explain the trade-offs between using a linked list vs an array for implementing a queue.
    Consider: memory usage, time complexity for operations, and cache performance.
    Which would you recommend for a high-frequency trading system and why?
    """

    print("\n" + "="*80)
    print("RECURSIVE LANGUAGE MODEL DEMO")
    print("="*80)

    # Run demo with ecosystem problem (more complex, should trigger decomposition)
    print("\n>>> Solving Problem 2: Ecosystem Analysis\n")
    solve(problem3, max_depth=3)


if __name__ == "__main__":
    main()
