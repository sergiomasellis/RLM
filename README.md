# Recursive Language Model (RLM)

A LangGraph-based implementation of recursive problem decomposition where an LLM breaks down complex problems into subproblems and solves them hierarchically.

Based on the paper "Recursive Language Models" by Zhang, Kraska, and Khattab.

## Overview

RLM tackles complex problems by recursively decomposing them into simpler subproblems, solving each subproblem (potentially through further decomposition), and aggregating the results into a final solution.

### Key Features

- **Automatic Problem Analysis**: Determines whether a problem should be decomposed or solved directly
- **Recursive Decomposition**: Breaks complex problems into 2-4 manageable subproblems
- **Dependency-Aware Execution**: Solves subproblems in topological order based on dependencies
- **Solution Aggregation**: Synthesizes subproblem solutions into coherent final answers
- **Configurable Depth**: Control maximum recursion depth to balance thoroughness vs. efficiency
- **Execution Tracing**: Full visibility into the decomposition and solving process

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        RLM Graph                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐                                              │
│   │  START   │                                              │
│   └────┬─────┘                                              │
│        │                                                    │
│        ▼                                                    │
│   ┌──────────┐                                              │
│   │ Analyze  │ ─── Decide: decompose or solve directly?     │
│   └────┬─────┘                                              │
│        │                                                    │
│        ├──────────────────────┐                             │
│        │ (decompose)         │ (solve directly)            │
│        ▼                      ▼                             │
│   ┌──────────┐          ┌──────────────┐                    │
│   │Decompose │          │Solve Directly│                    │
│   └────┬─────┘          └──────┬───────┘                    │
│        │                       │                            │
│        ▼                       │                            │
│   ┌────────────────┐           │                            │
│   │Solve Subproblems│ ◄────────┤ (recursive)                │
│   └────┬───────────┘           │                            │
│        │                       │                            │
│        ▼                       │                            │
│   ┌──────────┐                 │                            │
│   │Aggregate │                 │                            │
│   └────┬─────┘                 │                            │
│        │                       │                            │
│        └───────────┬───────────┘                            │
│                    ▼                                        │
│               ┌─────────┐                                   │
│               │   END   │                                   │
│               └─────────┘                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Access to an OpenAI-compatible LLM API (default: LM Studio)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sergiomasellis/RLM.git
cd RLM
```

2. Install dependencies with uv:
```bash
uv sync
```

Or with pip:
```bash
pip install -e .
```

3. Configure your LLM endpoint in `main.py`:
```python
DEFAULT_MODEL = "your-model-name"
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"  # Your API endpoint
```

## Usage

### Basic Usage

```python
from main import solve

# Solve a complex problem
solution = solve(
    "What are the economic implications of transitioning to renewable energy?",
    max_depth=3,
    verbose=True
)
```

### Command Line

```bash
uv run python main.py
```

This runs the built-in demo with example problems.

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 3 | Maximum recursion depth |
| `verbose` | True | Print execution trace |
| `DEFAULT_MODEL` | "allenai/olmo-3-32b-think" | LLM model to use |
| `MIN_SUBPROBLEMS` | 2 | Minimum subproblems per decomposition |
| `MAX_SUBPROBLEMS` | 4 | Maximum subproblems per decomposition |

## How It Works

### 1. Problem Analysis

The system first analyzes the input problem to determine if it should be:
- **Decomposed**: Split into smaller subproblems (for complex, multi-part problems)
- **Solved Directly**: Answered without decomposition (for simple, atomic problems)

### 2. Decomposition

When decomposition is chosen, the LLM:
- Identifies 2-4 distinct subproblems
- Establishes dependencies between subproblems
- Defines an aggregation strategy for combining solutions

### 3. Recursive Solving

Each subproblem is solved by recursively invoking the RLM graph:
- Subproblems may themselves be decomposed (up to `max_depth`)
- Dependencies are resolved in topological order
- Context from solved dependencies is passed to dependent subproblems

### 4. Aggregation

Solutions are combined using the defined aggregation strategy to produce the final answer.

## Example Problems

### Multi-Step Reasoning
```python
problem = """
A company has 3 departments: Engineering (40 people), Sales (25 people),
and Marketing (15 people). Each produces different value per employee.
Which department should they hire for to grow by 20%?
"""
solve(problem)
```

### Ecosystem Analysis
```python
problem = """
Consider an ecosystem: Rabbits eat grass, Foxes eat rabbits,
Eagles eat both. If drought reduces grass by 50%, what are
the cascading effects?
"""
solve(problem)
```

### Technical Trade-offs
```python
problem = """
Explain the trade-offs between linked list vs array for implementing
a queue. Which would you recommend for high-frequency trading?
"""
solve(problem)
```

## Project Structure

```
RLM/
├── main.py           # Core implementation
├── pyproject.toml    # Project configuration
├── uv.lock           # Dependency lock file
├── .python-version   # Python version specification
├── .gitignore        # Git ignore rules
├── LICENSE           # MIT License
└── README.md         # This file
```

## Data Structures

### RLMState

The main state object passed through the graph:

```python
class RLMState(TypedDict):
    problem: str                      # Current problem to solve
    depth: int                        # Current recursion depth
    max_depth: int                    # Maximum allowed depth
    should_decompose: bool            # Decomposition decision
    subproblems: list[SubProblemState]  # Decomposed subproblems
    subproblem_solutions: dict[int, str]  # Solutions keyed by ID
    aggregation_strategy: str         # How to combine solutions
    solution: str                     # Final solution
    trace: list[str]                  # Execution trace
```

### SubProblemState

Represents a single subproblem:

```python
class SubProblemState(TypedDict):
    id: int                 # Unique identifier
    description: str        # Problem description
    dependencies: list[int] # IDs of prerequisite subproblems
    solution: str | None    # Solution once solved
    depth: int              # Recursion depth
```

## Dependencies

- `langchain-openai>=1.1.7` - OpenAI-compatible LLM interface
- `langchain[ollama]>=1.2.6` - LangChain core with Ollama support
- `langgraph>=1.0.6` - Graph-based LLM orchestration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Zhang, Kraska, and Khattab - "Recursive Language Models"
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)

## Acknowledgments

- The LangChain team for the excellent LangGraph framework
- The authors of the RLM paper for the conceptual foundation
