# ADR-0007: Autotuning Strategy Classes

## Status

Accepted

## Context

The cartesian product of problem sizes and kernel configurations can be enormous. A MatMul with 32 M values × 32 N values × 8 K values × 50 kernel configs = 409,600 benchmarks. Running all of them is impractical. We need intelligent search strategies to find good configurations without exhaustive sweeps.

## Decision

Introduce a **Strategy** class that defines how to explore the (problem size × config) search space. Strategies receive the full space definition and yield subsets of points to evaluate, using results from previous evaluations to guide the search.

```python
class Strategy(Protocol):
    def suggest(
        self,
        space: SearchSpace,
        results: list[AutotuneResult],
    ) -> list[SearchPoint]:
        """Suggest the next batch of points to evaluate."""
        ...

    def is_converged(self, results: list[AutotuneResult]) -> bool:
        """Whether the strategy considers the search complete."""
        ...
```

### Built-in strategies

| Strategy | Use case |
|----------|----------|
| **Exhaustive** | Small spaces — run everything |
| **BasinHopping** | Global optimization with local perturbation |
| **BayesianOptimization** | Sample-efficient search using surrogate models |
| **DualAnnealing** | Global optimization for highly non-convex landscapes |

### Composability

Strategies can be composed. For example, a common pattern is to use Bayesian optimization to find promising regions, then exhaustive search within those regions:

```python
strategy = TwoPhase(
    explore=BayesianOptimization(n_initial=50, n_iterations=200),
    exploit=Exhaustive(),  # only within top-k regions
)
```

### Integration with the pipeline

The autotuner loop:

1. Strategy suggests a batch of `SearchPoint`s (each is a (problem_size, config) pair)
2. Backend compiles and benchmarks the batch
3. Results (including Observer metrics) are fed back to the strategy
4. Repeat until `is_converged()` or budget exhausted

The strategy operates on both axes: it prunes both **problem sizes** (which sizes to test) and **configs** (which kernel configurations to try), since performance patterns often transfer across nearby sizes.

## Consequences

### Positive

- Pluggable — users can implement custom strategies for domain-specific heuristics
- Batch-oriented `suggest()` enables parallel benchmark execution
- Convergence check prevents wasted compute on diminishing returns
- Built-in strategies cover common optimization approaches

### Negative

- Strategy quality directly impacts autotune result quality — a bad strategy can miss the optimal config
- Bayesian optimization adds a dependency (likely `scikit-optimize` or `optuna`)
- Strategy hyperparameters (n_initial, n_iterations) are themselves tuning knobs

## Related Decisions

- [ADR-0005](0005-problem-specification-format.md) — defines the problem size space
- [ADR-0006](0006-source-as-ir-native-compilation.md) — compilation produces the configs to search over
- [ADR-0008](0008-observer-custom-metrics.md) — observer metrics feed into strategy results
