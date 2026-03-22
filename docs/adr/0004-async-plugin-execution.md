# ADR-0004: Async Plugin Execution

## Status

Accepted

## Context

ADR-0001 established that the pipeline exposes plugin hooks at each stage. Plugins handle monitoring, visualization, and analysis. We need to decide how plugins execute relative to the main pipeline.

## Considered Options

### Option 1: Synchronous execution

Plugins run inline, blocking the pipeline at each hook.

- **Pros**: Simple execution model, guaranteed ordering, plugins see consistent state
- **Cons**: Slow plugins block the entire pipeline, no parallelism between plugins

### Option 2: Async execution (chosen)

Plugins run asynchronously, scheduled around regular pipeline tasks.

- **Pros**: Pipeline is never blocked by plugins, plugins can do expensive work (network calls, graph rendering) without impacting build time, flexible scheduling
- **Cons**: Plugins see potentially stale state, ordering not guaranteed, error handling is more complex

## Decision

Use **async plugin execution**. Plugins are dispatched asynchronously at lifecycle events and scheduled around pipeline tasks. The pipeline does not wait for plugins to complete before proceeding to the next stage.

Design principles:
- Plugins receive an immutable snapshot of pipeline state at the time of the event
- Plugin failures are logged but do not fail the pipeline
- A plugin can optionally declare itself as "critical" to block progression (e.g., a compliance gate)
- The pipeline provides a `await_plugins()` barrier for stages that need all plugins to complete before proceeding

## Consequences

### Positive

- Pipeline throughput is not bottlenecked by plugin execution time
- Plugins can perform expensive operations (HTTP calls, graph generation) freely
- Natural fit for monitoring/observability use cases where fire-and-forget is acceptable

### Negative

- Plugins cannot modify pipeline state (they receive snapshots)
- Debugging plugin timing issues is harder
- Need to handle plugin lifecycle (startup, shutdown, error recovery)

## Related Decisions

- [ADR-0001](0001-llvm-inspired-pipeline-architecture.md) — parent architecture decision
