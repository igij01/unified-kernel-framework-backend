# ADR-0008: Observer for Custom Autotuning Metrics

## Status

Accepted

## Context

Autotuning typically measures wall-clock execution time. But kernel developers often need additional metrics to understand *why* a configuration is fast or slow:

- Register usage (from NCU)
- Shared memory occupancy
- Memory throughput (GB/s)
- Arithmetic intensity
- Warp divergence

These metrics are critical for understanding kernel behavior and making informed optimization decisions, but they vary by use case — no fixed set covers all needs.

## Decision

Introduce an **Observer** class that has access to the underlying device during autotuning runs. Observers collect custom metrics for each benchmark point, and their return values are automatically appended to the autotune result record.

```python
class Observer(Protocol):
    def setup(self, device: DeviceHandle) -> None:
        """Called once before the autotuning run starts.
        Use to initialize profiling tools, counters, etc."""
        ...

    def before_run(self, device: DeviceHandle, point: SearchPoint) -> None:
        """Called before each kernel invocation."""
        ...

    def after_run(self, device: DeviceHandle, point: SearchPoint) -> dict[str, float]:
        """Called after each kernel invocation.
        Returns a dict of metric_name → value, automatically stored in results."""
        ...

    def teardown(self, device: DeviceHandle) -> None:
        """Called once after the autotuning run completes."""
        ...
```

### DeviceHandle

`DeviceHandle` wraps the GPU device and provides access to:

- Device properties (SM count, memory size, compute capability)
- Profiling APIs (NCU metrics via `ncu` CLI or CUPTI)
- Memory state (allocated, free)

### Built-in observers

| Observer | Metrics |
|----------|---------|
| **TimingObserver** | Wall-clock time (always active by default) |
| **NCUObserver** | Register usage, shared memory, occupancy, throughput |
| **MemoryObserver** | Peak memory allocation during kernel execution |

### Result integration

Observer metrics are stored alongside timing data in the autotune result database:

```
kernel_hash | arch | problem_size | config | time_ms | registers | shared_mem_bytes | occupancy | ...
```

Analysis plugins can then query and visualize any metric, not just time.

### Multiple observers

Multiple observers can be attached to a single autotuning run. Each contributes its own columns to the result record. If two observers return the same metric name, the later one overwrites (with a warning).

## Consequences

### Positive

- Developers get deep insight into kernel behavior during autotuning, not just timing
- Extensible — custom observers for domain-specific metrics (e.g., numerical error tracking)
- Metrics stored in the database alongside results — available for analysis plugins
- `before_run`/`after_run` hooks give precise control over measurement boundaries

### Negative

- NCU profiling significantly slows down kernel execution — observers that invoke profiling tools should be opt-in, not default
- `DeviceHandle` abstraction must wrap multiple profiling APIs (CUPTI, NCU CLI, `nvidia-smi`)
- Some metrics require running the kernel multiple times (NCU replays), which the observer must coordinate with the autotuner

### Open Questions

- [ ] Should observers be able to influence the autotuning strategy (e.g., skip configs that exceed register limits)?
- [ ] How to handle NCU's kernel replay requirement — does the observer control the number of runs, or does the autotuner?

## Related Decisions

- [ADR-0003](0003-database-for-autotune-storage.md) — observer metrics stored in the database
- [ADR-0007](0007-autotuning-strategies.md) — strategies consume observer metrics in results
