# Throughput with several runs at once: P6, worker processes, no TensorBoard, 30,720 transitions each

Total = sum over runs of each run's own training speed while all were running.

| Runs at once | Per run (transitions/s) | Total (transitions/s) | 2M runs per hour | Wall time (s) |
|---:|---|---:|---:|---:|
| 4 | 1263–1279 | 5076 | 9.1 | 30 |
| 8 | 892–901 | 7180 | 12.9 | 42 |
| 12 | 756–767 | 9152 | 16.5 | 51 |
| 16 | 616–626 | 9919 | 17.9 | 62 |
