# Throughput with several runs at once: P6, in-process environments, no TensorBoard, 30,720 transitions each

Total = sum over runs of each run's own training speed while all were running.

| Runs at once | Per run (transitions/s) | Total (transitions/s) | 2M runs per hour | Wall time (s) |
|---:|---|---:|---:|---:|
| 8 | 790–802 | 6352 | 11.4 | 42 |
| 16 | 537–556 | 8756 | 15.8 | 62 |
| 24 | 433–443 | 10496 | 18.9 | 77 |
| 32 | 344–401 | 11554 | 20.8 | 96 |
