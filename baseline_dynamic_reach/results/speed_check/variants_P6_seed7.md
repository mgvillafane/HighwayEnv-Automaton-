# Speed-up candidates: P6, seed 7, 61,440 transitions, one run alone

| Setup | Transitions/s | Setup time (s) | Training log vs reference | Final weights vs reference |
|---|---:|---:|---|---|
| worker processes + TensorBoard (train.py until now) | 1729 | 2.4 | identical (1033 episodes) | identical |
| worker processes, no TensorBoard | 1739 | 2.4 | identical (1033 episodes) | identical |
| in-process environments + TensorBoard | 911 | 0.8 | identical (1033 episodes) | identical |
| in-process environments, no TensorBoard | 898 | 0.8 | identical (1033 episodes) | identical |
