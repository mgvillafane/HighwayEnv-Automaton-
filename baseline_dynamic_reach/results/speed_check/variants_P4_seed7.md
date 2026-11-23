# Speed-up candidates: P4, seed 7, 61,440 transitions, one run alone

| Setup | Transitions/s | Setup time (s) | Training log vs reference | Final weights vs reference |
|---|---:|---:|---|---|
| worker processes + TensorBoard (train.py until now) | 1443 | 2.3 | identical (1588 episodes) | identical |
| worker processes, no TensorBoard | 1440 | 2.4 | identical (1588 episodes) | identical |
| in-process environments + TensorBoard | 821 | 0.8 | identical (1588 episodes) | identical |
| in-process environments, no TensorBoard | 823 | 0.8 | identical (1588 episodes) | identical |
