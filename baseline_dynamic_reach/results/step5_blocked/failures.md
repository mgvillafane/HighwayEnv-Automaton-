# BACK/LEFT/RIGHT episodes that B5 fails on at least one seed

Outcomes: S reached, C crash, T timeout, L anchor lost. Distances in meters, speeds in m/s.
Replay any of them (from the repository root):

    .venv/bin/python -B baseline_dynamic_reach/examples.py show B5_seed7 --episode <episode>
    .venv/bin/python -B baseline_dynamic_reach/examples.py record B5_seed7 --episodes <episode> <episode>

| Episode | Slot | Lane changes | Slot ahead | Tag | Blocker gap | Blocker speed | B5 seeds 7/8/9 | Scripted |
|---:|---|---:|---:|---|---:|---:|---|---|
| 10002 | back | 0 | 90 | blocked | 35 | 12.0 | CCC | S |
| 10009 | right | 0 | 98 | blocked | 35 | 12.0 | CCC | T |
| 10030 | right | 1 | 96 | clear |  |  | CCC | C |
| 10044 | left | 0 | 103 | blocked | 35 | 12.0 | CCC | T |
| 10045 | left | 0 | 68 | blocked | 35 | 12.0 | CCC | T |
| 10051 | right | 0 | 68 | blocked | 35 | 12.0 | CCC | T |
| 10058 | left | 0 | 74 | blocked | 35 | 12.0 | CCC | T |
| 10062 | left | 0 | 102 | blocked | 35 | 12.0 | CCC | T |
| 10066 | left | 0 | 103 | blocked | 35 | 12.0 | CCC | T |
| 10068 | right | 0 | 69 | blocked | 35 | 12.0 | CCC | T |
| 10071 | left | 1 | 101 | blocked | 66 | 16.5 | CCC | S |
| 10076 | right | 0 | 105 | blocked | 35 | 12.0 | CCC | T |
| 10081 | right | 1 | 107 | blocked | 71 | 13.0 | CCC | T |
| 10084 | right | 2 | 105 | clear |  |  | CCC | T |
| 10088 | back | 0 | 93 | blocked | 35 | 12.0 | CCC | C |
| 10096 | right | 1 | 101 | blocked | 70 | 14.3 | CCC | T |
| 10099 | back | 0 | 88 | blocked | 35 | 12.0 | CCC | T |
| 10104 | right | 1 | 103 | blocked | 68 | 13.5 | CCC | T |
| 10109 | left | 1 | 71 | clear |  |  | CCC | T |
| 10117 | left | 0 | 72 | blocked | 35 | 12.0 | CCC | S |
| 10120 | right | 0 | 107 | blocked | 35 | 12.0 | CCC | T |
| 10139 | right | 0 | 70 | blocked | 35 | 12.0 | CCC | T |
| 10169 | right | 0 | 99 | blocked | 35 | 12.0 | CCC | T |
| 10177 | right | 0 | 67 | blocked | 35 | 12.0 | CCC | C |
| 10188 | left | 0 | 72 | blocked | 35 | 12.0 | CCC | T |
