# BACK/LEFT/RIGHT episodes that B4 fails on at least one seed

Outcomes: S reached, C crash, T timeout, L anchor lost. Distances in meters, speeds in m/s.
Replay any of them (from the repository root):

    .venv/bin/python -B baseline_dynamic_reach/examples.py show B4_seed7 --episode <episode>
    .venv/bin/python -B baseline_dynamic_reach/examples.py record B4_seed7 --episodes <episode> <episode>

| Episode | Slot | Lane changes | Slot ahead | Tag | Blocker gap | Blocker speed | B4 seeds 7/8/9 | Scripted |
|---:|---|---:|---:|---|---:|---:|---|---|
| 10002 | back | 0 | 90 | blocked | 35 | 12.0 | CCC | S |
| 10009 | right | 0 | 98 | blocked | 35 | 12.0 | CCT | T |
| 10022 | left | 1 | 101 | clear |  |  | CCT | S |
| 10023 | right | 1 | 74 | clear |  |  | SSC | S |
| 10025 | right | 1 | 66 | clear |  |  | SST | S |
| 10027 | right | 2 | 70 | clear |  |  | SST | S |
| 10028 | left | 1 | 99 | blocked | 67 | 12.6 | SST | S |
| 10030 | right | 1 | 96 | clear |  |  | CCC | C |
| 10031 | back | 2 | 54 | clear |  |  | SCS | S |
| 10038 | back | 1 | 87 | clear |  |  | SSC | S |
| 10039 | right | 1 | 106 | clear |  |  | SSC | T |
| 10041 | left | 0 | 65 | blocked | 35 | 12.0 | SSC | S |
| 10044 | left | 0 | 103 | blocked | 35 | 12.0 | CCC | T |
| 10045 | left | 0 | 68 | blocked | 35 | 12.0 | CCC | T |
| 10047 | right | 2 | 103 | clear |  |  | SSC | S |
| 10051 | right | 0 | 68 | blocked | 35 | 12.0 | CCC | T |
| 10053 | left | 1 | 104 | clear |  |  | SST | T |
| 10058 | left | 0 | 74 | blocked | 35 | 12.0 | CCC | T |
| 10059 | right | 1 | 101 | clear |  |  | SSC | S |
| 10062 | left | 0 | 102 | blocked | 35 | 12.0 | CCC | T |
| 10066 | left | 0 | 103 | blocked | 35 | 12.0 | CCC | T |
| 10068 | right | 0 | 69 | blocked | 35 | 12.0 | CCC | T |
| 10069 | left | 1 | 103 | clear |  |  | SST | S |
| 10071 | left | 1 | 101 | blocked | 66 | 16.5 | SCS | S |
| 10076 | right | 0 | 105 | blocked | 35 | 12.0 | CCT | T |
| 10081 | right | 1 | 107 | blocked | 71 | 13.0 | CCC | T |
| 10082 | right | 1 | 100 | blocked | 70 | 15.9 | SSC | S |
| 10084 | right | 2 | 105 | clear |  |  | CCT | T |
| 10088 | back | 0 | 93 | blocked | 35 | 12.0 | CCC | C |
| 10091 | right | 1 | 102 | blocked | 71 | 13.1 | SSC | S |
| 10096 | right | 1 | 101 | blocked | 70 | 14.3 | CCC | T |
| 10099 | back | 0 | 88 | blocked | 35 | 12.0 | CCC | T |
| 10104 | right | 1 | 103 | blocked | 68 | 13.5 | CCT | T |
| 10109 | left | 1 | 71 | clear |  |  | CCS | T |
| 10117 | left | 0 | 72 | blocked | 35 | 12.0 | CCS | S |
| 10120 | right | 0 | 107 | blocked | 35 | 12.0 | CCT | T |
| 10125 | right | 1 | 71 | clear |  |  | SST | S |
| 10127 | left | 1 | 108 | blocked | 72 | 12.3 | SSC | S |
| 10130 | left | 1 | 99 | blocked | 66 | 12.3 | SCC | S |
| 10136 | right | 1 | 107 | blocked | 70 | 15.5 | SST | S |
| 10139 | right | 0 | 70 | blocked | 35 | 12.0 | CCC | T |
| 10143 | left | 1 | 68 | clear |  |  | SST | S |
| 10154 | right | 2 | 71 | clear |  |  | SST | S |
| 10155 | right | 1 | 67 | clear |  |  | SST | S |
| 10158 | right | 1 | 102 | blocked | 71 | 16.9 | SST | S |
| 10169 | right | 0 | 99 | blocked | 35 | 12.0 | CCC | T |
| 10172 | right | 2 | 71 | clear |  |  | SST | S |
| 10174 | left | 1 | 107 | blocked | 71 | 12.7 | SST | S |
| 10177 | right | 0 | 67 | blocked | 35 | 12.0 | CCC | C |
| 10186 | back | 0 | 86 | blocked | 35 | 12.0 | SSC | S |
| 10188 | left | 0 | 72 | blocked | 35 | 12.0 | CCC | T |
| 10194 | right | 1 | 35 | clear |  |  | SSC | S |
| 10195 | right | 1 | 98 | clear |  |  | SST | S |
