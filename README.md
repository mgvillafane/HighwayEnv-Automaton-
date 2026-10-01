# Highway Simulator

A minimal random-action highway driving simulator using Farama Foundation's
`highway-env`. The environment uses one controlled ego vehicle and the standard
discrete meta-actions. The `observation` returned by the environment is available
in the main loop for future controller and symbolic-proposition experiments.

## Setup (Windows PowerShell)

From this directory, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

The simulator opens a graphical highway window and runs one episode with random
discrete actions. Close the window or wait for the 40-second episode to finish;
the environment is closed when the program exits.