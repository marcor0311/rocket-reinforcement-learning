# rocket-reinforcement-learning

## LunarLander v3

The project uses Gymnasium's installed `LunarLander-v3` environment. Its implementation stays inside the installed Gymnasium package; `.venv/` is the local Python environment and is ignored by Git.

### Setup

Python 3.12 is recommended for the Box2D dependency.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Watch a random agent

```bash
python watch_lunar_lander.py
```

This opens the human-rendered environment and samples random actions so you can see the lander and controls in motion. Close the window to stop.
