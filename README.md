---

title: Data Cleaning RL Environment Server
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:

* openenv
* reinforcement-learning
* data-cleaning

---

# 🧠 Data Cleaning RL Environment

A real-world OpenEnv environment that simulates **data cleaning workflows** as a sequential decision-making problem.
Designed for training and evaluating AI agents on tasks like handling missing values, removing outliers, and normalizing data.

---

# 🚀 Quick Start

```python
from data_cleaning_env import Action, DataCleaningEnv

try:
    # Start environment from Docker
    env = DataCleaningEnv.from_docker_image("data-cleaning-env:latest")

    # Reset environment
    result = env.reset()
    obs = result.observation
    print("Initial missing ratio:", obs.missing_ratio)

    # Take actions
    actions = [
        Action(action_type="fill_missing_mean", reason="Handle missing", confidence=0.9),
        Action(action_type="remove_outliers", reason="Clean anomalies", confidence=0.9),
        Action(action_type="normalize_data", reason="Reduce skewness", confidence=0.8),
    ]

    for action in actions:
        result = env.step(action)
        obs = result.observation

        print("Action:", action.action_type)
        print("Reward:", obs.reward)
        print("Remaining steps:", obs.steps_remaining)

finally:
    env.close()
```

---

# 🐳 Build Docker Image

```bash
docker build -t data-cleaning-env:latest .
```

---

# ☁️ Deploy to Hugging Face

```bash
openenv push --repo-id Sukhesh029/data-cleaning-env
```

After deployment:

👉 https://huggingface.co/spaces/Sukhesh029/data-cleaning-env

---

# 🧠 Environment Overview

This environment models **data preprocessing as an RL problem**:

```text
Observation → Action → Dataset Update → Reward → Next Observation
```

---

# 🎯 Tasks

### 🟢 Easy

* Focus: Missing value handling

### 🟡 Medium

* Missing values + Outliers

### 🔴 Hard

* Full pipeline (missing + outliers + skewness)

---

# 🎮 Action Space

Each action includes:

* `action_type`
* `reason`
* `confidence`

### Supported Actions

* `fill_missing_mean`
* `remove_outliers`
* `normalize_data`
* `do_nothing`

---

# 📊 Observation Space

The agent observes:

* `num_rows`
* `num_columns`
* `missing_ratio`
* `outlier_ratio`
* `skewness`
* `data_types`
* `steps_remaining`
* `reward`
* `done`

---

# 🏆 Reward Function

The environment provides **step-wise feedback**:

* Positive reward → improves data quality
* Negative reward → redundant or ineffective actions
* Penalizes repeated or low-confidence actions

---

# 🧪 Example Output

```text
[START] task=easy env=data_cleaning_env model=baseline
[STEP] step=1 action=fill_missing_mean reward=0.30 done=false
[STEP] step=2 action=normalize_data reward=0.20 done=false
...
[END] success=true steps=5 rewards=...
```

---

# 🔍 API Endpoints

| Endpoint | Description            |
| -------- | ---------------------- |
| `/reset` | Initialize environment |
| `/step`  | Apply action           |
| `/state` | Get current state      |
| `/docs`  | Swagger API            |

---

# ⚙️ Running Locally

```bash
uvicorn server.app:app --reload
```

Open:
👉 http://localhost:8000/docs

---

# 🧪 Testing Environment

```bash
python inference.py
```

This runs a baseline agent and outputs:

* Step-by-step actions
* Rewards
* Final score

---

# 📁 Project Structure

```
data_cleaning_env/
├── models.py
├── inference.py
├── grader.py
├── tasks/
│   ├── easy.py
│   ├── medium.py
│   └── hard.py
├── server/
│   ├── app.py
│   ├── data_cleaning_env_environment.py
│   └── Dockerfile
├── openenv.yaml
└── README.md
```

---

# 🌍 Use Cases

* Automated data preprocessing
* RL-based pipeline optimization
* Decision-making systems
* AI agents for data quality improvement

---

# 🏁 Summary

This project demonstrates how **real-world workflows can be modeled as RL environments**, enabling intelligent systems to learn optimal strategies for data cleaning.

---
