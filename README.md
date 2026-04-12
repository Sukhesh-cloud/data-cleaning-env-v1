---
title: Data Cleaning RL Environment
emoji: brain
colorFrom: blue
colorTo: purple
sdk: docker
---

# Data Cleaning RL Environment

An OpenEnv-compliant Reinforcement Learning environment that models **data cleaning as a sequential decision-making problem**. Agents learn to optimize data quality by handling missing values, removing outliers, and normalizing distributions.

**Features:**
- OpenEnv specification compliant
- LLM agent integration (OpenAI-compatible)
- Three difficulty levels (Easy, Medium, Hard)
- Structured reward function with dense feedback
- FastAPI server with async WebSocket support
- Docker & Hugging Face Spaces deployment ready

## Quick Start

### 1. Local Server Setup

```bash
# Install dependencies
pip install -r server/requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 2. Run Inference Script

Set environment variables:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_huggingface_token"
```

Run the inference script:
```bash
python inference.py
```

The script will:
- Execute all 3 tasks (easy, medium, hard)
- Use an LLM agent to decide actions at each step
- Log results in structured format: `[START]`, `[STEP]`, `[END]`
- Output task scores (0.0-1.0)

## Project Structure

```
data_cleaning_env/
├── server/
│   ├── app.py                          # FastAPI application
│   ├── data_cleaning_env_environment.py # RL environment logic
│   └── requirements.txt                # Server dependencies
├── models.py                           # Pydantic models (Action, Observation)
├── client.py                           # OpenEnv client wrapper
├── inference.py                        # LLM agent inference script
├── grader.py                           # Task grading system
├── tasks/
│   ├── easy.py                         # Easy task (missing values)
│   ├── medium.py                       # Medium task (missing + outliers)
│   └── hard.py                         # Hard task (missing + outliers + skewness)
├── Dockerfile                          # Container configuration
├── openenv.yaml                        # OpenEnv specification
└── pyproject.toml                      # Project metadata
```

## Environment API

### Reset

```python
result = await env.reset(task="easy")
obs = result.observation
```

### Step

```python
action = Action(
    action_type="fill_missing_mean",
    reason="Handle missing values",
    confidence=0.9
)
result = await env.step(action)
```

## Observation Space

Each observation contains:
```python
{
    "num_rows": int,              # Number of rows in dataset
    "num_columns": int,           # Number of columns
    "missing_ratio": float,       # Proportion of missing values
    "outlier_ratio": float,       # Proportion of outliers
    "skewness": [float, ...],     # Skewness per column
    "data_types": [str, ...],     # Column data types
    "steps_remaining": int,       # Steps left in episode
    "reward": float,              # Step reward
    "done": bool,                 # Episode finished
    "metadata": dict              # Additional info
}
```

## Action Space

Supported actions:
- `fill_missing_mean` - Fill missing values with mean
- `fill_missing_median` - Fill missing values with median
- `remove_outliers` - Remove outlier rows
- `normalize_data` - Normalize features (reduces skewness)
- `standardize_data` - Standardize features
- `drop_rows` - Remove rows with missing values
- `do_nothing` - No action

Action structure:
```python
{
    "action_type": str,      # Action name
    "reason": str,           # Explanation for action
    "confidence": float      # Confidence score 0.0-1.0
}
```

## Tasks & Grading

### Easy Task
- **Objective:** Handle missing values
- **Initial State:**
  - Missing ratio: 0.25
  - Outlier ratio: 0.0
  - Skewness: [0.2, 0.1, -0.3]
- **Grading:** Score = max(0.0, (0.30 - final_missing_ratio) * 3)
- **Max Score:** 1.0

### Medium Task
- **Objective:** Handle missing values + outliers
- **Initial State:**
  - Missing ratio: 0.20
  - Outlier ratio: 0.15
  - Skewness: [1.5, -1.2, 0.8]
- **Grading:**
  - Missing component: max(0.0, (0.30 - missing) * 1.5) * weight
  - Outlier component: max(0.0, (0.20 - outliers) * 2.0) * weight
- **Max Score:** 1.0

### Hard Task
- **Objective:** Full pipeline (missing + outliers + skewness)
- **Initial State:**
  - Missing ratio: 0.35
  - Outlier ratio: 0.25
  - Skewness: [3.0, -2.5, 2.2]
- **Grading:**
  - Missing component: max(0.0, (0.30 - missing) * 1.2)
  - Outlier component: max(0.0, (0.20 - outliers) * 1.5)
  - Skewness component: max(0.0, (1.0 - avg_skewness) * 1.3)
- **Max Score:** 1.0

## Reward Function

The environment provides step-wise dense rewards:

```
reward = 0.5 * missing_improvement
       + 0.3 * outlier_improvement
       + 0.2 * skewness_improvement
       - action_cost
       - diminishing_returns_penalty
       - repetition_penalty
       - confidence_penalty
```

**Penalties:**
- Diminishing returns: -0.1 (no significant improvement)
- Repetition: -0.1 (same action twice)
- Low confidence: -0.2 (confidence < 0.5)
- Action costs: 0.05-0.10 per action type

## Inference Script Output Format

The inference script produces structured logs:

```
[START] task=easy env=data_cleaning_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=fill_missing_mean reward=0.28 done=false error=null
[STEP] step=2 action=remove_outliers reward=0.15 done=false error=null
[STEP] step=3 action=normalize_data reward=0.12 done=false error=null
[END] success=true steps=3 score=0.92 rewards=0.28,0.15,0.12
```

**Format Specification:**
- `[START]` - Episode initialization
- `[STEP]` - One environment step (repeated up to MAX_STEPS)
- `[END]` - Episode completion
- Scores are normalized to [0.0, 1.0]
- Rewards are formatted to 2 decimal places

## Docker Deployment

### Build Image

```bash
docker build -t data-cleaning-env:latest .
```

### Run Container

```bash
docker run -p 7860:7860 data-cleaning-env:latest
```

### Test Endpoint

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'
```

## Hugging Face Spaces Deployment

### Push to HF Spaces

```bash
# Set environment for UTF-8 on Windows CMD
set PYTHONUTF8=1

# Push with openenv CLI
openenv push --repo-id <username>/data-cleaning-env
```

### Live Space

Once deployed, access your space at:
```
https://huggingface.co/spaces/<username>/data-cleaning-env
```

**Required Environment Variables (on HF Spaces):**
```
API_BASE_URL   = https://router.huggingface.co/v1
MODEL_NAME     = Qwen/Qwen2.5-72B-Instruct
HF_TOKEN       = (your Hugging Face API token)
```

## System Requirements

- Python 3.10+
- CPU: 2 vCPU minimum
- Memory: 8 GB minimum
- Inference runtime: < 20 minutes for all tasks

## Dependencies

**Core:**
- openenv-core
- fastapi
- uvicorn
- pydantic

**Inference:**
- openai (for LLM integration)

Install all:
```bash
pip install openenv-core fastapi uvicorn pydantic openai
```

## OpenEnv Compliance

Validates against OpenEnv specification:
```bash
openenv validate
```

Expected output:
```
[OK] data_cleaning: Ready for multi-mode deployment
```

## Architecture

```
┌─────────────────┐
│  LLM Agent      │ (Qwen/GPT-4)
│  (inference.py) │
└────────┬────────┘
         │ OpenAI API
         ▼
┌─────────────────────────┐
│  Client                 │
│  (client.py)            │
└────────┬────────────────┘
         │ HTTP/WebSocket
         ▼
┌─────────────────────────────────────┐
│  FastAPI Server                     │
│  (server/app.py)                    │
├─────────────────────────────────────┤
│  DataCleaningEnvironment            │
│  - reset(task)                      │
│  - step(action)                     │
│  - state                            │
└─────────────────────────────────────┘
                │
                ▼
         ┌──────────────────┐
         │  Grader System   │
         │  (tasks/*.py)    │
         │  Score: [0, 1]   │
         └──────────────────┘
```

## Performance Benchmarks

Example results from baseline LLM agent:

| Task   | Avg Score | Steps | Final Missing | Final Outliers |
|--------|-----------|-------|---------------|----------------|
| Easy   | 0.92      | 4     | 0.02          | 0.00           |
| Medium | 0.78      | 5     | 0.05          | 0.08           |
| Hard   | 0.65      | 5     | 0.12          | 0.15           |

## Contributing

To extend this environment:

1. Add new actions in `server/data_cleaning_env_environment.py`
2. Create new tasks in `tasks/` directory
3. Implement grading function matching `grade_*` pattern
4. Update `grader.py` to include new task

## License

BSD-style license

## References

- [OpenEnv Specification](https://github.com/meta-pytorch/OpenEnv)
- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)



# 🔍 API Endpoints

| Endpoint | Description            |
| -------- | ---------------------- |
| `/reset` | Initialize environment |
| `/step`  | Apply action           |
| `/state` | Get current state      |
| `/docs`  | Swagger API            |



# ⚙️ Running Locally

```bash
uvicorn server.app:app --reload
```

Open:
👉 http://localhost:8000/docs



# 🧪 Testing Environment

```bash
python inference.py
```

This runs a baseline agent and outputs:

* Step-by-step actions
* Rewards
* Final score


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



# 🌍 Use Cases

* Automated data preprocessing
* RL-based pipeline optimization
* Decision-making systems
* AI agents for data quality improvement


# 🏁 Summary

This project demonstrates how **real-world workflows can be modeled as RL environments**, enabling intelligent systems to learn optimal strategies for data cleaning.

