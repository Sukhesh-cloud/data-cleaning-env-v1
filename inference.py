import asyncio
import os
import json
from typing import List, Optional

from openai import OpenAI

from models import Action
from grader import grade
from client import DataCleaningEnv

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")  # If you are using docker image
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
print(os.getenv("HF_TOKEN"))
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASKS = ["easy", "medium", "hard"]
BENCHMARK = "data_cleaning_env"
MAX_STEPS = 5
TEMPERATURE = 0.2
MAX_TOKENS = 150


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def choose_action(client: OpenAI, obs, history: List[str]) -> Action:
    prompt = f"""
You are a data cleaning agent.
Rules:
- Avoid repeating same action
- Choose action based on biggest problem
- Improve dataset step by step
Dataset:
- Missing ratio: {obs.missing_ratio}
- Outlier ratio: {obs.outlier_ratio}
- Skewness: {obs.skewness}

Previous actions: {', '.join(history)}

Choose ONE action from:
fill_missing_mean, remove_outliers, normalize_data, fill_missing_median, standardize_data, drop_rows, do_nothing

Respond ONLY in JSON:
{{
  "action_type": "...",
  "reason": "...",
  "confidence": 0.0-1.0
}}
"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        parsed = json.loads(completion.choices[0].message.content)

        return Action(
            action_type=parsed["action_type"],
            reason=parsed.get("reason", ""),
            confidence=float(parsed.get("confidence", 0.8))
        )
    except Exception as e:
        print(f"[DEBUG] Model request failed: {e}", flush=True)
        return Action(
            action_type="do_nothing",
            reason="fallback",
            confidence=0.5
        )


async def run_task(task_name: str, client: OpenAI) -> None:
    if IMAGE_NAME:
        env = await DataCleaningEnv.from_docker_image(IMAGE_NAME)
    else:
        env = DataCleaningEnv(base_url="http://localhost:8000")  # or HF space URL if needed

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task_name)
        obs = result.observation
        env.last_observation = obs

        for step in range(1, MAX_STEPS + 1):
            action = choose_action(client, obs, history)

            result = await env.step(action)
            obs = result.observation
            env.last_observation = obs

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action.action_type, reward=reward, done=done, error=error)

            history.append(action.action_type)

            if done:
                break

        score = grade(task_name, env)
        success = score >= 0.5  # adjust threshold if needed

    except Exception as e:
        print(f"[DEBUG] Task {task_name} failed: {e}", flush=True)
        success = False
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in TASKS:
        await run_task(task, client)


if __name__ == "__main__":
    asyncio.run(main())