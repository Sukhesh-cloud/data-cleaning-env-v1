import os
import json
from openai import OpenAI

from models import Action
from grader import grade
from client import DataCleaningEnv   # ✅ CHANGE THIS

TASKS = ["easy", "medium", "hard"]
ENV_NAME = "data_cleaning_env"

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

MAX_STEPS = 5

# ✅ REQUIRED CLIENT (fixes your failure)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


# 🔥 REPLACE RULE-BASED WITH LLM
def choose_action(obs):

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

Choose ONE action from:
fill_missing_mean, remove_outliers, normalize_data, fill_missing_median, standardize_data

Respond ONLY in JSON:
{{
  "action_type": "...",
  "reason": "...",
  "confidence": 0.0-1.0
}}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    try:
        parsed = json.loads(response.choices[0].message.content)

        return Action(
            action_type=parsed["action_type"],
            reason=parsed.get("reason", ""),
            confidence=float(parsed.get("confidence", 0.8))
        )

    except:
        return Action(
            action_type="fill_missing_mean",
            reason="fallback",
            confidence=0.5
        )


async def run_task(task_name):

    env = DataCleaningEnv(base_url="http://localhost:8000")

    result = await env.reset(task=task_name)
    obs = result.observation

    print(f"[START] task={task_name} env=data_cleaning_env model={MODEL_NAME}")

    rewards = []
    success = False
    steps = 0

    for step in range(1, MAX_STEPS + 1):
        try:
            action = choose_action(obs)

            result = await env.step(action)
            obs = result.observation

            reward = round(obs.reward, 2)
            rewards.append(reward)

            done = obs.done

            print(
                f"[STEP] step={step} action={action.action_type} "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )
            env.last_observation = obs
            steps = step

            if done:
                score = grade(task_name, env)
                success = score > 0.5
                break

        except Exception as e:
            print(
                f"[STEP] step={step} action=error "
                f"reward=0.00 done=true error={str(e)}"
            )
            break

    await env.close()

    rewards_str = ",".join([f"{r:.2f}" for r in rewards])

    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} rewards={rewards_str}"
    )

    print("Score:", grade(task_name, env))

async def main():
    for task in TASKS:
        await run_task(task)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())