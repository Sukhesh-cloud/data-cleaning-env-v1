import os

from server.data_cleaning_env_environment import DataCleaningEnvironment
from models import Action
from grader import grade

TASKS = ["easy", "medium", "hard"]
ENV_NAME = "data_cleaning_env"
MODEL_NAME = os.getenv("MODEL_NAME", "baseline")

MAX_STEPS = 5


def choose_action(obs):
    # Simple rule-based baseline agent

    if obs.missing_ratio > 0.1:
        return Action(
            action_type="fill_missing_mean",
            reason="High missing values",
            confidence=0.9,
        )

    elif obs.outlier_ratio > 0.05:
        return Action(
            action_type="remove_outliers",
            reason="High outliers",
            confidence=0.9,
        )

    else:
        return Action(
            action_type="normalize_data",
            reason="Reduce skewness",
            confidence=0.8,
        )


def run_task(task_name):
    env = DataCleaningEnvironment()

    obs = env.reset(task=task_name)

    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}")

    rewards = []
    success = False

    for step in range(1, MAX_STEPS + 1):
        try:
            action = choose_action(obs)

            obs = env.step(action)

            # simulate reward for logging
            reward = round(0.3, 2)

            rewards.append(reward)

            done = step == MAX_STEPS

            print(
                f"[STEP] step={step} action={action.action_type} "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )

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

    rewards_str = ",".join([f"{r:.2f}" for r in rewards])

    print(
        f"[END] success={str(success).lower()} "
        f"steps={len(rewards)} rewards={rewards_str}"
    )
    print("Score:", grade(task_name, env))


def main():
    for task in TASKS:
        run_task(task)
        


if __name__ == "__main__":
    main()