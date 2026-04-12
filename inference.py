import asyncio
import os
import json
import sys
import time
from typing import List, Optional

from openai import OpenAI

from models import Action
from grader import grade
from client import DataCleaningEnv

# ===== CONFIGURATION =====
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASKS = ["easy", "medium", "hard"]
BENCHMARK = "data_cleaning_env"
MAX_STEPS = 5
TEMPERATURE = 0.2
MAX_TOKENS = 150
CONNECTION_TIMEOUT = 30
CONNECTION_RETRIES = 3


# ===== LOGGING FUNCTIONS =====
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


def debug_log(msg: str) -> None:
    """Print debug messages to stderr to avoid interfering with stdout logging"""
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)


# ===== ENVIRONMENT CONNECTION =====
async def get_environment(timeout: int = CONNECTION_TIMEOUT) -> Optional[DataCleaningEnv]:
    """
    Connect to environment with retry logic.
    Try Docker first, then localhost, then fail gracefully.
    """
    env = None
    
    # Strategy 1: Docker Image
    if IMAGE_NAME:
        try:
            debug_log(f"Attempting to start Docker image: {IMAGE_NAME}")
            env = await asyncio.wait_for(
                DataCleaningEnv.from_docker_image(IMAGE_NAME),
                timeout=timeout
            )
            debug_log("Docker environment started successfully")
            return env
        except asyncio.TimeoutError:
            debug_log(f"Docker startup timed out after {timeout}s")
        except Exception as e:
            debug_log(f"Docker strategy failed: {e}")
    
    # Strategy 2: Local Server with Retries
    try:
        debug_log("Attempting to connect to local server at http://localhost:8000")
        for attempt in range(CONNECTION_RETRIES):
            try:
                env = DataCleaningEnv(base_url="http://localhost:8000")
                # Test connection with a simple reset
                test_result = await asyncio.wait_for(
                    env.reset(task="easy"),
                    timeout=10
                )
                debug_log(f"Successfully connected to local server on attempt {attempt + 1}")
                # Reset again to get clean state
                await env.reset(task="easy")
                return env
            except (asyncio.TimeoutError, ConnectionError) as e:
                wait_time = 2 ** attempt  # Exponential backoff
                debug_log(f"Attempt {attempt + 1}/{CONNECTION_RETRIES} failed: {e}. Retrying in {wait_time}s...")
                if attempt < CONNECTION_RETRIES - 1:
                    await asyncio.sleep(wait_time)
    except Exception as e:
        debug_log(f"Local connection strategy failed: {e}")
    
    return None


# ===== ACTION SELECTION =====
def choose_action(client: OpenAI, obs, history: List[str]) -> Action:
    """
    Use LLM to choose next action.
    Falls back to rule-based if LLM unavailable.
    """
    # Rule-based fallback strategy
    def get_fallback_action():
        # Smart heuristic: choose action based on largest problem
        if obs.missing_ratio > 0.1:
            return "fill_missing_mean"
        elif obs.outlier_ratio > 0.1:
            return "remove_outliers"
        elif any(abs(s) > 1.0 for s in obs.skewness):
            return "normalize_data"
        else:
            return "do_nothing"
    
    prompt = f"""
You are a data cleaning agent. Analyze the current dataset state and choose the best action.

Current State:
- Missing ratio: {obs.missing_ratio:.2f}
- Outlier ratio: {obs.outlier_ratio:.2f}
- Skewness: {obs.skewness}
- Previous actions: {', '.join(history) if history else 'none'}

Available actions:
1. fill_missing_mean - Fill missing with mean
2. remove_outliers - Remove outlier rows
3. normalize_data - Reduce skewness
4. fill_missing_median - Fill missing with median
5. standardize_data - Standardize features
6. drop_rows - Drop rows with missing
7. do_nothing - No action

Respond in JSON format:
{{
  "action_type": "action_name",
  "reason": "brief explanation",
  "confidence": 0.5
}}
"""

    try:
        if not client or not API_KEY:
            debug_log("LLM client not available, using rule-based fallback")
            return Action(
                action_type=get_fallback_action(),
                reason="rule-based fallback",
                confidence=0.7
            )

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=15
        )
        
        response_text = completion.choices[0].message.content.strip()
        parsed = json.loads(response_text)
        
        return Action(
            action_type=parsed.get("action_type", get_fallback_action()),
            reason=parsed.get("reason", ""),
            confidence=float(parsed.get("confidence", 0.7))
        )
    except json.JSONDecodeError as e:
        debug_log(f"JSON parsing error: {e}")
        return Action(
            action_type=get_fallback_action(),
            reason="json-parse-fallback",
            confidence=0.6
        )
    except Exception as e:
        debug_log(f"LLM request failed: {e}, using rule-based fallback")
        return Action(
            action_type=get_fallback_action(),
            reason="exception-fallback",
            confidence=0.6
        )


# ===== MAIN TASK RUNNER =====
async def run_task(task_name: str, client: Optional[OpenAI], env: Optional[DataCleaningEnv]) -> None:
    """
    Run a single task with comprehensive error handling.
    """
    if not env:
        debug_log(f"Skipping {task_name}: No environment available")
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        try:
            result = await asyncio.wait_for(env.reset(task=task_name), timeout=15)
            obs = result.observation
            env.last_observation = obs
        except asyncio.TimeoutError:
            raise Exception("Environment reset timed out")
        except Exception as e:
            raise Exception(f"Failed to reset environment: {e}")

        # Run steps
        for step in range(1, MAX_STEPS + 1):
            try:
                # Choose action
                action = choose_action(client, obs, history)
                
                # Take step
                try:
                    result = await asyncio.wait_for(
                        env.step(action),
                        timeout=15
                    )
                    obs = result.observation
                    env.last_observation = obs
                except asyncio.TimeoutError:
                    raise Exception("Environment step timed out")

                reward = getattr(result, 'reward', 0.0) or 0.0
                done = getattr(result, 'done', False)

                rewards.append(reward)
                steps_taken = step
                last_error = None

                log_step(step=step, action=action.action_type, reward=reward, done=done, error=None)
                history.append(action.action_type)

                if done:
                    break

            except Exception as e:
                debug_log(f"Step {step} failed: {e}")
                last_error = str(e)
                log_step(step=step, action="error", reward=0.0, done=True, error=last_error)
                break

        # Grade the task
        try:
            score = grade(task_name, env)
            success = score >= 0.5
        except Exception as e:
            debug_log(f"Grading failed: {e}")
            score = 0.0
            success = False

    except Exception as e:
        debug_log(f"Task {task_name} encountered critical error: {e}")
        last_error = str(e)
        success = False
        score = 0.0
    
    finally:
        # Log end result
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ===== MAIN ENTRY POINT =====
async def main() -> None:
    """Main inference loop"""
    debug_log("="*60)
    debug_log("Starting Data Cleaning Environment Inference")
    debug_log(f"Model: {MODEL_NAME}")
    debug_log(f"API Base URL: {API_BASE_URL}")
    debug_log(f"Image Name: {IMAGE_NAME or 'None'}")
    debug_log(f"Tasks: {TASKS}")
    debug_log("="*60)

    # Validate setup
    if not API_KEY:
        debug_log("WARNING: No API key found (HF_TOKEN or API_KEY)")
        debug_log("LLM-based action selection will fail. Using rule-based fallback.")

    # Initialize LLM client
    client = None
    try:
        if API_KEY:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            debug_log("OpenAI client initialized successfully")
    except Exception as e:
        debug_log(f"Failed to initialize OpenAI client: {e}")
        client = None

    # Get environment
    env = await get_environment(timeout=CONNECTION_TIMEOUT)
    if not env:
        debug_log("FATAL: Could not connect to environment via any method")
        print("[END] success=false steps=0 score=0.0 rewards=", flush=True)
        return

    # Run all tasks
    try:
        for task in TASKS:
            await run_task(task, client, env)
            await asyncio.sleep(1)  # Small delay between tasks
    except Exception as e:
        debug_log(f"Critical error in main loop: {e}")
    finally:
        # Cleanup
        try:
            await env.close()
            debug_log("Environment closed successfully")
        except Exception as e:
            debug_log(f"Error closing environment: {e}")

    debug_log("Inference complete")


# ===== SCRIPT ENTRY =====
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        debug_log("Inference interrupted by user")
        sys.exit(0)
    except Exception as e:
        debug_log(f"Unhandled exception in main: {e}")
        print("[END] success=false steps=0 score=0.0 rewards=", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())