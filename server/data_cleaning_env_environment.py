from uuid import uuid4

from models import Observation, Action
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from tasks.easy import get_easy_dataset
from tasks.medium import get_medium_dataset
from tasks.hard import get_hard_dataset


class DataCleaningEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.max_steps = 5
        self.current_step = 0
        self.dataset = None
        self.last_action = None

    # 🔄 RESET
    def reset(self, task: str = "easy") -> Observation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_step = 0
        self.last_action = None

        if task == "easy":
            base = get_easy_dataset()
        elif task == "medium":
            base = get_medium_dataset()
        else:
            base = get_hard_dataset()

        self.dataset = {
            "num_rows": 1000,
            "num_columns": 10,
            **base,
            "data_types": ["numeric", "categorical"],
        }

        return Observation(
            num_rows=self.dataset["num_rows"],
            num_columns=self.dataset["num_columns"],
            missing_ratio=self.dataset["missing_ratio"],
            outlier_ratio=self.dataset["outlier_ratio"],
            skewness=self.dataset["skewness"],
            data_types=self.dataset["data_types"],
            steps_remaining=self.max_steps,
            reward=0.0,
            done=False,
            metadata={"step": 0}
        )

    # 🧠 ADVANCED REWARD FUNCTION
    def _compute_reward(self, old_state, new_state, action_type):

        delta_missing = old_state["missing_ratio"] - new_state["missing_ratio"]
        delta_outlier = old_state["outlier_ratio"] - new_state["outlier_ratio"]

        old_skew = sum(abs(s) for s in old_state["skewness"])
        new_skew = sum(abs(s) for s in new_state["skewness"])
        delta_skew = old_skew - new_skew

        reward = (
            0.5 * delta_missing +
            0.3 * delta_outlier +
            0.2 * (delta_skew / len(old_state["skewness"]))
        )

        # 🔥 diminishing returns
        if delta_missing < 0.01:
            reward -= 0.1

        if delta_outlier < 0.01:
            reward -= 0.1

        # 🔥 repetition penalty
        if self.last_action == action_type:
            reward -= 0.1

        # 🔥 action cost (innovation)
        action_cost = {
            "fill_missing_mean": 0.05,
            "remove_outliers": 0.08,
            "normalize_data": 0.07,
            "fill_missing_median": 0.06,
            "drop_rows": 0.1,
            "standardize_data": 0.08,
        }

        reward -= action_cost.get(action_type, 0.05)

        return round(reward, 3)

    # 🎮 STEP FUNCTION
    def step(self, action: Action) -> Observation:  # type: ignore[override]
        self._state.step_count += 1
        self.current_step += 1

        action_type = getattr(action, "action_type", None)
        confidence = getattr(action, "confidence", 0.5)

        old_state = self.dataset.copy()

        # 🔧 APPLY ACTIONS (NO REWARD HERE)
        if action_type == "fill_missing_mean":
            self.dataset["missing_ratio"] *= 0.7

        elif action_type == "remove_outliers":
            self.dataset["outlier_ratio"] *= 0.6

        elif action_type == "normalize_data":
            self.dataset["skewness"] = [s * 0.5 for s in self.dataset["skewness"]]

        elif action_type == "fill_missing_median":
            self.dataset["missing_ratio"] *= 0.75

        elif action_type == "drop_rows":
            self.dataset["num_rows"] = int(self.dataset["num_rows"] * 0.9)
            self.dataset["missing_ratio"] *= 0.5

        elif action_type == "standardize_data":
            self.dataset["skewness"] = [s * 0.3 for s in self.dataset["skewness"]]

        elif action_type == "do_nothing":
            pass

        else:
            # invalid action penalty
            reward = -0.2
            done = True
            return Observation(**self._get_observation(reward, done))

        # 🧠 COMPUTE REWARD
        reward = self._compute_reward(old_state, self.dataset, action_type)

        # 🧠 human-in-loop penalty
        if confidence < 0.5:
            reward -= 0.2

        # track last action
        self.last_action = action_type

        done = self.current_step >= self.max_steps

        return Observation(**self._get_observation(reward, done))

    # 📦 HELPER
    def _get_observation(self, reward, done):
        return {
            "num_rows": self.dataset["num_rows"],
            "num_columns": self.dataset["num_columns"],
            "missing_ratio": round(self.dataset["missing_ratio"], 2),
            "outlier_ratio": round(self.dataset["outlier_ratio"], 2),
            "skewness": self.dataset["skewness"],
            "data_types": self.dataset["data_types"],
            "steps_remaining": self.max_steps - self.current_step,
            "reward": round(reward, 3),
            "done": done,
            "metadata": {
                "step": self.current_step,
                "last_action": self.last_action
            }
        }

    # 📦 STATE
    @property
    def state(self) -> State:
        return self._state