from uuid import uuid4
import random

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

    # 🔄 RESET
    def reset(self, task: str = "easy") -> Observation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_step = 0

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
        )

    # 🎮 STEP FUNCTION
    def step(self, action: Action) -> Observation:  # type: ignore[override]
        self._state.step_count += 1
        self.current_step += 1

        reward = 0

        # Apply actions
        if action.action_type == "fill_missing_mean":
            self.dataset["missing_ratio"] *= 0.7
            reward += 0.3

        elif action.action_type == "remove_outliers":
            self.dataset["outlier_ratio"] *= 0.6
            reward += 0.3

        elif action.action_type == "normalize_data":
            self.dataset["skewness"] = [s * 0.5 for s in self.dataset["skewness"]]
            reward += 0.2

        elif action.action_type == "do_nothing":
            reward -= 0.05

        else:
            reward -= 0.2  # invalid action

        # 🧠 Human-in-the-loop simulation
        if action.confidence < 0.5:
            reward -= 0.2

        # done condition
        done = self.current_step >= self.max_steps

        return Observation(
            num_rows=self.dataset["num_rows"],
            num_columns=self.dataset["num_columns"],
            missing_ratio=round(self.dataset["missing_ratio"], 2),
            outlier_ratio=round(self.dataset["outlier_ratio"], 2),
            skewness=self.dataset["skewness"],
            data_types=self.dataset["data_types"],
            steps_remaining=self.max_steps - self.current_step,
        )

    # 📦 STATE
    @property
    def state(self) -> State:
        return self._state