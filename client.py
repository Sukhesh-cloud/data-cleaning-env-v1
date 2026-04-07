# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# """Data Cleaning Env Environment Client."""

# from typing import Dict

# from openenv.core import EnvClient
# from openenv.core.client_types import StepResult
# from openenv.core.env_server.types import State

# from models import Action, Observation


# class DataCleaningEnv(
#     EnvClient[Action, Observation, State]
# ):
#     """
#     Client for the Data Cleaning Env Environment.

#     This client maintains a persistent WebSocket connection to the environment server,
#     enabling efficient multi-step interactions with lower latency.
#     Each client instance has its own dedicated environment session on the server.

#     Example:
#         >>> # Connect to a running server
#         >>> with DataCleaningEnv(base_url="http://localhost:8000") as client:
#         ...     result = client.reset()
#         ...     print(result.observation.echoed_message)
#         ...
#         ...     result = client.step(DataCleaningAction(message="Hello!"))
#         ...     print(result.observation.echoed_message)

#     Example with Docker:
#         >>> # Automatically start container and connect
#         >>> client = DataCleaningEnv.from_docker_image("data_cleaning_env-env:latest")
#         >>> try:
#         ...     result = client.reset()
#         ...     result = client.step(DataCleaningAction(message="Test"))
#         ... finally:
#         ...     client.close()
#     """

#     def _step_payload(self, action: Action) -> Dict:
#         """
#         Convert DataCleaningAction to JSON payload for step message.

#         Args:
#             action: DataCleaningAction instance

#         Returns:
#             Dictionary representation suitable for JSON encoding
#         """
#         return {
#             "message": action.message,
#         }

#     def _parse_result(self, payload: Dict) -> StepResult[Observation]:
#         """
#         Parse server response into StepResult[DataCleaningObservation].

#         Args:
#             payload: JSON response data from server

#         Returns:
#             StepResult with DataCleaningObservation
#         """
#         obs_data = payload.get("observation", {})
#         observation = Observation(
#             echoed_message=obs_data.get("echoed_message", ""),
#             message_length=obs_data.get("message_length", 0),
#             done=payload.get("done", False),
#             reward=payload.get("reward"),
#             metadata=obs_data.get("metadata", {}),
#         )

#         return StepResult(
#             observation=observation,
#             reward=payload.get("reward"),
#             done=payload.get("done", False),
#         )

#     def _parse_state(self, payload: Dict) -> State:
#         """
#         Parse server response into State object.

#         Args:
#             payload: JSON response from state request

#         Returns:
#             State object with episode_id and step_count
#         """
#         return State(
#             episode_id=payload.get("episode_id"),
#             step_count=payload.get("step_count", 0),
#         )

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import Action, Observation


class DataCleaningEnv(
    EnvClient[Action, Observation, State]
):

    # 🔧 SEND ACTION TO SERVER
    def _step_payload(self, action: Action) -> Dict:
        return {
            "action_type": action.action_type,
            "reason": action.reason,
            "confidence": action.confidence,
        }

    # 🔧 PARSE RESPONSE FROM SERVER
    def _parse_result(self, payload: Dict) -> StepResult[Observation]:

        obs_data = payload.get("observation", {})

        observation = Observation(
            num_rows=obs_data.get("num_rows", 0),
            num_columns=obs_data.get("num_columns", 0),
            missing_ratio=obs_data.get("missing_ratio", 0.0),
            outlier_ratio=obs_data.get("outlier_ratio", 0.0),
            skewness=obs_data.get("skewness", []),
            data_types=obs_data.get("data_types", []),
            steps_remaining=obs_data.get("steps_remaining", 0),
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    # 🔧 PARSE STATE
    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )