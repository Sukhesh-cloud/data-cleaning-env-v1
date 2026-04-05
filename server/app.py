
try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e


from models import Action,Observation
from server.data_cleaning_env_environment import DataCleaningEnvironment


# Create the app with web interface and README integration
app = create_app(
    DataCleaningEnvironment,
    Action,
    Observation,
    env_name="data_cleaning_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__=="__main__":
    main()
