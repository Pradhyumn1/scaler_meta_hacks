"""
customer-support-env server/app.py
Uses the official OpenEnv create_fastapi_app so that all standard endpoints
(/health, /metadata, /schema, /mcp, /reset, /step, /state) are registered.
"""
import sys
import os

# Ensure the repo root is on sys.path so 'env' can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app
from env import CustomerSupportEnvironment, CustomerAction, CustomerObservation

app = create_fastapi_app(
    env=CustomerSupportEnvironment,       # factory callable
    action_cls=CustomerAction,
    observation_cls=CustomerObservation,
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
