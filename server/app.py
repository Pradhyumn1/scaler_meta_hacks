from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
from env import CustomerSupportEnv, Action, Observation, State

app = FastAPI(title="OpenEnv - Customer Support")
env = CustomerSupportEnv()

class ResetResponse(BaseModel):
    observation: Observation

class StepRequest(BaseModel):
    action: Action

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

class StateResponse(BaseModel):
    state: State

@app.post("/reset", response_model=ResetResponse)
def reset_env():
    obs = env.reset()
    return ResetResponse(observation=obs)

@app.post("/step", response_model=StepResponse)
def step_env(request: StepRequest):
    obs, reward, done, info = env.step(request.action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)

@app.get("/state", response_model=StateResponse)
def state_env():
    state = env.get_state()
    return StateResponse(state=state)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
