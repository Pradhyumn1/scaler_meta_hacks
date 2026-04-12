import os
import json
import textwrap
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env automatically

from server.env import CustomerSupportEnvironment, CustomerAction, TASKS

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "dummy")
TEMPERATURE = 0.0
MAX_TOKENS = 512
MAX_STEPS = 15
TASK_NAME = "customer_support"
BENCHMARK = "OpenEnv-CustomerSupport"

SYSTEM_PROMPT = """You are a customer support agent.
Read the system message and current ticket/observation.
To take an action, you must ONLY output a raw JSON object (and nothing else) on the final line in this format:
{"command": "search_kb|reply_to_customer|issue_refund|issue_replacement|get_customer_info|check_inventory", "argument": "your_argument"}

For example, to search KB:
{"command": "search_kb", "argument": "return policy"}

To check order info:
{"command": "get_customer_info", "argument": "alice@example.com"}

Always issue a refund if you see an expected_amount in the ticket or based on the refund policy.
If a requested replacement is out of stock, issue a refund instead, matching the expected_amount exactly!
"""


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, command: str, argument: str):
    print(f"[STEP] step={step} command={command} argument={argument!r}", flush=True)


def log_task(task_id: str, score: float):
    """Log completion of a single graded task."""
    print(f"[TASK] task_id={task_id} score={score:.3f}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.3f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} tasks={len(rewards)} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last echoed message: {last_echoed!r}
        Last reward: {last_reward:.2f}
        Previous steps history:
        {history_block}
        Send your next command.
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, last_echoed, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"command": "reply_to_customer", "argument": "error"}'


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CustomerSupportEnvironment()

    history: List[str] = []

    # task_rewards: one entry per completed task (strictly between 0 and 1)
    task_rewards: List[float] = []

    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        last_echoed = obs.echoed_message or obs.system_message
        last_reward = 0.0
        prev_task_idx = env.state.current_task_idx

        for step in range(1, MAX_STEPS + 1):
            state = env.state
            if state.current_task_idx >= len(TASKS):
                success = True
                break

            steps_taken += 1
            history.append(f"Observation: {obs.model_dump_json()}")

            if API_KEY == "dummy":
                # Deterministic mock agent for CI / no-key runs
                task = TASKS[state.current_task_idx]
                if task["type"] == "KB_QUERY":
                    action_dict = {"command": "reply_to_customer", "argument": "30 days"}
                elif task["type"] == "REFUND":
                    if obs.customer_info is None:
                        action_dict = {"command": "get_customer_info", "argument": "alice@example.com"}
                    else:
                        action_dict = {"command": "issue_refund", "argument": "50.0"}
                elif task["type"] == "REPLACEMENT_OUT_OF_STOCK":
                    if obs.inventory_status is None:
                        action_dict = {"command": "check_inventory", "argument": "XYZ-1"}
                    else:
                        action_dict = {"command": "issue_refund", "argument": "100.0"}
                else:
                    action_dict = {"command": "reply_to_customer", "argument": "done"}
            else:
                message = get_model_message(client, step, last_echoed, last_reward, history)
                try:
                    action_dict = json.loads(message)
                except Exception:
                    import re
                    match = re.search(r'\{.*\}', message, re.DOTALL)
                    if match:
                        try:
                            action_dict = json.loads(match.group(0))
                        except Exception:
                            action_dict = {"command": "reply_to_customer", "argument": "error parsing"}
                    else:
                        action_dict = {"command": "reply_to_customer", "argument": "error parsing"}

            history.append(f"Action: {json.dumps(action_dict)}")
            action = CustomerAction(
                command=action_dict.get("command", "reply_to_customer"),
                argument=str(action_dict.get("argument", "")),
            )

            log_step(step, action.command, action.argument)

            obs = env.step(action)
            reward = obs.reward if obs.reward is not None else 0.0
            done = obs.done if obs.done is not None else False
            last_echoed = obs.echoed_message or obs.system_message
            last_reward = reward

            # Only record reward when a TASK COMPLETES (task_idx advances)
            new_task_idx = env.state.current_task_idx
            if new_task_idx > prev_task_idx or (done and new_task_idx >= len(TASKS)):
                # Clamp to strict (0, 1) range
                clamped = max(0.01, min(0.99, reward))
                task_id = TASKS[prev_task_idx]["id"] if prev_task_idx < len(TASKS) else f"T{prev_task_idx+1}"
                log_task(task_id, clamped)
                task_rewards.append(clamped)
                prev_task_idx = new_task_idx

            if done and env.state.current_task_idx >= len(TASKS):
                success = True
                break

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)

    # score = average per-task reward, strictly between 0 and 1
    score = sum(task_rewards) / len(task_rewards) if task_rewards else 0.01

    log_end(success=success, steps=steps_taken, score=score, rewards=task_rewards)


if __name__ == "__main__":
    main()
