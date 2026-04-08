import json
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel

class Action(BaseModel):
    command: str
    argument: str

class Observation(BaseModel):
    echoed_message: str = ""
    system_message: str
    current_ticket: Optional[Dict[str, Any]] = None
    customer_info: Optional[Dict[str, Any]] = None
    inventory_status: Optional[Dict[str, Any]] = None
    command_result: Optional[str] = None
    knowledge_base_result: Optional[str] = None

class State(BaseModel):
    tasks_completed: int
    current_task_idx: int
    total_score: float

class CustomerSupportEnv:
    def __init__(self):
        self.state = State(tasks_completed=0, current_task_idx=0, total_score=0.0)
        self.tasks = [
            {"id": "T1", "type": "KB_QUERY", "desc": "Customer is asking about the general return policy.", "expected_action": "reply_to_customer", "required_content": "30 days"},
            {"id": "T2", "type": "REFUND", "desc": "Customer email alice@example.com order #1029 wants a refund. Item was damaged.", "email": "alice@example.com", "expected_action": "issue_refund", "expected_amount": 50.0},
            {"id": "T3", "type": "REPLACEMENT_OUT_OF_STOCK", "desc": "Customer email bob@example.com order #2041 wants a replacement for product XYZ-1. It is out of stock.", "email": "bob@example.com", "expected_action": "issue_refund", "expected_amount": 100.0}
        ]
        self.kb = {
            "return_policy": "Our return policy allows returns within 30 days of purchase for a full refund."
        }
        self.customers = {
            "alice@example.com": {"order_id": "1029", "product": "Widget A", "amount": 50.0},
            "bob@example.com": {"order_id": "2041", "product": "XYZ-1", "amount": 100.0}
        }
        self.inventory = {
            "Widget A": {"stock": 10},
            "XYZ-1": {"stock": 0}
        }

    def reset(self) -> Observation:
        self.state = State(tasks_completed=0, current_task_idx=0, total_score=0.0)
        return self._get_initial_observation()

    def _get_initial_observation(self) -> Observation:
        idx = self.state.current_task_idx
        if idx >= len(self.tasks):
            return Observation(system_message="All tasks completed.")
        
        task = self.tasks[idx]
        return Observation(
            system_message=f"New ticket arrived. Task {idx+1}/{len(self.tasks)}. Analyze the ticket and use commands to solve it.",
            current_ticket=task,
            echoed_message=f"Task {idx+1} started."
        )

    def step(self, action: Action):
        idx = self.state.current_task_idx
        if idx >= len(self.tasks):
            return self._get_initial_observation(), 0.0, True, {}
        
        task = self.tasks[idx]
        cmd = action.command
        arg = action.argument
        
        reward = 0.0
        done = False
        obs = Observation(system_message=f"Executed {cmd} with {arg}")
        obs.echoed_message = f"Executed {cmd}"

        if cmd == "search_kb":
            obs.knowledge_base_result = json.dumps(self.kb)
            reward = 0.1
        elif cmd == "get_customer_info":
            if arg in self.customers:
                obs.customer_info = self.customers[arg]
                reward = 0.2
            else:
                obs.customer_info = {"error": "Customer not found"}
        elif cmd == "check_inventory":
            if arg in self.inventory:
                obs.inventory_status = self.inventory[arg]
                reward = 0.2
            else:
                obs.inventory_status = {"error": "Product not found"}
        elif cmd == "reply_to_customer":
            if task["type"] == "KB_QUERY":
                if "30" in arg.lower() and "days" in arg.lower():
                    reward = 1.0
                else:
                    reward = 0.0
                done = True
            elif task["type"] == "REPLACEMENT_OUT_OF_STOCK":
                if "refund" in arg.lower() or "stock" in arg.lower():
                    pass
                obs.command_result = "Message sent to customer."
            else:
                obs.command_result = "Message sent. But task remains unresolved."
        elif cmd == "issue_refund":
            try:
                amt = float(arg)
                if "expected_amount" in task:
                    if abs(amt - task["expected_amount"]) < 0.1:
                        reward = 1.0
                        done = True
                    else:
                        reward = 0.0
                        done = True
                else:
                    reward = -0.5
            except ValueError:
                reward = -0.5
        elif cmd == "issue_replacement":
            if task["type"] == "REPLACEMENT_OUT_OF_STOCK":
                reward = -0.5
                obs.command_result = "Failed: Item is out of stock! Find an alternative resolution."
            else:
                obs.command_result = "Replacement issued."
        else:
            obs.command_result = f"Unknown command: {cmd}"

        if done:
            self.state.tasks_completed += 1
            self.state.current_task_idx += 1
            self.state.total_score += reward
            if self.state.current_task_idx >= len(self.tasks):
                return obs, reward, True, {}
            else:
                next_obs = self._get_initial_observation()
                next_obs.echoed_message = f"Task completed with reward {reward}. " + next_obs.echoed_message
                return next_obs, reward, False, {}

        return obs, reward, False, {}

    def get_state(self) -> State:
        return self.state
