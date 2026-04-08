# Customer Support OpenEnv

This is a real-world task simulation environment built for the OpenEnv framework.

## Environment Description & Motivation
This environment models a standard Customer Support ticketing system. Agents are evaluated on their ability to triage tickets, look up information using available commands, and correctly process operations like refunds or replies. 
Unlike games or toys, support agent simulation directly matches real-world use cases for large language models, where they must orchestrate multiple tools (`search_kb`, `get_customer_info`, `check_inventory`, `issue_refund`, etc.) and decide on a multi-step workflow.

## Setup and Usage Instructions

### Docker (Hugging Face Spaces)
The environment provides a typical Hugging Face Space Dockerfile that exposes port `7860`.
```bash
docker build -t customer-support-env .
docker run -p 7860:7860 customer-support-env
```
Once deployed, the endpoints `/reset`, `/step`, and `/state` are available.

### Baseline Inference
You can run the baseline script to reproduce agent scores on the tasks:
```bash
python inference.py
```
This requires `OPENAI_API_KEY` to be set, or it will use a mock built-in deterministic proxy to ensure standard formatting is printed. It strictly outputs logs in `[START]`, `[STEP]`, and `[END]` formats.

## Action Space
`command` must be one of the following strings, and `argument` is a simple text string or JSON value.
* `search_kb` (argument: query str)
* `get_customer_info` (argument: email)
* `check_inventory` (argument: product_name)
* `issue_refund` (argument: exact numerical amount)
* `issue_replacement` (argument: product_name)
* `reply_to_customer` (argument: response message text)

## Observation Space
The environment returns structured Pydantic models containing:
* `system_message`: Environment directions or narrative.
* `current_ticket`: Content of the active customer ticket.
* `customer_info`: If requested via command.
* `inventory_status`: If requested via command.
* `knowledge_base_result`: Search results.

## Task Descriptions (Difficulty Range)
1. **Easy**: Customer asks a simple general question (e.g. return policy). Agent must output `search_kb` and then `reply_to_customer` with the exact policy. Grader checks string contents of the agent's reply.
2. **Medium**: Customer asks for a refund. Agent must use `get_customer_info` (with the email) to look up the order amount. Then execute `issue_refund` with the exact correct numerical value. Grader validates the exact numerical correctness inside the command.
3. **Hard**: Customer requests a replacement for a damaged good. Agent must `check_inventory` and discover it is out of stock. The agent must handle this implicit rule by falling back to `issue_refund` instead. The agent is scored stringently on whether they caught the OOS edge case.
