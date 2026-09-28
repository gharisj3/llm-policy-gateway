# LLM Policy Gateway

An HTTP gateway for enforcing tenant policies between applications and language
model providers. The project is being built in stages; the current scaffold
provides a health endpoint and offline checks.

## Local setup

Requires Python 3.11 or newer.

```sh
python -m venv .venv
python -m pip install -e ".[dev]"
python -m uvicorn llm_policy_gateway.app:app --host 127.0.0.1 --port 8080
```

Run the checks with `make lint` and `make test`. Copy `.env.example` to `.env`
before configuring services in later stages.
