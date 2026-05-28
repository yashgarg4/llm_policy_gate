.PHONY: install demo check dashboard test

POLICY = examples/sentinel_policy.yaml
DEMO_PROMPT ?= Explain what a LangGraph agent is in simple terms.
VENV = .venv/Scripts

install:
	pip install -e ".[dev]"

demo:
	$(VENV)/python examples/demo_agent.py --prompt "$(DEMO_PROMPT)"

check:
	$(VENV)/python cli/check.py --policy $(POLICY) --test-prompt "$(DEMO_PROMPT)"

dashboard:
	$(VENV)/streamlit run dashboard/app.py

test:
	$(VENV)/pytest tests/ -v
