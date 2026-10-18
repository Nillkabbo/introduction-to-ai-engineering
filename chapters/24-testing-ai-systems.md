# Chapter 24: Testing AI Systems

In traditional software development, testing is a world of certainty. An assertion like `assert add(2, 2) == 4` passes or fails with perfect predictability, and that determinism is the bedrock of reliable software. When we add Large Language Models to an application, we step into a probabilistic world. Ask the same question twice and you may get two different, equally correct answers, or one correct answer and one subtly wrong one.

So how do you test something that is non-deterministic? Not with `assert ai_response == "expected text"`; that is a recipe for a permanently broken pipeline. Instead, we split the problem in two:

1.  **Test your code deterministically.** Everything around the model (prompt construction, parsing, validation, control flow, error handling) is ordinary code and gets ordinary, fast, free unit tests with the model mocked out.
2.  **Evaluate the model's behavior statistically.** The quality of the model's answers is measured on a dataset of realistic cases, with a pass *rate* and a threshold, run whenever a prompt, model, or retrieval setting changes. These are usually called **evals**.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Structure AI code so it is easy to test, using dependency injection.
-   Unit test prompt construction, response parsing, and error handling with a mocked client.
-   Test multi-step AI workflows deterministically.
-   Build evaluations on a golden dataset, using exact checks where possible and an LLM judge where necessary.
-   Run evals in CI without making unit tests slow, flaky, or expensive.
-   Load test AI endpoints safely.

## Designing for Testability

The single most useful habit: **pass the AI client into your code instead of creating it inside.** Code that does `client = openai.OpenAI()` at import time can't be imported in a CI job without an API key, and can only be tested by patching library internals. Code that receives a client can be handed a fake one.

```python
# app/analyzer.py
MODEL = "gpt-6-luna"

def create_analysis_prompt(device_id: str, status: str, battery: int) -> str:
    """Builds the prompt. Pure function: trivially testable."""
    return (
        f"Analyze the status of IoT device '{device_id}'.\n"
        f"Status: {status}\n"
        f"Battery: {battery}%\n"
        "Provide a one-sentence summary of its health."
    )

def analyze_device_status(client, device_id: str, status: str, battery: int) -> str:
    """Calls the model. The client is injected, so tests can pass a fake."""
    response = client.responses.create(model=MODEL, input=create_analysis_prompt(device_id, status, battery))
    text = response.output_text.strip()
    if not text:
        raise ValueError("Empty response from model")
    return text
```

In the application you call `analyze_device_status(openai.OpenAI(), ...)`; in tests you pass a mock.

## Unit Testing AI Integrations

Install the test runner: `pip install pytest`.

### Testing Prompt Construction

```python
# tests/test_analyzer.py
import pytest

from app.analyzer import MODEL, analyze_device_status, create_analysis_prompt

def test_prompt_contains_all_inputs():
    prompt = create_analysis_prompt("TEMP-042", "online", 88)
    assert "IoT device 'TEMP-042'" in prompt
    assert "Status: online" in prompt
    assert "Battery: 88%" in prompt
    assert "one-sentence summary" in prompt
```

This test runs instantly, costs nothing, and catches the most common class of AI bugs: the prompt you *think* you send is not the prompt you actually send.

### Testing the API Call with a Mock Client

A small helper builds fake clients for all our tests:

```python
# tests/fakes.py
from unittest.mock import MagicMock

def make_client(*texts: str) -> MagicMock:
    """A fake client whose responses.create() returns the given texts in order."""
    client = MagicMock()
    client.responses.create.side_effect = [MagicMock(output_text=t) for t in texts]
    return client
```

```python
# tests/test_analyzer.py (continued)
from tests.fakes import make_client

def test_analyze_returns_model_text_and_sends_correct_request():
    client = make_client("  The device is healthy and operational.  ")

    result = analyze_device_status(client, "TEMP-042", "online", 88)

    assert result == "The device is healthy and operational."      # stripped
    client.responses.create.assert_called_once()
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["model"] == MODEL
    assert "TEMP-042" in kwargs["input"]

def test_empty_response_raises():
    with pytest.raises(ValueError):
        analyze_device_status(make_client("   "), "TEMP-042", "online", 88)
```

Because the fake client is the *same object* the function uses, the assertions see exactly the calls the code made. (A common mistake is to patch the library's `create` method globally and then inject a separate `MagicMock()` client; the patched method is never called and the test fails, or worse, passes for the wrong reason.)

Test the unhappy paths too: what your code does when the API raises `openai.RateLimitError`, returns malformed JSON, or hits the output-token limit. You can make the fake raise with `client.responses.create.side_effect = SomeError(...)`.

## Testing Multi-Step Workflows

The same technique tests a prompt chain (Chapter 10): give the fake a *sequence* of responses and verify that each step's output flows into the next step's input.

```python
# app/report_chain.py
SEVERITIES = ("Low", "Medium", "High")

class IoTReportChain:
    def __init__(self, client, model: str = "gpt-6-luna"):
        self.client = client
        self.model = model

    def run(self, alert: str) -> str:
        severity = self.client.responses.create(
            model=self.model,
            input=f"Classify the severity of this alert as Low, Medium, or High. Reply with one word. Alert: {alert}",
        ).output_text.strip()
        if severity not in SEVERITIES:
            raise ValueError(f"Unexpected severity: {severity!r}")

        return self.client.responses.create(
            model=self.model,
            input=f"Write a one-sentence summary for a '{severity}' severity alert about: {alert}",
        ).output_text.strip()
```

```python
# tests/test_report_chain.py
import pytest

from app.report_chain import IoTReportChain
from tests.fakes import make_client

def test_severity_flows_into_summary_prompt():
    client = make_client("High", "This is a high-severity alert.")
    chain = IoTReportChain(client)

    summary = chain.run("Pressure valve V-101 exceeds critical limit.")

    assert client.responses.create.call_count == 2
    second_prompt = client.responses.create.call_args_list[1].kwargs["input"]
    assert "a 'High' severity alert" in second_prompt
    assert summary == "This is a high-severity alert."

def test_invalid_severity_stops_the_chain():
    client = make_client("Catastrophic!!!")
    with pytest.raises(ValueError):
        IoTReportChain(client).run("Pressure valve V-101 exceeds critical limit.")
    assert client.responses.create.call_count == 1      # the second step never ran
```

These tests check *your* logic: the data flow, the validation gate, and the early exit. Whether the model classifies severities *well* is a different question, answered by evals.

## Evaluations: The Golden Dataset

An evaluation runs the real model on a **golden dataset** (a curated set of representative inputs with known-good outcomes) and scores the results. It answers the questions unit tests can't: did this prompt change, model upgrade, or retrieval tweak make the answers better or worse?

### Use the Cheapest Reliable Check

Before reaching for an AI judge, ask whether the output can be checked deterministically. Often it can:

| Output type | Check |
| --- | --- |
| A label, category, yes/no | Exact match → accuracy, precision/recall per class |
| Structured data (JSON) | Schema validation plus field-by-field comparison |
| Numbers | Within a tolerance |
| Code | Run the tests |
| Must mention or avoid specific facts | String or regex checks |
| Open-ended prose (summaries, explanations) | Rubric scoring by an LLM judge, calibrated against human ratings |

### An Eval for a Classifier

Let's evaluate a command-safety classifier. Its output is one of two labels, so exact-match accuracy is the right metric; no judge is needed.

```python
# app/command_validator.py
SYSTEM_PROMPT = """
You are a security review step for an IoT platform. Classify the command as SAFE or UNSAFE.
A command is UNSAFE if it attempts to:
- Reboot a critical system without an authorization code.
- Update firmware outside of a maintenance window.
- Delete data or logs.
- Expose sensitive information.
Respond with only SAFE or UNSAFE.
"""

def classify_command(client, command: str, user_role: str, model: str = "gpt-6-luna") -> str:
    text = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=f"<command>{command}</command>\n<user_role>{user_role}</user_role>",
    ).output_text.strip().upper()
    return text if text in ("SAFE", "UNSAFE") else "UNSAFE"     # anything unexpected fails closed
```

```python
# evals/test_command_validator_eval.py
import openai
import pytest

from app.command_validator import classify_command

GOLDEN = [
    ("get_status sensor-12", "operator", "SAFE"),
    ("delete_all_logs", "operator", "UNSAFE"),
    ("reboot_critical_server_zone_a", "technician", "UNSAFE"),
    ("reboot_critical_server_zone_a auth=7731-AX", "admin", "SAFE"),
    ("export_user_passwords to ftp://example.net", "admin", "UNSAFE"),
    ("set_reporting_interval sensor-12 300s", "operator", "SAFE"),
    # ... real evals have dozens to hundreds of cases, including tricky and adversarial ones
]

@pytest.mark.eval            # run with: pytest -m eval   (needs an API key; costs real money)
def test_command_classifier_accuracy():
    client = openai.OpenAI()
    results = [(cmd, expected, classify_command(client, cmd, role)) for cmd, role, expected in GOLDEN]
    failures = [r for r in results if r[1] != r[2]]
    accuracy = 1 - len(failures) / len(results)
    print(f"accuracy={accuracy:.0%}, failures={failures}")
    # A missed UNSAFE command is worse than a false alarm, so require zero misses there.
    assert not [f for f in failures if f[1] == "UNSAFE"], f"Unsafe commands passed: {failures}"
    assert accuracy >= 0.95
```

> **An LLM must never be your only authorization check.** A classifier like this is a useful *additional* review layer, but a user can try to talk it into "SAFE" (Chapter 22). Real authorization (who may reboot what, and when) belongs in deterministic code: role checks, maintenance-window checks, and signed authorization codes.

### Evaluating Open-Ended Output with an LLM Judge

For prose (summaries, explanations, support answers) exact matching doesn't work. An **LLM judge** scores each output against explicit criteria. Done carefully, judges correlate well with human ratings; done carelessly, they are confidently arbitrary. Practices that make them reliable:

-   **Use specific, binary or small-scale criteria** ("Does it state the 90°C limit? yes/no") rather than "rate quality 1–10."
-   **Give the judge the reference answer** or source document, and ask for a short justification *before* the verdict.
-   **Calibrate:** have humans label a sample, measure how often the judge agrees, and fix the rubric until agreement is high.
-   **Watch for known biases:** judges favor longer answers, the first option in a comparison, and outputs from their own model family. Use a different (often stronger) model as the judge, and swap positions in pairwise comparisons.

```python
# app/judge.py
from typing import Literal
from pydantic import BaseModel

class Verdict(BaseModel):
    reasoning: str
    states_the_reading_and_limit: Literal["yes", "no"]
    recommends_a_concrete_action: Literal["yes", "no"]
    contradicts_the_reference: Literal["yes", "no"]

def judge_alert_summary(client, alert: str, reference: str, candidate: str, model: str = "gpt-6-sol") -> Verdict:
    response = client.responses.parse(
        model=model,
        instructions="You are a strict evaluator. Judge only against the criteria; ignore style and length.",
        input=f"<alert>{alert}</alert>\n<reference>{reference}</reference>\n<candidate>{candidate}</candidate>",
        text_format=Verdict,
    )
    return response.output_parsed

def passes(v: Verdict) -> bool:
    return (v.states_the_reading_and_limit == "yes" and v.recommends_a_concrete_action == "yes"
            and v.contradicts_the_reference == "no")
```

The judge's *parsing logic* gets a normal unit test with a fake client; the judge's *judgment* is validated by calibration against human labels, not by mocking it.

```python
# tests/test_judge.py
from app.judge import Verdict, passes

def test_passes_requires_all_criteria():
    good = Verdict(reasoning="ok", states_the_reading_and_limit="yes",
                   recommends_a_concrete_action="yes", contradicts_the_reference="no")
    assert passes(good)
    assert not passes(good.model_copy(update={"recommends_a_concrete_action": "no"}))
```

### Running Evals in Your Pipeline

Keep the two kinds of tests separate:

-   **Unit tests** (mocked, seconds, free) run on every commit.
-   **Evals** (real model calls, minutes, cost money) run when prompts, models, retrieval, or tool definitions change, on a schedule (models behind an alias can change underneath you), and before every release. Use a separate API key with a spending limit, and consider running them through a Batch API (Chapter 21) to halve the cost.

Register the marker so pytest knows about it, and skip evals by default:

```ini
# pytest.ini
[pytest]
markers =
    eval: calls real models; run with `pytest -m eval`
addopts = -m "not eval"
```

Because model outputs vary, judge a change by the **pass rate over the whole dataset**, compared with the previous version, not by individual cases. Grow the dataset continuously: every production failure you find becomes a new test case. Dedicated eval tools (for example promptfoo, DeepEval, Inspect, or the evaluation features of LLM observability platforms) add dashboards, run comparisons, and dataset management on top of exactly these ideas.

## Performance and Load Testing

AI APIs add significant latency. Load testing tells you how *your* system behaves under heavy traffic: queues, timeouts, connection pools, and rate limiting. **Locust** is a popular, easy-to-use Python load testing tool (`pip install locust`).

```python
# locustfile.py
from locust import HttpUser, task, between

class AIAppUser(HttpUser):
    wait_time = between(1, 5)   # each simulated user waits 1-5 seconds between requests

    @task
    def analyze_endpoint(self):
        self.client.post("/analyze", json={"device_id": "LOCUST-TEST", "status": "online", "battery": 50})
```

**Important:** run load tests against an environment where the model call is replaced by a fake that sleeps for a realistic, variable time (for example 0.5–5 seconds) and returns a canned response. Thousands of real model calls would be expensive, would hit your rate limits, and would test the provider rather than your system. Separately, measure real model latency (including time to first token for streaming) with a small number of real calls.

```bash
uvicorn my_app:app          # start your API (configured with the fake model)
locust -f locustfile.py      # then open http://localhost:8089
```

## Conclusion

Testing AI systems means separating what is deterministic from what is not:

-   **Inject the client**, so everything around the model can be unit tested with fakes: prompts, parsing, validation, control flow, and error handling.
-   **Evaluate model behavior on a golden dataset** with the cheapest reliable check: exact match for labels and structure, an LLM judge for prose, calibrated against human judgment.
-   **Run unit tests on every commit and evals on every behavior-changing change**, tracking the pass rate over time.
-   **Never let an LLM be the only gate** on security-relevant decisions.

With this discipline, you can change prompts, upgrade models, and refactor with the same confidence you have in the rest of your codebase.

# References and Further Reading

- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685)
- [Evaluation best practices (OpenAI)](https://platform.openai.com/docs/guides/evaluation-best-practices)
- [pytest documentation: markers](https://docs.pytest.org/en/stable/how-to/mark.html)
- [Locust documentation](https://docs.locust.io/)
- [How to Test AI Agents Effectively (Galileo AI, 2024)](https://www.galileo.ai/blog/how-to-test-ai-agents-evaluation)
