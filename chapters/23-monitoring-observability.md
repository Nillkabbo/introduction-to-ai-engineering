# Chapter 23: Monitoring and Observability in AI Systems

In traditional software, when something breaks, it often breaks loudly. A server crashes, an endpoint returns a `500` error, a database query times out. You get an alert, you check the logs, and you find the root cause. AI applications, however, can fail silently and insidiously. A model can start "hallucinating" more often, its responses can become subtly less accurate, or a change in user behavior can cause costs to skyrocket without a single error ever being thrown.

This is why **observability** is not just a best practice in AI engineering—it is a fundamental requirement for building and maintaining trustworthy systems. Standard monitoring tells you if your application is *up*; AI observability tells you if your application is *smart*, *safe*, and *sane*. It's the art of understanding the internal state of your AI system from its external outputs.

This chapter will teach you how to instrument your AI applications with the "three pillars of observability"—logging, metrics, and tracing—tailored specifically to the unique challenges of production AI.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Implement structured logging to capture the complete context of every AI interaction.
-   Define and track AI-specific metrics for performance, cost, and quality.
-   Use distributed tracing to follow a request through a complex chain of AI services.
-   Build a real-time monitoring dashboard for an IoT system using Prometheus and Grafana concepts.
-   Set up automated alerts for critical AI-related issues like cost spikes, error rates, and quality proxies such as validation failures.

## The Three Pillars of AI Observability

Modern observability is built on three core types of telemetry data. For AI applications, each has a unique and critical role.

1.  **Logs:** Detailed, timestamped records of discrete events. *AI Twist: A log isn't just an error message; it's the full prompt, the model's response, the tokens used, and the latency of a specific AI call.*
2.  **Metrics:** Aggregated, numerical data measured over time. *AI Twist: Metrics go beyond CPU/memory to include `tokens_per_minute`, `cost_per_user`, `hallucination_rate`, and `cache_hit_ratio`.*
3.  **Traces:** A complete view of a single request as it travels through multiple services. *AI Twist: Tracing is essential for debugging complex prompt chains or multi-agent systems where one user input can trigger multiple, cascading AI calls.*

Let's build a system that incorporates all three.

## Pillar 1: Structured Logging for AI

The first step to understanding your AI is to log every interaction with rich context. The best practice is to use **structured logging**, where log entries are written as JSON objects, not plain text. This makes them machine-readable and easy to query in log management systems like Elasticsearch, Datadog, or Splunk.

We'll use the `structlog` library, which integrates seamlessly with Python's standard `logging` module to produce structured JSON logs.

First, install `structlog`: `pip install structlog`.

### Instrumenting an AI Call with Structured Logging

Let's create a centralized function for making AI calls that automatically logs all the important details.

```python
# structured_logging_example.py
import time
import uuid

import openai
import structlog
from dotenv import load_dotenv

# --- Configure structlog for JSON output ---
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

# --- Load API Key and Initialize Client ---
load_dotenv()
client = openai.OpenAI()
PROMPT_VERSION = "device-explainer@2026-10-01"

def monitored_ai_call(prompt: str, user_id: str, model: str = "gpt-6-luna") -> str:
    """Makes an AI call with comprehensive structured logging."""
    
    request_id = str(uuid.uuid4())
    start = time.monotonic()

    log.info("ai_call_start", request_id=request_id, user_id=user_id, model=model,
             prompt_version=PROMPT_VERSION, prompt_chars=len(prompt))

    try:
        response = client.responses.create(model=model, input=prompt)
        latency_ms = (time.monotonic() - start) * 1000
        usage = response.usage

        log.info(
            "ai_call_success",
            request_id=request_id,
            user_id=user_id,
            model=model,
            prompt_version=PROMPT_VERSION,
            provider_request_id=response.id,         # lets you match your logs with the provider's
            latency_ms=round(latency_ms, 2),
            input_tokens=usage.input_tokens,
            cached_tokens=usage.input_tokens_details.cached_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
            # Full prompts and responses may contain personal data: log them only where
            # your privacy policy allows, with access control and a retention limit.
            prompt=prompt,
            response=response.output_text,
        )
        return response.output_text
        
    except openai.OpenAIError as e:
        latency_ms = (time.monotonic() - start) * 1000
        log.error(
            "ai_call_failure",
            request_id=request_id,
            user_id=user_id,
            model=model,
            latency_ms=round(latency_ms, 2),
            error_message=str(e),
            error_type=type(e).__name__
        )
        raise

# --- Example Usage ---
try:
    monitored_ai_call(
        prompt="Explain the difference between MQTT and HTTP for IoT data.",
        user_id="user-123"
    )
except Exception:
    pass
```

Running this code produces beautifully structured JSON logs that can be easily ingested and analyzed:

```json
{"event": "ai_call_start", "request_id": "...", "user_id": "user-123", "model": "gpt-6-luna", "prompt_version": "device-explainer@2026-10-01", "prompt_chars": 58, "level": "info", "timestamp": "2026-10-18T14:30:00.123Z"}
{"event": "ai_call_success", "request_id": "...", "user_id": "user-123", "model": "gpt-6-luna", "latency_ms": 1333.0, "input_tokens": 15, "cached_tokens": 0, "output_tokens": 320, "reasoning_tokens": 192, "prompt": "...", "response": "...", "level": "info", "timestamp": "2026-10-18T14:30:01.456Z"}
```

With logs like these, you can easily answer questions like:
-   "Show me all failed API calls for `gpt-6-sol` in the last hour."
-   "Which prompt version produced this bad answer?"
-   "Find all interactions from `user-123` that took longer than 3 seconds."
-   "What was the exact prompt that led to this specific erroneous response?"

## Pillar 2: Metrics for AI Systems

Metrics are numerical data points collected over time. They give you a high-level view of your system's health and performance. For AI applications, we need to track both traditional system metrics and AI-specific ones.

We'll use the `prometheus-client` library, which is the standard for exposing metrics that can be scraped by a **Prometheus** server and visualized in a **Grafana** dashboard.

First, install the library: `pip install prometheus-client`.

### Defining AI-Specific Metrics

Let's define a set of essential metrics for any AI application.

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# --- AI API Metrics ---
AI_API_CALLS_TOTAL = Counter(
    'ai_api_calls_total',
    'Total number of AI API calls.',
    ['model', 'provider', 'status'] # Labels to slice data by
)

AI_API_CALL_DURATION_SECONDS = Histogram(
    'ai_api_call_duration_seconds',
    'Latency of AI API calls.',
    ['model', 'provider']
)

# --- Token and Cost Metrics ---
TOKENS_PROCESSED_TOTAL = Counter(
    'ai_tokens_processed_total',
    'Total tokens processed by the AI.',
    ['model', 'provider', 'direction'] # direction can be 'input' or 'output'
)

ESTIMATED_COST_USD_TOTAL = Counter(
    'ai_estimated_cost_usd_total',
    'Estimated cumulative cost of AI API calls in USD.',
    ['model', 'provider']
)

# --- Quality and Safety Metrics ---
VALIDATION_FAILURES_TOTAL = Counter(
    'ai_validation_failures_total',
    'Total times AI output failed Pydantic or other validation.',
    ['model']
)

CONTENT_SAFETY_FLAGS_TOTAL = Counter(
    'ai_content_safety_flags_total',
    'Total times AI output was flagged by content safety filters.',
    ['model', 'category']
)

# --- Application-Level Metrics ---
ACTIVE_CONVERSATIONS = Gauge(
    'ai_active_conversations',
    'Number of currently active AI conversations.'
)
```

### Instrumenting Our Code with Metrics

Now, let's update our `monitored_ai_call` function to update these metrics with each call.

```python
# monitored_ai_with_metrics.py
# (Includes code from logging example and metrics definitions)

# Illustrative prices in USD per 1M tokens; load real ones from configuration (Chapter 21).
MODEL_PRICES = {"gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50}}

def monitored_ai_call_with_metrics(prompt: str, user_id: str, model: str = "gpt-6-luna") -> str:
    """An AI call instrumented with structured logging and Prometheus metrics."""
    labels = {"model": model, "provider": "openai"}

    # The Histogram context manager times the block, including failures.
    with AI_API_CALL_DURATION_SECONDS.labels(**labels).time():
        try:
            response = client.responses.create(model=model, input=prompt)
        except openai.OpenAIError as e:
            AI_API_CALLS_TOTAL.labels(**labels, status="failure").inc()
            log.error("ai_call_failure", user_id=user_id, model=model, error_type=type(e).__name__)
            raise

    AI_API_CALLS_TOTAL.labels(**labels, status="success").inc()
    usage = response.usage
    cached = usage.input_tokens_details.cached_tokens
    TOKENS_PROCESSED_TOTAL.labels(**labels, direction="input").inc(usage.input_tokens)
    TOKENS_PROCESSED_TOTAL.labels(**labels, direction="output").inc(usage.output_tokens)

    price = MODEL_PRICES[model]
    cost = ((usage.input_tokens - cached) * price["input"]
            + cached * price["cached_input"]
            + usage.output_tokens * price["output"]) / 1_000_000
    ESTIMATED_COST_USD_TOTAL.labels(**labels).inc(cost)

    log.info("ai_call_success", user_id=user_id, model=model, prompt_version=PROMPT_VERSION,
             input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, cost_usd=round(cost, 6))
    return response.output_text
```

Note what is *not* a label: `user_id`. Every distinct label value creates a separate time series in Prometheus, so high-cardinality values (user IDs, request IDs, prompt text) belong in logs and traces, not in metric labels.

To make these metrics accessible, expose a `/metrics` endpoint. In a FastAPI application, mount it on the app itself (as in the project below); in a standalone worker process, run a small metrics server:

```python
from prometheus_client import start_http_server

start_http_server(9100)  # a port your web server isn't using
```

A Prometheus server then scrapes the endpoint periodically and stores the history for dashboards and alerts.

## Pillar 3: Distributed Tracing

While metrics give you the "what" (e.g., "latency is high"), **distributed tracing** gives you the "where" and "why." A trace follows a single request through its entire journey across all the microservices and components it touches. This is indispensable for debugging complex AI workflows.

We'll use **OpenTelemetry**, the industry standard for tracing.

First, install the necessary packages: `pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp`.

### Setting Up a Tracer

You configure a tracer once at the start of your application. It will send trace data to a "collector" (like Jaeger or Honeycomb) which then visualizes it.

```python
# telemetry_setup.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# For demo purposes, we'll print traces to the console.
# In production, you'd use an OTLPSpanExporter to send to a collector.
provider = TracerProvider()
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)

# Sets the global tracer provider
trace.set_tracer_provider(provider)

# Gets a tracer from the global provider
tracer = trace.get_tracer(__name__)
```

### Instrumenting Our Code with Tracing

We can now use this tracer to create "spans," which represent individual units of work within the trace.

```python
# monitored_ai_with_tracing.py
# (Includes all previous code)

def final_monitored_ai_call(prompt: str, user_id: str, model: str = "gpt-6-luna") -> str:
    """The final version of our monitored function, now with tracing."""
    
    # tracer.start_as_current_span creates a new span for the trace.
    # The 'with' statement ensures it's properly closed.
    with tracer.start_as_current_span(f"chat {model}") as span:
        # Attribute names follow the OpenTelemetry GenAI semantic conventions where they exist,
        # so any compatible backend can display them.
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("app.user_id", user_id)
        span.set_attribute("app.prompt_version", PROMPT_VERSION)
        
        # Now, call our existing function that handles logging and metrics
        try:
            result = monitored_ai_call_with_metrics(prompt, user_id, model)
            
            # Add results to the span
            span.set_attribute("response.length", len(result))
            span.set_status(trace.Status(trace.StatusCode.OK))
            
            return result
        except Exception as e:
            # Record the exception in the trace for easy debugging
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
```

When you run this code, OpenTelemetry will print a JSON representation of the trace to your console, showing the duration, attributes, and status of the AI call. In a production system with multiple services, traces from each service are linked by a common `trace_id`, giving you an end-to-end view of the request. For an agent, one user request becomes a tree of spans (each model call and each tool call), which is by far the fastest way to see where a multi-step run went wrong.

You rarely need to write all the spans by hand. OpenTelemetry **instrumentation libraries** exist for the OpenAI, Anthropic, and Google SDKs and for popular frameworks, and the official Anthropic SDK can emit spans itself. Specialized **LLM observability platforms** (open-source options such as Langfuse and Arize Phoenix, and commercial ones) build on these traces to show prompts, responses, token costs, and evaluation scores per request.

## Measuring Quality, Not Just Uptime

Logs, metrics, and traces tell you whether the system is up, fast, and affordable. They don't tell you whether the answers are *good*, which is the failure mode this chapter opened with. Quality monitoring combines several signals:

-   **Cheap automatic checks on every response:** schema validation failures, refusals, empty or truncated answers, responses that should contain a citation but don't.
-   **User feedback:** thumbs up/down, corrections, and behavioral signals such as the user immediately rephrasing the question.
-   **Sampled evaluation:** score a small, random sample of production traffic with an LLM judge or human reviewers against your quality criteria (Chapter 24), and track the score over time as a metric.
-   **Drift alerts:** alert when any of these move, for example after a model version, prompt, or retrieval change, or when the mix of user requests shifts.

## Alerting: Turning Metrics into Action

Dashboards only help when someone is looking at them. Alerts watch the metrics for you and page a person when something needs attention. With Prometheus, you write alert rules as queries over the metrics defined above, and Alertmanager routes the resulting alerts to Slack, email, or an on-call tool.

```yaml
# alert_rules.yml
groups:
  - name: ai-application
    rules:
      - alert: AICostSpike
        # Spend over the last hour is more than 3x the same hour last week.
        expr: |
          sum(increase(ai_estimated_cost_usd_total[1h]))
            > 3 * sum(increase(ai_estimated_cost_usd_total[1h] offset 1w))
        for: 15m
        labels: {severity: page}
        annotations:
          summary: "AI spend is 3x higher than last week"

      - alert: AIHighErrorRate
        expr: |
          sum(rate(ai_api_calls_total{status="failure"}[5m]))
            / sum(rate(ai_api_calls_total[5m])) > 0.05
        for: 10m
        labels: {severity: page}
        annotations:
          summary: "More than 5% of AI calls are failing"

      - alert: AIValidationFailuresRising
        # A proxy for quality: outputs that fail schema or business-rule checks.
        expr: |
          sum(rate(ai_validation_failures_total[30m]))
            / sum(rate(ai_api_calls_total{status="success"}[30m])) > 0.02
        for: 30m
        labels: {severity: ticket}
        annotations:
          summary: "More than 2% of AI outputs fail validation"

      - alert: AISlowResponses
        expr: |
          histogram_quantile(0.95,
            sum by (le) (rate(ai_api_call_duration_seconds_bucket[5m]))) > 20
        for: 10m
        labels: {severity: ticket}
        annotations:
          summary: "p95 AI latency is above 20 seconds"
```

A few rules of thumb keep alerts useful:

-   **Alert on symptoms users feel** (errors, latency, cost, quality), not on every internal metric. Each page should need a human to act.
-   **Use `for:` to ignore short blips**, and compare against a baseline (such as `offset 1w`) for metrics like cost that vary by time of day.
-   **Match the label values to your code.** These rules use the `"success"` and `"failure"` status values set by `monitored_ai_call` above; if you change the labels in code, change the rules too.
-   **You can't alert on hallucinations directly.** Alert on proxies you can measure: validation failures, a falling score from sampled evaluation, or a rise in negative user feedback. Provider spending limits are a useful last line of defence against runaway cost.

## Practical Project: A Real-Time IoT Monitoring Dashboard

Let's build a simple but complete application that brings all three pillars together. We'll create a FastAPI service that simulates receiving IoT data, analyzes it with our fully-monitored AI function, and exposes a metrics endpoint for Prometheus.

```python
# complete_monitoring_app.py
import asyncio
import random
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI
from prometheus_client import make_asgi_app

# Assumes the logging, metrics, and tracing setup above, and final_monitored_ai_call.

def analyze_reading(data: dict) -> None:
    prompt = f"Analyze this IoT sensor data and determine if it indicates an anomaly. Data: {data}"
    try:
        final_monitored_ai_call(prompt, user_id=data.get("device_id", "unknown"))
    except openai.OpenAIError:
        pass   # already logged, counted, and recorded on the span

async def simulate_iot_traffic():
    """Generates sample readings and analyzes them, as an external source would."""
    while True:
        data = {
            "device_id": f"sensor-{random.randint(1, 10)}",
            "temperature": round(70 + random.uniform(-5, 15), 2),
            "humidity": round(50 + random.uniform(-10, 10), 2),
        }
        # The OpenAI client is synchronous: run it in a thread so the event loop stays free.
        asyncio.create_task(asyncio.to_thread(analyze_reading, data))
        await asyncio.sleep(random.uniform(0.5, 2.0))

@asynccontextmanager
async def lifespan(app: FastAPI):
    simulator = asyncio.create_task(simulate_iot_traffic())   # start background traffic
    yield
    simulator.cancel()                                          # stop it on shutdown

app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app())   # Prometheus scrapes http://localhost:8000/metrics

@app.post("/iot/data")
async def process_iot_data(data: dict, background_tasks: BackgroundTasks):
    """Receives IoT data and schedules an AI analysis after the response is sent."""
    background_tasks.add_task(analyze_reading, data)
    return {"status": "accepted", "message": "Data received and queued for analysis."}

# To run: uvicorn complete_monitoring_app:app
# Then visit http://localhost:8000/metrics. The console shows the JSON logs and traces.
```

A detail worth noticing: the simulator calls `analyze_reading` directly rather than calling the endpoint function with a hand-made `BackgroundTasks()` object. Background tasks only run when FastAPI itself handles a request; tasks added to an object you create yourself are silently never executed. Every real call to `POST /iot/data` does run its background task. (Simulated traffic is great for a demo, but remember it costs real money per call; for load testing, mock the model, Chapter 24.)

With this application running, you have a complete observability solution:
-   **Logs:** Every AI call produces a structured JSON log on the console.
-   **Metrics:** The `/metrics` endpoint provides real-time data ready for Prometheus.
-   **Traces:** The console will show detailed traces for each AI interaction.

You now have the visibility needed to confidently run, debug, and scale your AI applications in production.

## Key Takeaways

-   Log every AI call as a structured JSON event (with `structlog`) carrying a request ID, the provider's response ID, model, prompt version, latency, and input, cached, output, and reasoning tokens. Log full prompts and responses only where your privacy policy allows, with access control and a retention limit.
-   Expose Prometheus metrics for calls by status, latency (a `Histogram`), tokens by direction, estimated cost, validation failures, and safety flags. Keep label cardinality low: user IDs, request IDs, and prompt text belong in logs and traces, never in metric labels.
-   Use OpenTelemetry spans for each model and tool call, named with the GenAI semantic conventions (`gen_ai.request.model` and friends), and record exceptions and status on the span. For agents, the span tree is the fastest way to see where a run went wrong, and instrumentation libraries or LLM observability platforms save you writing spans by hand.
-   Uptime, latency, and cost don't tell you whether answers are good. Add cheap checks on every response, user feedback, sampled evaluation tracked as a metric over time, and alerts when any of them drift after a model, prompt, or retrieval change.
-   In a FastAPI service, mount `/metrics` with `make_asgi_app()`, run synchronous model calls with `asyncio.to_thread` so the event loop stays free, and never call an endpoint with a hand-made `BackgroundTasks()`, because those tasks never run.

## Exercises

**1. Warm-up:** Exercise the failure path on purpose: call `final_monitored_ai_call` with a model name that doesn't exist and catch the error. You are done when you can point to all three records of the same failure: an `ai_call_failure` log line with its `error_type`, `ai_api_calls_total{status="failure"}` incremented on the `/metrics` page, and a span on the console with status `ERROR` and the recorded exception.

**2. Build:** Trace a two-step prompt chain (classify an IoT alert's severity, then summarize it, as in Chapter 10) under one parent span named for the request, with one child span per model call. On each child span, set `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` from the response's usage. Done means the console output shows both child spans with the same `trace_id`, each pointing to the parent's span ID, and their token attributes match the token counts in the corresponding log lines.

**3. Stretch:** Add quality monitoring to the project's `analyze_reading`. Change its prompt to require a verdict line of `ANOMALY` or `NORMAL`, then count responses that are empty, cut off (`response.status == "incomplete"`), or missing that verdict, using a new counter with a low-cardinality `reason` label. Then keep a rolling window of the last 100 calls in memory and emit a `quality_alert` log event and set a `Gauge` when the bad-response ratio exceeds 5%. Test it by temporarily swapping the model call for a stub that returns empty text one time in five: you are done when the alert fires within the first window and clears after you remove the stub.

# References and Further Reading

- [OpenTelemetry semantic conventions for generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Prometheus: metric and label naming (on cardinality)](https://prometheus.io/docs/practices/naming/)
- [What You Actually Need to Monitor AI Systems in Production (Sentry, 2025)](https://blog.sentry.io/what-you-actually-need-to-monitor-ai-systems-in-production/)
- [Chapter 12 - Observability (Azure AI in Production Guide)](https://azure.github.io/AI-in-Production-Guide/chapters/chapter_12_keeping_log_observability)
- [Production-Ready Observability Platform for AI Systems (Medium, 2023)](https://medium.com/@bijit211987/production-ready-observability-platform-for-ai-systems-17923d19639b)
- [A Simple Recipe for LLM Observability (Medium, 2025)](https://statistician-in-stilettos.medium.com/a-simple-recipe-for-llm-observability-5c6f24c96310)
- [AI Observability: A Complete Guide to Monitoring Model Performance in Production (Galileo AI, 2025)](https://www.galileo.ai/blog/ai-observability)

<!-- chapter-nav -->

---

[← Chapter 22: Security and Safety in AI Applications](22-security-safety.md) | [Table of Contents](../README.md) | [Chapter 24: Testing AI Systems →](24-testing-ai-systems.md)
