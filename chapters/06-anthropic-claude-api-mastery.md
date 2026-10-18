# Chapter 6: Mastering the Anthropic Claude API

While OpenAI's GPT models are a fantastic entry point into the world of AI, a skilled developer knows that having multiple tools in the toolkit is essential. Anthropic's Claude models are a strong alternative, known for careful instruction following, long-document analysis, coding, and agentic tool use.

This chapter is your guide to the Claude API. We will cover everything from your first API call to applications that use Claude's strengths in document analysis, vision, and tool use. Knowing both OpenAI and Anthropic (and Google, in the next chapter) lets you choose the best model for each task, and avoid depending on a single vendor.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Make your first Claude API call and understand its core differences from OpenAI's API.
-   Read Claude's responses correctly, including when the model thinks before answering.
-   Write detailed system prompts to precisely control Claude's behavior.
-   Analyze long documents, PDFs, and images.
-   Implement Claude's tool use, including parallel tool calls.
-   Reduce cost with prompt caching, and get schema-valid JSON with structured outputs.
-   Choose between Claude models (and between providers) for a given task.

> **Version note.** This chapter reflects the `anthropic` Python SDK 1.x and Claude models as of October 2026. Model IDs used here: `claude-opus-5-5` (the default and most capable Opus model), `claude-sonnet-5-5` (faster and cheaper), and `claude-haiku-5-5` (the cheapest and fastest). Older IDs such as `claude-3-5-sonnet-20240620` have been retired and no longer work.

## Your First Claude API Call: A New Dialect

Interacting with Claude will feel familiar yet distinct. The core concepts of roles and messages are the same, but the API has its own "dialect."

### The Absolute Minimum

First, install the `anthropic` library and create an API key in the Anthropic Console. Store it in the `ANTHROPIC_API_KEY` environment variable (or your `.env` file).

```bash
pip install anthropic
```

Here is the "Hello, World!" of the Claude API:

```python
import anthropic

# The client reads ANTHROPIC_API_KEY from the environment. Never hard-code keys.
client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)

# The response content is a list of blocks. Print the text blocks.
for block in message.content:
    if block.type == "text":
        print(block.text)
```

Let's break down the key differences from the OpenAI API:
-   The client is `anthropic.Anthropic()` and the core method is `client.messages.create()`.
-   You **must** specify `max_tokens`. On current models this budget also covers Claude's internal thinking, so don't set it too low; a small value can cut the answer off (`message.stop_reason == "max_tokens"`).
-   The response's `content` is a **list of blocks** of different types: `thinking`, `text`, `tool_use`, and so on. Current Claude models think before answering by default, so the first block is often a `thinking` block. Always select blocks by their `type`; never assume `content[0]` is text.

### A Reusable Claude Function

Let's turn this into a reusable function with a small helper for extracting text.

```python
import anthropic
from dotenv import load_dotenv

load_dotenv()
claude_client = anthropic.Anthropic()
CLAUDE_MODEL = "claude-opus-5-5"

def text_of(message) -> str:
    """Joins the text blocks of a Claude response."""
    return "".join(block.text for block in message.content if block.type == "text")

def ask_claude(prompt: str, system: str | None = None) -> str:
    """Sends a single prompt to Claude and returns the response text."""
    kwargs = {"system": system} if system else {}
    try:
        message = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
    except anthropic.RateLimitError:
        return "Rate limited; try again shortly."
    except anthropic.APIConnectionError:
        return "Network error reaching the Anthropic API."
    except anthropic.APIStatusError as e:
        return f"Anthropic API error (HTTP {e.status_code})."
    if message.stop_reason == "refusal":
        # Safety classifiers can decline a request (HTTP 200, no answer text).
        return "The model declined this request."
    return text_of(message)

print(ask_claude("In one paragraph, what is the difference between MQTT and CoAP?"))
```

### Controlling Thinking with Effort

On current Claude models, thinking is adaptive: Claude decides how much to think. You control the overall depth, and therefore cost and latency, with the `effort` setting:

```python
message = claude_client.messages.create(
    model=CLAUDE_MODEL,
    max_tokens=16000,
    output_config={"effort": "low"},   # low | medium | high | xhigh | max
    messages=[{"role": "user", "content": "Classify this alert as Power, Connectivity, or Sensor: 'Battery at 3%'"}],
)
print(text_of(message))
```

Use `low` for simple, high-volume tasks like classification or extraction, and `high` or above for hard reasoning, planning, and coding. Sampling parameters such as `temperature` are not accepted on the newest models; as with OpenAI's reasoning models, you get predictable output by constraining it (fixed labels, structured outputs) rather than by turning down randomness.

## Mastering Claude's System Prompts

Claude follows detailed **system prompts** closely. Where OpenAI's Chat Completions puts the system message inside the message list, Claude takes a dedicated `system` parameter.

```python
iot_expert_prompt = """
You are an expert IoT (Internet of Things) diagnostic engineer with 15 years of field experience. You are methodical, cautious, and practical.

When a user describes a problem, follow these steps:
1.  Acknowledge the user's problem.
2.  Ask at least two clarifying questions about the device, environment, or symptoms.
3.  Based on the initial description, give the 2-3 most likely root causes, from most to least probable.
4.  For each potential cause, suggest a simple, safe diagnostic step.
5.  Conclude by asking the user to report back with their findings.

Prioritize safety and data integrity above all else.
"""

user_problem = "A dozen of my temperature sensors in the warehouse suddenly went offline."
print(ask_claude(user_problem, system=iot_expert_prompt))
```

The output will follow the prescribed structure. Two prompting tips that matter more with current models than with older ones: explain *why* a rule exists (Claude generalizes better from reasons than from bare commands), and avoid shouting ("NEVER", "MUST" in capitals). Modern models follow calm, clear instructions well, and over-emphatic rules tend to be over-applied.

## Conversations and Long-Context Analysis

Claude's conversation flow is like OpenAI's: you build and send the history. Current Claude models have context windows of up to one million tokens, enough to analyze entire manuals, contracts, or long transcripts in a single request.

Let's create a bot that "reads" a document and then answers questions about it.

```python
class DocumentAnalyzerBot:
    def __init__(self, document_text: str):
        self.system_prompt = f"""
You are a document analysis assistant. Answer the user's questions using only the document below.
If the answer is not in the document, say "That information is not available in the provided document."

<document>
{document_text}
</document>
"""
        self.history: list[dict] = []

    def ask(self, user_question: str) -> str:
        self.history.append({"role": "user", "content": user_question})
        message = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            system=self.system_prompt,
            messages=self.history,
        )
        answer = text_of(message)
        self.history.append({"role": "assistant", "content": answer})
        return answer

device_manual = """
Product Manual: SmartSense T-1000 Temperature Sensor
Section 1: Installation. Mount the sensor away from direct heat sources.
Section 2: Battery. Uses two 3V CR2032 batteries. Expected life: 24 months.
Section 3: Connectivity. Connects via Zigbee 3.0. A flashing blue light indicates it is searching for a network. A solid blue light indicates it is connected.
Section 4: Troubleshooting. If readings are inaccurate, perform a recalibration cycle by holding the reset button for 10 seconds. If the device is unresponsive, replace the batteries.
"""

analyzer = DocumentAnalyzerBot(device_manual)
print("Bot:", analyzer.ask("How do I connect my new sensor to the network?"))
print("\nBot:", analyzer.ask("What kind of battery does it use?"))
print("\nBot:", analyzer.ask("What is the warranty period?"))
```

The bot answers the first two questions from the text and should report that the warranty information is not available. Wrapping the document in XML-style tags (`<document>…</document>`) helps the model separate your instructions from the material it is analyzing.

### Prompt Caching: Don't Pay Full Price for the Same Document Twice

In the bot above, the whole manual is re-sent with every question. For a real manual that might be 100,000 tokens per call. **Prompt caching** lets the API reuse the processed prefix of a prompt across calls: cached input is billed at a small fraction of the normal input price and is processed faster.

```python
message = claude_client.messages.create(
    model=CLAUDE_MODEL,
    max_tokens=16000,
    cache_control={"type": "ephemeral"},   # cache the prompt prefix automatically
    system=analyzer.system_prompt,
    messages=[{"role": "user", "content": "Summarize the troubleshooting section."}],
)
print(message.usage.cache_creation_input_tokens, message.usage.cache_read_input_tokens)
```

The first call writes the cache; follow-up calls within the cache lifetime (five minutes by default, refreshed on each hit) read from it. Caching matches the *prefix* exactly, so keep stable content (system prompt, documents, tool definitions) first and put changing content (the user's question, timestamps) last. Also note that prefixes below a minimum size (roughly 1,000 to 4,000 tokens depending on the model) are not cached at all. Chapter 21 discusses caching across providers.

## Multimodal: Documents, Vision, and Tool Use

### PDFs and Images

Claude can read PDFs directly, including their tables, charts, and scanned pages, as well as images. Pass them as content blocks, before your text question.

```python
import base64

def analyze_pdf(pdf_path: str, question: str) -> str:
    with open(pdf_path, "rb") as f:
        pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")
    message = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_data}},
                {"type": "text", "text": question},
            ],
        }],
    )
    return text_of(message)

def analyze_iot_dashboard(image_path: str, media_type: str = "image/png") -> str:
    """Sends a screenshot of an IoT dashboard to Claude for analysis."""
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    message = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                {"type": "text", "text": "Summarize the overall system status and identify any devices or metrics that need immediate attention."},
            ],
        }],
    )
    return text_of(message)

# print(analyze_pdf("sensor_manual.pdf", "List every safety warning in this manual."))
# print(analyze_iot_dashboard("dashboard.png"))
```

Images can also be passed by URL (`"source": {"type": "url", "url": "https://..."}`), and large files can be uploaded once with the Files API and referenced by ID.

### Tool Use: Claude's Function Calling

Claude's "tool use" is its equivalent of OpenAI's function calling. The loop is the same: describe the tools, Claude requests one or more calls, you execute them, send the results back, and repeat until Claude answers.

```python
import json

def get_live_device_status(device_id: str) -> dict:
    """A dummy function to simulate fetching live device data."""
    print(f"--- TOOL EXECUTED: get_live_device_status({device_id}) ---")
    if device_id == "TEMP-042":
        return {"status": "online", "temperature": "22.5°C", "battery": "88%"}
    return {"status": "unknown", "error": f"Device {device_id} not found"}

TOOL_FUNCTIONS = {"get_live_device_status": get_live_device_status}

CLAUDE_TOOLS = [{
    "name": "get_live_device_status",
    "description": "Get the live status of a specific IoT device by its ID. Use this whenever the user asks about a device's current state.",
    "input_schema": {
        "type": "object",
        "properties": {"device_id": {"type": "string", "description": "The device ID, e.g. 'TEMP-042'"}},
        "required": ["device_id"],
        "additionalProperties": False,
    },
    "strict": True,   # guarantees the tool input matches the schema
}]

def iot_tool_bot(user_prompt: str, max_turns: int = 5) -> str:
    messages = [{"role": "user", "content": user_prompt}]

    for _ in range(max_turns):
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            tools=CLAUDE_TOOLS,
            messages=messages,
        )
        if response.stop_reason != "tool_use":
            return text_of(response)

        # Keep Claude's full turn (including thinking and tool_use blocks) in the history.
        messages.append({"role": "assistant", "content": response.content})

        # Claude may request several tools at once; answer all of them in ONE user message.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            function = TOOL_FUNCTIONS.get(block.name)
            if function is None:
                tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                     "content": f"Unknown tool: {block.name}", "is_error": True})
                continue
            result = function(**block.input)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})

    return "Stopped: too many tool-calling turns."

print(iot_tool_bot("What is the live status of devices TEMP-042 and HUM-007?"))
```

Notice the details that matter in production:
-   **Every `tool_use` block gets a `tool_result`** with the matching `tool_use_id`, and all results go back in a single user message.
-   **Failures are reported, not hidden.** `is_error: True` tells Claude the call failed so it can adjust.
-   **The loop is bounded** by `max_turns`.

The SDK also includes a tool runner (`client.beta.messages.tool_runner`) that runs this loop for you; writing it by hand once is the best way to understand what it does.

### Structured Outputs

When you need machine-readable output rather than prose, ask for JSON that matches a schema and let the API enforce it:

```python
from pydantic import BaseModel
from typing import Literal

class AlertTriage(BaseModel):
    device_id: str
    category: Literal["Power", "Connectivity", "Sensor", "Security"]
    urgent: bool

result = claude_client.messages.parse(
    model=CLAUDE_MODEL,
    max_tokens=16000,
    messages=[{"role": "user", "content": "Triage this alert: 'GW-01 battery at 4%, shutting down in 10 minutes.'"}],
    output_format=AlertTriage,
)
print(result.parsed_output)
```

Chapter 11 covers structured outputs across all providers in depth.

## Choosing a Model (and a Provider)

Within Claude, the trade-off is the usual one between capability, speed, and cost:

| Model | Use it for |
| --- | --- |
| `claude-opus-5-5` | Hard reasoning, complex agents and coding, high-stakes analysis. The default choice. |
| `claude-sonnet-5-5` | High-volume production work that still needs strong reasoning, at lower cost and latency. |
| `claude-haiku-5-5` | Simple, latency-sensitive, high-volume tasks: classification, extraction, routing. |

Before switching to a cheaper model for cost reasons, try the more capable model at lower `effort`; it is often cheaper overall than a weaker model that needs retries.

Choosing *between providers* is less about fixed strengths and more about measurement. In 2026, OpenAI, Anthropic, and Google all offer reasoning models, million-token contexts, vision, tool use, and structured outputs, and their relative strengths shift with every release. Decide with criteria rather than reputation:

1.  **Quality on your task**, measured on your own examples (Chapter 24).
2.  **Cost per successful task**, not per token: a model that needs fewer retries can be cheaper even at a higher price.
3.  **Latency** requirements, including time to first token.
4.  **Features you depend on** (for example a specific built-in tool, batch processing, or data-residency options).
5.  **Compliance and data handling**: retention policies, regional processing, and contractual terms.

Building a thin abstraction over providers (Chapter 8) keeps you free to switch as the answers change.

## Conclusion

You now have the essentials of the Anthropic Claude API. Its structure differs from OpenAI's in small but important ways.

By adding Claude to your toolkit, you can choose the best model for each job and blend providers to build more capable and resilient applications.

## Key Takeaways

-   `client.messages.create` requires `max_tokens`, which also covers thinking, and returns a list of typed content blocks. Read text by selecting blocks where `block.type == "text"`, never `content[0]`, and check `stop_reason` for `"max_tokens"` and `"refusal"`.
-   Control thinking depth, cost, and latency with `output_config={"effort": ...}`, not `temperature`, which the newest models do not accept. Use `low` for classification and extraction, `high` or above for hard reasoning, and try the more capable model at lower effort before switching to a cheaper model.
-   The system prompt goes in the dedicated `system` parameter. Wrap documents in XML-style tags such as `<document>`, and explain why a rule exists instead of shouting it in capitals.
-   Prompt caching (`cache_control={"type": "ephemeral"}`) bills a repeated prompt prefix at a fraction of the input price. Put stable content first and changing content last, and confirm hits with `usage.cache_read_input_tokens`; prefixes below the minimum size are not cached.
-   In a tool loop, append Claude's full `response.content` to the history, return a `tool_result` for every `tool_use` block in a single user message, report failures with `is_error: True`, and bound the number of turns.
-   PDFs and images are `document` and `image` content blocks placed before the text question, and `messages.parse` with a Pydantic model returns schema-valid structured output.

## Exercises

**1. Warm-up:** Measure prompt caching on the `DocumentAnalyzerBot`. Replace the short `device_manual` with a manual long enough to pass the minimum cacheable size (several thousand tokens), send two questions with `cache_control={"type": "ephemeral"}`, and print `input_tokens`, `cache_creation_input_tokens`, and `cache_read_input_tokens` for each call. Then add the current timestamp to the start of the system prompt and repeat. Done means your output shows a cache write on the first call, a cache read on the second, and zero cache reads once the timestamp breaks the prefix.

**2. Build:** Add a second tool to `iot_tool_bot`, such as `get_battery_history(device_id, days)`, and make `get_live_device_status` raise an exception for unknown IDs instead of returning an error dictionary. Change the loop so that any exception raised by a tool function becomes a `tool_result` with `is_error: True` and the error message, rather than crashing the program. Done means "What is the status and recent battery trend of TEMP-042 and HUM-007?" ends with an answer that explains HUM-007 could not be read, and a per-turn log shows the number of `tool_use` blocks always equals the number of `tool_result` blocks.

**3. Stretch:** Build a labelled set of 15 alerts with the expected `category` and `urgent` values, and run each one through the `AlertTriage` structured-output call at `effort` `"low"` and `"high"`. Record category accuracy, urgency accuracy, median latency, and total input plus output tokens, and count refusals or parse failures as wrong answers. Done means you have a results table that includes cost per correct triage, calculated from the token counts and the current published prices, and a one-sentence recommendation based on it.

# References and Further Reading

- Claude API documentation. https://platform.claude.com/docs/
- Claude models overview (current IDs, context windows, pricing). https://platform.claude.com/docs/en/about-claude/models/overview
- Prompt caching. https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Anthropic Python SDK. https://github.com/anthropics/anthropic-sdk-python
- Building effective agents (Anthropic, 2024). https://www.anthropic.com/engineering/building-effective-agents
- Build with Claude (Anthropic Academy). https://www.anthropic.com/learn/build-with-claude
