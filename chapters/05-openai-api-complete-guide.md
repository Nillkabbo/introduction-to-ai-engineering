# Chapter 5: A Developer's Guide to the OpenAI API

In the previous chapters, we explored the concepts behind AI. Now it's time to roll up our sleeves and become practitioners. This chapter is a hands-on guide to the OpenAI API, the interface that lets your Python applications talk to OpenAI's GPT models. We will move from the simplest possible API call to applications that use function calling, vision, and audio.

By the end of this chapter, you will not just understand the API; you will be able to use it to build real applications.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Make API calls with OpenAI's **Responses API**, and recognize the older Chat Completions style when you meet it.
-   Use instructions (system messages) to give your AI a distinct role.
-   Manage conversation history to give your AI memory.
-   Control output with reasoning effort, output limits, and (where supported) temperature.
-   Implement streaming for real-time, interactive user experiences.
-   Give your AI tools by implementing **function calling**, including multiple tool calls per turn.
-   Analyze images and generate or transcribe audio.

> **A note on versions.** This chapter reflects the OpenAI Python SDK and models as of October 2026. OpenAI recommends the **Responses API** (`client.responses.create`) for new projects. The older **Chat Completions API** (`client.chat.completions.create`) is still supported, and you will find it in most code written before 2025, so we show how the two map onto each other. The former Assistants API was shut down in August 2026; its replacement is the Responses API together with the Conversations API.

## Your First API Call: The "Hello, World!" of AI

Every programming journey begins with a "Hello, World!" example. For the OpenAI API, it's a simple request that asks the model to say hello. This small step contains the core pattern of every interaction you will have with the API.

### The Absolute Minimum Code

```python
import openai

# 1. Initialize the client. It automatically reads the OPENAI_API_KEY environment variable.
client = openai.OpenAI()

# 2. Send a request to the Responses API.
response = client.responses.create(
    model="gpt-6-luna",
    input="Say hello to the AI world!",
)

# 3. Read the generated text.
print(response.output_text)
```

That's it. You've sent a message to a state-of-the-art model and received a response. Let's turn this into a reusable function, using the secure configuration from Chapter 3 and handling errors explicitly.

### A Practical, Reusable Function

```python
import os
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")

def ask_ai(prompt: str, instructions: str | None = None) -> str:
    """Sends a single prompt to the AI and returns the response text."""
    try:
        response = client.responses.create(
            model=MODEL,
            instructions=instructions,
            input=prompt,
        )
        return response.output_text
    except openai.RateLimitError:
        return "The service is busy. Please try again shortly."
    except openai.APIConnectionError:
        return "Could not reach the AI service. Check your network connection."
    except openai.APIStatusError as e:
        return f"The AI service returned an error (HTTP {e.status_code})."

explanation = ask_ai("Explain what an API is in one simple sentence.")
print(f"AI Response: {explanation}")
```

Catching specific exception types lets you react differently to different failures; Chapter 8 builds on this. Note that the SDK already retries rate-limit and server errors twice by default (configurable with `openai.OpenAI(max_retries=...)`), so these handlers only see errors that persisted.

### How the Responses API Maps to Chat Completions

You will meet both styles, so here is the translation:

| Concept | Responses API | Chat Completions API |
| --- | --- | --- |
| Call | `client.responses.create(...)` | `client.chat.completions.create(...)` |
| Prompt | `input="..."` or `input=[{"role": ..., "content": ...}]` | `messages=[{"role": ..., "content": ...}]` |
| System prompt | `instructions="..."` (or a `system`/`developer` message in `input`) | a `system`/`developer` message |
| Output text | `response.output_text` | `response.choices[0].message.content` |
| Output limit | `max_output_tokens=` | `max_completion_tokens=` (formerly `max_tokens`) |
| JSON schema output | `text={"format": {...}}` / `client.responses.parse(...)` | `response_format=...` / `client.chat.completions.parse(...)` |
| Tool definition | `{"type": "function", "name": ..., "parameters": ...}` | `{"type": "function", "function": {"name": ..., "parameters": ...}}` |
| Token usage | `response.usage.input_tokens` / `output_tokens` | `response.usage.prompt_tokens` / `completion_tokens` |

## Instructions: Giving Your AI a Role

By default, the AI acts as a generic helpful assistant. To build specialized applications, you give it a role and a set of rules. In the Responses API you do this with the `instructions` parameter (the equivalent of a system message).

```python
question = "What should I wear for a business meeting?"

print(f"Generic Assistant: {ask_ai(question)}")

print("Fashion Expert:", ask_ai(
    question,
    instructions="You are a world-class fashion consultant. Give confident, specific, modern advice in under 120 words.",
))
```

The instructions change the tone, style, and content of the response. They are your primary tool for tailoring the AI to your application.

## Building Conversations: Giving Your AI Memory

Each API call is independent; the model has no memory of past calls. To hold a conversation, you send the relevant history with each request. You can manage the history yourself, which works with every provider and gives you full control:

```python
class SimpleChatbot:
    def __init__(self, instructions: str = "You are a helpful assistant."):
        self.instructions = instructions
        self.history: list[dict] = []
    
    def chat(self, user_message: str) -> str:
        """Adds a user message to the history and gets the AI's response."""
        self.history.append({"role": "user", "content": user_message})
        
        response = client.responses.create(
            model=MODEL,
            instructions=self.instructions,
            input=self.history,
        )
        
        self.history.append({"role": "assistant", "content": response.output_text})
        return response.output_text

iot_expert_bot = SimpleChatbot("You are a friendly IoT expert.")

print("Bot:", iot_expert_bot.chat("My temperature sensor is giving weird readings."))
print("Bot:", iot_expert_bot.chat("They are all exactly 25.0°C, which seems unlikely."))
print("Bot:", iot_expert_bot.chat("What could cause that?"))
```

Because we send the full history each time, the final answer knows we're talking about a sensor stuck at exactly 25.0°C, a classic sign of a sensor returning a default or cached value.

Alternatively, the Responses API can keep the conversation on OpenAI's side. Pass the previous response's ID and send only the new message:

```python
first = client.responses.create(model=MODEL, input="My temperature sensor reads exactly 25.0°C all day.")
follow_up = client.responses.create(
    model=MODEL,
    previous_response_id=first.id,   # the server supplies the earlier turns
    input="What could cause that?",
)
print(follow_up.output_text)
```

This is convenient and works well with reasoning models, because the model's earlier reasoning is carried forward. The trade-offs: the conversation is stored with OpenAI (consider your data-retention requirements), you are still billed for the earlier turns as input tokens, and the approach is specific to one provider.

## Controlling the Output

The API gives you several "dials" to control the output. Which ones apply depends on whether the model is a *reasoning* model.

-   **`reasoning={"effort": ...}`:** Current GPT models reason before they answer. The effort level (`"low"`, `"medium"`, `"high"`, and on some models `"none"`, `"xhigh"`, or `"max"`) trades answer quality on hard problems against latency and cost. Use low effort for simple extraction or classification, and higher effort for multi-step reasoning, planning, or code.
-   **`max_output_tokens`:** A hard cap on the response length. On reasoning models it includes the hidden reasoning tokens, so setting it too low can produce an empty or cut-off answer. Check `response.status == "incomplete"` to detect truncation.
-   **`temperature`:** Controls randomness in sampling (`0` focused, around `1` more varied). **Reasoning models reject sampling parameters** like `temperature` and `top_p` while reasoning is on. Some models let you turn reasoning off (`reasoning={"effort": "none"}`), and then accept them; others always reason. Check the model's documentation.

```python
def tagline(effort: str) -> str:
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": effort},
        max_output_tokens=2000,
        input="Write a one-sentence tagline for a smart coffee mug.",
    )
    return response.output_text

print("Low effort:", tagline("low"))
print("High effort:", tagline("high"))

# Sampling controls apply only when reasoning is turned off, on models that allow that.
creative = client.responses.create(
    model=MODEL,
    reasoning={"effort": "none"},
    temperature=1.0,
    input="Write a one-sentence tagline for a smart coffee mug.",
)
print("Higher temperature:", creative.output_text)
```

> **Determinism without `temperature=0`.** Older tutorials set `temperature=0` to get "deterministic" answers. Even then, outputs were never guaranteed to be identical, and on reasoning models the option may not exist. To get *predictable* output, constrain it instead: ask for a fixed set of labels, and use Structured Outputs (Chapter 11) so the response must match a schema.

## Streaming Responses: Real-Time Interaction

For a better user experience, especially with longer responses, you can **stream** the output. The API sends events as the response is generated, so you can display text as it arrives.

```python
def stream_a_story():
    """Demonstrates a streaming API call."""
    stream = client.responses.create(
        model=MODEL,
        input="Tell me a short story about a robot who learns to paint.",
        stream=True,
    )
    
    print("AI Story: ", end="")
    for event in stream:
        # Many event types are emitted; text arrives in output_text delta events.
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
    print()

stream_a_story()
```

With reasoning models there can be a pause before the first text event while the model thinks. If that matters for your UI, show a "thinking…" indicator, lower the reasoning effort, or request a reasoning summary (`reasoning={"effort": "medium", "summary": "auto"}`) and stream it.

## Function Calling: Giving Your AI Tools

This is one of the most powerful features of the API. **Function calling** lets you describe your own Python functions to the model; the model can then ask to call them to get information or take actions.

This is how you connect your AI to the outside world: databases, other APIs, or hardware.

The process is a loop:
1.  **The model decides to call a function.** You send a prompt plus a list of tool definitions. Instead of (or in addition to) text, the response contains one or more `function_call` items, for example "call `get_current_weather` with `location='Tokyo'`".
2.  **You execute the function.** Your code runs the real function and sends the result back in a new request, tagged with the call's `call_id`.
3.  **Repeat** until the model answers with text instead of another function call.

### A Weather Bot Example

```python
import json

# Our actual tool. It could be any Python function.
def get_current_weather(location: str) -> dict:
    """A dummy function to get weather data."""
    print(f"--- Calling weather tool for {location} ---")
    if "tokyo" in location.lower():
        return {"location": "Tokyo", "temperature": "15°C", "condition": "Cloudy"}
    if "london" in location.lower():
        return {"location": "London", "temperature": "10°C", "condition": "Rainy"}
    return {"error": f"No weather data for {location}"}

AVAILABLE_TOOLS = {"get_current_weather": get_current_weather}

TOOLS = [{
    "type": "function",
    "name": "get_current_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'San Francisco, CA'"}
        },
        "required": ["location"],
        "additionalProperties": False,
    },
    # strict=True guarantees the arguments match the schema exactly.
    "strict": True,
}]

def weather_bot(user_prompt: str, max_turns: int = 5) -> str:
    conversation: list = [{"role": "user", "content": user_prompt}]

    for _ in range(max_turns):
        response = client.responses.create(model=MODEL, input=conversation, tools=TOOLS)

        # Keep everything the model produced (reasoning, text, function calls) in the history.
        conversation += response.output

        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            return response.output_text

        # The model may request several calls in one turn; answer every one of them.
        for call in function_calls:
            tool = AVAILABLE_TOOLS.get(call.name)
            if tool is None:
                result = {"error": f"Unknown tool: {call.name}"}
            else:
                result = tool(**json.loads(call.arguments))
            conversation.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result),
            })

    return "Stopped: too many tool-calling turns."

print(weather_bot("What's the weather in Tokyo?"))
print(weather_bot("Should I bring an umbrella if I'm going to London or Tokyo today?"))
```

Three details make this production-worthy rather than a demo:

-   **It loops.** A task may need several rounds of tool calls (look something up, then act on it).
-   **It handles every call in a turn.** The second question typically produces two parallel `get_current_weather` calls; every call must receive an output with the matching `call_id`.
-   **It caps the number of turns.** A model stuck in a loop cannot run up an unbounded bill.

Chapter 14 builds on this with a tool registry, permissions, and security.

## Multimodal Capabilities: Vision and Audio

Current GPT models are multimodal: they accept images (and PDF files) alongside text.

### Vision: Analyzing Images

You can pass an image by URL, as a base64 data URL, or as an uploaded file ID.

```python
import base64

def analyze_image(image_url: str, prompt: str) -> str:
    """Asks the model a question about an image."""
    response = client.responses.create(
        model=MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": image_url, "detail": "auto"},
            ],
        }],
    )
    return response.output_text

def local_image_as_data_url(path: str, mime_type: str = "image/png") -> str:
    with open(path, "rb") as f:
        return f"data:{mime_type};base64,{base64.b64encode(f.read()).decode()}"

# print(analyze_image("https://example.com/control-panel.jpg", "Describe this image in detail."))
# print(analyze_image(local_image_as_data_url("dashboard.png"), "Which metrics need attention?"))
```

PDFs work the same way with an `input_file` content part (for example `{"type": "input_file", "file_id": ...}` after uploading with `client.files.create(..., purpose="user_data")`), so you can ask questions about manuals and reports directly.

### Audio: Speech-to-Text and Text-to-Speech

The API also provides dedicated audio models:

-   **Transcription (speech-to-text):** `gpt-4o-transcribe` and `gpt-4o-mini-transcribe` (the older `whisper-1` is still available).
-   **Speech generation (text-to-speech):** `gpt-4o-mini-tts`, which also accepts `instructions` about tone and delivery.

Audio model names change on their own schedule (OpenAI announced in October 2026 that many GPT-4-series models will be removed in early 2027), so check the deprecations page and the current model list before you pin one.

```python
def text_to_speech_demo():
    """Converts text into an audio file."""
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input="Warning: pump P-07 pressure has exceeded the safe limit.",
        instructions="Speak calmly and clearly, like a control-room announcement.",
    ) as response:
        response.stream_to_file("alert.mp3")
    print("Audio saved to alert.mp3")

def transcribe_demo(path: str) -> str:
    """Transcribes an audio file to text."""
    with open(path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=audio_file)
    return transcript.text

text_to_speech_demo()
# print(transcribe_demo("technician_voice_note.mp3"))
```

For live, two-way voice conversations, OpenAI also offers a dedicated real-time voice API. It is out of scope here, but it builds on the same ideas.

## Conclusion

You now have a practical understanding of the OpenAI API. You've moved from a simple "Hello, World!" to function calling, streaming, and multimodal analysis.

These are the building blocks you will use in every AI application you create.

## Key Takeaways

-   The core call is `client.responses.create(model=MODEL, instructions=..., input=...)`, read with `response.output_text`. Use the mapping table to translate older Chat Completions code (`messages`, `choices[0].message.content`, `max_completion_tokens`).
-   The model is stateless. Either resend the full history on every call, or chain turns with `previous_response_id`, which is convenient but stores the conversation with OpenAI, still bills earlier turns as input, and ties you to one provider.
-   On reasoning models, trade quality against cost and latency with `reasoning={"effort": ...}`, and cap length with `max_output_tokens`, which includes reasoning tokens (check `response.status == "incomplete"`). `temperature` is only accepted when reasoning is off; for predictable output, constrain it with fixed labels and Structured Outputs.
-   Stream with `stream=True` and print only `response.output_text.delta` events. Expect a pause before the first text on reasoning models.
-   A correct function-calling loop appends `response.output` to the history, answers every `function_call` with a `function_call_output` carrying the matching `call_id`, repeats until no calls remain, and caps the number of turns.
-   Images and PDFs are `input_image` and `input_file` content parts; audio uses dedicated transcription and speech models whose names change, so check the deprecations page before pinning one.

## Exercises

**1. Warm-up:** Modify `SimpleChatbot.chat` to record `response.usage.input_tokens` and `response.usage.output_tokens` after every call, then hold a five-turn conversation with the IoT expert bot. Print a small table of turn number, input tokens, and output tokens. Done means the table shows input tokens growing every turn, which confirms that the full history is resent and billed each time.

**2. Build:** Turn the weather bot into a device bot with two tools, `get_device_status(device_id)` and `get_battery_level(device_id)`, backed by a dictionary of a few fake devices. Make the dispatch robust: wrap each tool call in `try/except` and return `{"error": ...}` when the function raises or the device ID is unknown, instead of crashing the loop. Done means "Compare the battery levels of TEMP-01, TEMP-02 and HUM-99" returns one answer that mentions HUM-99 could not be found, and a log line per call shows that every `call_id` received exactly one `function_call_output`.

**3. Stretch:** Build a labelled set of 10 IoT alerts, each tagged Power, Connectivity, or Sensor, and classify every alert with `reasoning={"effort": "low"}` and again with `"high"`, instructing the model to answer with the label only. For each effort level, record accuracy (an answer that is not exactly one of the three labels counts as wrong), median latency measured with `time.perf_counter()`, and total `output_tokens`, and treat `response.status == "incomplete"` as a failure. Done means you have a two-row results table and a one-sentence, data-backed recommendation for which effort level to use for this task.

# References and Further Reading

- OpenAI API documentation. https://platform.openai.com/docs/
- OpenAI Python SDK (source, examples, changelog). https://github.com/openai/openai-python
- OpenAI API changelog (new models and deprecations). https://developers.openai.com/api/docs/changelog
- Assistants API migration guide (to the Responses API). https://developers.openai.com/api/docs/assistants/migration
