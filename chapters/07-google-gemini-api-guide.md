# Chapter 7: Mastering the Google Gemini API

Having explored the APIs from OpenAI and Anthropic, we now turn to the third major player: Google. Gemini models were designed as multimodal from the start: a single request can mix text, images, audio, video, and PDFs. That makes Gemini a strong choice for applications that need to understand the world through more than text.

This chapter is a guide to the **Google Gen AI SDK** (`google-genai`). We'll go from a first request to systems that analyze video and technical documents and call your own tools.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Make Gemini API calls with the `google-genai` SDK.
-   Guide Gemini's behavior with system instructions and control its thinking.
-   Process and analyze video, audio, and PDF content.
-   Use automatic function calling to connect Gemini to your own Python functions.
-   Build an industrial IoT predictive maintenance assistant that combines diagrams and live data.

> **Version note (October 2026).** Earlier tutorials, including the first edition of this book, used the `google-generativeai` package (`import google.generativeai as genai`, `genai.GenerativeModel(...)`). That SDK is deprecated and the Gemini 1.x models it was usually paired with are retired. Use `google-genai` (`from google import genai`). Google's newest interface, the **Interactions API**, became the recommended default for new projects in mid-2026, and new agentic features launch there first. The `generate_content` interface used in this chapter remains fully supported and is still the most complete for Python (for example, automatic function calling and batch processing). The concepts carry over directly.

## Your First Gemini API Call

### The Minimum Code

Install the SDK and create an API key in Google AI Studio. Store it as `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in your environment or `.env` file.

```bash
pip install google-genai
```

The simplest interaction looks like this:

```python
from google import genai

# The client reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Hello, Gemini!",
)

print(response.text)
```

The pattern is a single `client` object with service namespaces (`client.models`, `client.chats`, `client.files`), the same shape as the OpenAI and Anthropic SDKs.

### A Reusable Function

```python
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
client = genai.Client()
GEMINI_MODEL = "gemini-3.8-flash"     # Fast and cost-effective; "gemini-4-argon" for the hardest tasks.

def ask_gemini(prompt: str, system_instruction: str | None = None) -> str:
    """Sends a single prompt to Gemini and returns the text response."""
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        )
    except errors.APIError as e:
        return f"Gemini API error ({e.code}): {e.message}"
    return response.text

print(ask_gemini("In simple terms, what is multimodal AI?"))
```

## System Instructions and Thinking

Like the other providers, Gemini accepts a high-level directive, here called `system_instruction`, passed in the request's `config`.

```python
iiot_expert_instruction = """
You are an expert Industrial IoT (IIoT) engineer specializing in predictive maintenance for heavy machinery. Your analysis should always consider:
- Safety implications first and foremost.
- The impact on production uptime.
- Cost-effectiveness of proposed solutions.
- The harsh operating environment (vibration, dust, temperature extremes).
Provide clear, actionable, and prioritized recommendations.
"""

user_problem = "Vibration sensors on our main conveyor motor are showing a 30% increase in amplitude over the past week. What should be our plan?"

print(ask_gemini(user_problem, system_instruction=iiot_expert_instruction))
```

Current Gemini models also *think* before answering. You can tune how much with a thinking level (low for fast, simple tasks; high for complex reasoning):

```python
response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents="Plan a 3-step diagnostic sequence for an intermittently failing Modbus sensor.",
    config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="high")),
)
print(response.text)
```

## Building Conversations with Chat Sessions

The SDK provides chat sessions that keep the history for you:

```python
class GeminiChatbot:
    def __init__(self, system_instruction: str = "You are a helpful assistant."):
        self.chat = client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        )

    def send(self, user_message: str) -> str:
        """Sends a message; the session appends both turns to its history."""
        response = self.chat.send_message(user_message)
        return response.text

bot = GeminiChatbot("You are an IoT troubleshooting expert.")
print(bot.send("My smart thermostat is stuck at 15°C."))
print(bot.send("I've already tried restarting it. What should I check next?"))
```

The session object is convenient, but remember that it lives in your process's memory. In a web application with several servers, you would store the history in a database and pass it in with `history=` (Chapter 20).

## Multimodal Inputs

Gemini accepts a list of different content types in a single request.

### Analyzing Images

For small files, pass the bytes inline:

```python
def analyze_iot_setup(image_path: str, prompt: str, mime_type: str = "image/jpeg") -> str:
    """Analyzes a photo of an IoT installation."""
    with open(image_path, "rb") as f:
        image_part = types.Part.from_bytes(data=f.read(), mime_type=mime_type)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[image_part, prompt],
    )
    return response.text

# prompt = "This is a photo of one of our IoT control panels. Identify any potential safety hazards or incorrect wiring."
# print(analyze_iot_setup("iot_panel.jpg", prompt))
```

### Video, Audio, and PDFs with the Files API

Larger media (video, long audio, big PDFs) should be uploaded once with the Files API and then referenced. Uploaded files are processed asynchronously, so wait until the file is `ACTIVE` before using it.

```python
import time

def upload_and_wait(path: str, poll_seconds: int = 5) -> types.File:
    """Uploads a file and blocks until Gemini has finished processing it."""
    uploaded = client.files.upload(file=path)
    while uploaded.state == types.FileState.PROCESSING:
        time.sleep(poll_seconds)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state == types.FileState.FAILED:
        raise RuntimeError(f"Processing failed for {path}")
    return uploaded

def analyze_industrial_video(video_path: str, prompt: str) -> str:
    """Uploads and analyzes a video file to identify operational issues."""
    video_file = upload_and_wait(video_path)
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=[video_file, prompt])
        return response.text
    finally:
        client.files.delete(name=video_file.name)   # uploaded files also expire automatically

def summarize_pdf_manual(pdf_path: str) -> str:
    """Uploads a PDF and asks Gemini to summarize it."""
    pdf_file = upload_and_wait(pdf_path, poll_seconds=2)
    prompt = "Summarize this technical manual. Extract the key specifications, safety warnings, and the main troubleshooting steps."
    return client.models.generate_content(model=GEMINI_MODEL, contents=[pdf_file, prompt]).text

# print(analyze_industrial_video("conveyor_belt.mp4",
#     "Is the motion smooth? Note timestamps (MM:SS) of any signs of mechanical stress."))
# print(summarize_pdf_manual("sensor_manual.pdf"))
```

The same `upload_and_wait` pattern works for audio files: Gemini can transcribe speech and also reason about non-speech sounds, such as a motor that has started to grind.

## Function Calling with Gemini

Gemini supports function calling, and the Python SDK can run the whole loop automatically: pass ordinary Python functions as `tools`, and the SDK generates the schemas from their type hints and docstrings, calls the functions when the model asks, and sends the results back.

### Automatic Function Calling

```python
def get_sensor_reading(sensor_id: str) -> dict:
    """Gets the current reading for a specific IoT sensor.

    Args:
        sensor_id: The sensor's ID, for example 'TEMP-01' or 'VIBR-07'.
    """
    print(f"--- TOOL EXECUTED: get_sensor_reading({sensor_id}) ---")
    if "TEMP" in sensor_id:
        return {"value": 25.5, "unit": "°C"}
    if "VIBR" in sensor_id:
        return {"value": 1.2, "unit": "g"}
    return {"error": "Sensor not found"}

def set_device_state(device_id: str, state: str) -> dict:
    """Sets the state of a device.

    Args:
        device_id: The device to change, for example 'FAN-01'.
        state: Either 'on' or 'off'.
    """
    print(f"--- TOOL EXECUTED: set_device_state({device_id}, {state}) ---")
    return {"device_id": device_id, "status": "success", "new_state": state}

tool_config = types.GenerateContentConfig(tools=[get_sensor_reading, set_device_state])

response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents="What is the current reading for sensor TEMP-01?",
    config=tool_config,
)
print(response.text)

chat = client.chats.create(model=GEMINI_MODEL, config=tool_config)
print(chat.send_message("Please turn on the cooling fan FAN-01.").text)
```

Behind the scenes the SDK performs the same loop you built by hand in Chapters 5 and 6: send the prompt, receive a function-call request, run your function, send back the result, and return the final answer.

Automatic execution is convenient, but notice what it gives up: `set_device_state` runs without any confirmation step. For read-only tools that's fine; for tools that change the physical world, disable automatic calling (`automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)`) and handle the calls yourself so you can validate arguments, check permissions, log, and ask a human when needed (Chapter 14).

## Building a Predictive Maintenance Assistant with Gemini

Let's combine these capabilities. The assistant analyzes a technical diagram, looks up live sensor data and maintenance history through tools, and recommends actions.

```python
def get_maintenance_history(equipment_id: str) -> list[dict]:
    """Retrieves the maintenance history for a piece of equipment.

    Args:
        equipment_id: The equipment ID, for example 'MOTOR-007'.
    """
    print(f"--- TOOL: get_maintenance_history({equipment_id}) ---")
    return [
        {"date": "2025-11-10", "action": "Replaced bearing"},
        {"date": "2026-03-22", "action": "Lubricated motor"},
    ]

maintenance_config = types.GenerateContentConfig(
    system_instruction="You are a predictive maintenance expert for industrial machinery.",
    tools=[get_sensor_reading, get_maintenance_history],
)

def predictive_maintenance_analysis(equipment_id: str, diagram_path: str) -> str:
    """Performs an analysis using a diagram plus live data from tools."""
    with open(diagram_path, "rb") as f:
        diagram = types.Part.from_bytes(data=f.read(), mime_type="image/png")

    prompt = f"""
    Analyze the attached technical diagram for equipment {equipment_id}.
    Then use your tools to get its current vibration reading (sensor VIBR-{equipment_id})
    and its maintenance history.

    Based on all three sources (diagram, live data, history), give a predictive
    maintenance assessment and specific recommended actions.
    """
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[diagram, prompt],
        config=maintenance_config,
    )
    return response.text

# print(predictive_maintenance_analysis("MOTOR-007", "motor_diagram.png"))
```

This showcases Gemini's multimodality: text instructions and a diagram in the same request, plus live data fetched through tools, all combined in one recommendation.

## Conclusion

Google's Gemini API, accessed through the `google-genai` SDK, is a powerful tool for multimodal applications. Its handling of mixed inputs, especially long video and audio, sets it apart.

You have learned to:
-   Call Gemini with `client.models.generate_content`, using the current SDK.
-   Use `system_instruction` and thinking settings to guide the model.
-   Manage conversations with chat sessions.
-   Analyze images inline, and video, audio, and PDFs through the Files API.
-   Use automatic function calling, and know when to turn it off.

With Gemini alongside OpenAI and Anthropic, you have a complete toolkit and can choose the provider and model that best fit each task.

# References and Further Reading

- Gemini API documentation (Google AI for Developers). https://ai.google.dev/gemini-api/docs
- Gemini models (current model codes and limits). https://ai.google.dev/gemini-api/docs/models
- Google Gen AI Python SDK documentation. https://googleapis.github.io/python-genai/
- Interactions API overview. https://ai.google.dev/gemini-api/docs/interactions-overview
- Gemini API release notes. https://ai.google.dev/gemini-api/docs/changelog
