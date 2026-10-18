# Chapter 11: Mastering Structured Output

In software development, data is only useful if it's predictable. A web application expects a JSON object, a database expects typed fields, and a configuration system expects a specific format. Large language models generate free-form text, and that flexibility becomes a liability when your code needs to process the response.

This chapter is about turning model output into reliable, machine-readable structure. We will go from "please answer in JSON" to schema-enforced **Structured Outputs**, validate the results with Pydantic, and apply the same discipline to generating code and configuration.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Explain the difference between prompting for JSON, JSON mode, and schema-enforced Structured Outputs.
-   Use Structured Outputs with OpenAI, Anthropic, and Google, driven by Pydantic models.
-   Validate business rules that a schema can't express, and recover from failures.
-   Generate code under clear constraints, and know when templating should be deterministic instead.
-   Build a validated IoT device configuration generator.

## The Problem: The Unpredictability of Unstructured Text

```python
import json
import openai

client = openai.OpenAI()
MODEL = "gpt-6-luna"

sensor_reading = "device: TEMP-042, temp: 85°C, battery: 15%, signal: -92dBm"

response = client.responses.create(
    model=MODEL,
    input=f"Analyze this IoT sensor data and tell me what's wrong and what to do: {sensor_reading}",
)
print(response.output_text)
```

The response may be helpful for a human, but for a program it's a nightmare. How would your code reliably extract the severity, the issues, and the recommended action? Regular expressions over prose break the moment the wording changes.

## Level 1: Prompting for Structure

The first step is to describe the structure you want in the prompt:

```python
def structured_analysis(sensor_data: str) -> dict:
    prompt = f"""
Analyze the following IoT sensor data.
<data>{sensor_data}</data>

Respond with only a JSON object with this structure:
{{
  "device_id": string,
  "health_status": "Healthy" | "Warning" | "Critical",
  "issues": [{{"component": "Temperature" | "Battery" | "Connectivity", "description": string}}],
  "recommended_action": string
}}
"""
    text = client.responses.create(model=MODEL, input=prompt).output_text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse AI response as JSON.", "raw": text}

print(json.dumps(structured_analysis(sensor_reading), indent=2))
```

This usually works, but "usually" is the problem. The model may wrap the JSON in a code fence, add a friendly sentence before it, misspell a key, or invent a fourth status.

## Level 2: JSON Mode

**JSON mode** constrains the model to produce syntactically valid JSON. In the Responses API you enable it with `text={"format": {"type": "json_object"}}` (Chat Completions uses `response_format={"type": "json_object"}`). The prompt must still mention JSON and describe the fields.

```python
response = client.responses.create(
    model=MODEL,
    input=f"Analyze this IoT data and respond in JSON with keys device_id, status, issues, recommendation.\n\nData: {sensor_reading}",
    text={"format": {"type": "json_object"}},
)
data = json.loads(response.output_text)
```

JSON mode removes *syntax* errors, but it says nothing about the *schema*: keys can still be missing, misnamed, or of the wrong type. One failure mode remains even for syntax: if the output hits the token limit, the JSON is cut off, so check that the response completed (`response.status == "completed"`).

## Level 3: Structured Outputs (Schema-Enforced)

**Structured Outputs** go further: you give the API a JSON Schema, and generation is *constrained* so that the output always matches it, with every required key present, every enum value valid, and every type correct. All three major providers support this, and their SDKs accept a **Pydantic model** directly, so one Python class serves as your schema, your parser, and your type.

```python
from typing import Literal
from pydantic import BaseModel, Field

class DeviceIssue(BaseModel):
    component: Literal["Temperature", "Battery", "Connectivity"]
    description: str

class DeviceAnalysis(BaseModel):
    device_id: str
    health_status: Literal["Healthy", "Warning", "Critical"]
    issues: list[DeviceIssue]
    health_score: int = Field(ge=0, le=100, description="0 = failed, 100 = perfect")
    recommended_action: str
```

### OpenAI

```python
def analyze_openai(sensor_data: str) -> DeviceAnalysis:
    response = client.responses.parse(
        model=MODEL,
        instructions="You are an IoT diagnostics assistant.",
        input=f"Analyze this sensor reading: {sensor_data}",
        text_format=DeviceAnalysis,
    )
    return response.output_parsed       # already a validated DeviceAnalysis instance

analysis = analyze_openai(sensor_reading)
print(analysis.health_status, analysis.health_score)
for issue in analysis.issues:
    print("-", issue.component, ":", issue.description)
```

### Anthropic (Claude)

```python
import anthropic

claude = anthropic.Anthropic()
CLAUDE_MODEL = "claude-opus-5-5"

def analyze_claude(sensor_data: str) -> DeviceAnalysis:
    response = claude.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": f"Analyze this sensor reading: {sensor_data}"}],
        output_format=DeviceAnalysis,
    )
    return response.parsed_output
```

### Google (Gemini)

```python
from google import genai
from google.genai import types

gemini = genai.Client()
GEMINI_MODEL = "gemini-3.8-flash"

def analyze_gemini(sensor_data: str) -> DeviceAnalysis:
    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Analyze this sensor reading: {sensor_data}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DeviceAnalysis,
        ),
    )
    return response.parsed
```

Structured Outputs is the right default whenever your code consumes model output. A few things to know:

-   **Schema features are limited.** Providers support a large subset of JSON Schema, not all of it. Keep schemas simple: objects, arrays, enums (`Literal`), strings, numbers, booleans, and optional fields. Very long enums or deeply recursive schemas may be rejected; the SDK error tells you which keyword is unsupported.
-   **Schemas are guidance for content, too.** Field names and `description`s are part of the prompt; a field called `recommended_action` with a good description gets better content than one called `x`.
-   **A valid shape is not a correct answer.** The model can still put a wrong value in a well-formed field. Structure guarantees parseability, not truth.
-   **Refusals and truncation still exist.** If the model refuses or runs out of tokens, the parsed result can be missing; check for it rather than assuming success.

## Pydantic for Business Rules

Structured Outputs guarantees the shape. Your application usually has additional rules that a JSON Schema can't express, and Pydantic validators are the natural place for them:

```python
from pydantic import ValidationError, model_validator

class ValidatedAnalysis(DeviceAnalysis):
    @model_validator(mode="after")
    def status_matches_score(self):
        if self.health_status == "Critical" and self.health_score > 40:
            raise ValueError("A Critical device cannot have a health score above 40")
        if self.health_status == "Healthy" and self.issues:
            raise ValueError("A Healthy device should not list issues")
        return self

def analyze_with_rules(sensor_data: str, max_attempts: int = 2) -> ValidatedAnalysis:
    feedback = ""
    for _ in range(max_attempts):
        raw = analyze_openai(sensor_data + feedback)
        try:
            return ValidatedAnalysis.model_validate(raw.model_dump())
        except ValidationError as e:
            # Feed the specific rule violation back so the next attempt can fix it.
            feedback = f"\n\nYour previous answer violated these rules: {e.errors()[0]['msg']}. Fix this."
    raise RuntimeError("Could not produce an analysis that satisfies the business rules")
```

This two-layer design is the production standard: **schema enforcement** for structure, **code-level validation** for meaning, and a **bounded retry** that tells the model exactly what was wrong.

## Pattern: Constrained Code Generation

Generating structured data is one thing; generating functional, correct, and secure code is another. When asking a model to write code, give it explicit requirements and the context it can't guess.

```python
def generate_device_driver(device_spec: dict) -> str:
    """Generates a Python device driver class based on a specification."""
    prompt = f"""
Generate a complete Python class for an IoT device based on this specification.

<specification>
{json.dumps(device_spec, indent=2)}
</specification>

Requirements:
1.  Name the class `ModbusTempSensor` and use the `pymodbus` library (version 3.x).
2.  Full type hints and docstrings for the class and all public methods.
3.  Handle I/O errors (timeouts, CRC errors) explicitly; never swallow exceptions silently.
4.  Use the `logging` module, not print().
5.  Follow PEP 8.
Return only the code.
"""
    return client.responses.create(
        model="gpt-6-sol",                 # a stronger model is worth it for code
        input=prompt,
        reasoning={"effort": "high"},
    ).output_text

sensor_spec = {
    "device_type": "Modbus Temperature Sensor",
    "protocol": "Modbus RTU",
    "connection": "Serial (RS-485), 9600 baud, 8N1",
    "registers": {
        "temperature": {"address": 4001, "type": "int16", "scale_factor": 0.1},
        "humidity": {"address": 4002, "type": "uint16", "scale_factor": 0.1},
    },
}
print(generate_device_driver(sensor_spec))
```

Notice the requirement to name the library and its version: models otherwise mix APIs from different library versions. Generated code is a draft: run it, test it, lint it, and review it like a colleague's pull request before it goes anywhere near production.

### Template Filling: Use Code, Not the Model

A common mistake is to ask a model to fill placeholders in a code template. If the values are known (a class name, a route prefix), plain string formatting or a template engine does the job deterministically, instantly, and for free. Use the model only for the part that actually needs intelligence, such as the body of a function, and insert its output into the template.

```python
from string import Template

API_ENDPOINT_TEMPLATE = Template('''
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/$endpoint_prefix", tags=["$tag_name"])

class ${model_name}Response(BaseModel):
    id: str
    status: str

@router.get("/$endpoint_path/{item_id}", response_model=${model_name}Response)
async def get_$singular_name(item_id: str):
    """$endpoint_description"""
$body
''')

def generate_endpoint(params: dict, logic_description: str) -> str:
    # Only the business logic needs the model.
    body = client.responses.create(
        model=MODEL,
        input=(
            "Write only the indented body (4 spaces) of an async FastAPI handler that receives `item_id: str`. "
            f"It should: {logic_description}. Use a placeholder `db` object with an async `get_device(id)` method. "
            "Return a dict with keys 'id' and 'status', or raise HTTPException(404). No code fences."
        ),
    ).output_text
    return API_ENDPOINT_TEMPLATE.substitute(params, body=body.rstrip())

params = {
    "endpoint_prefix": "devices",
    "tag_name": "Devices",
    "model_name": "Device",
    "endpoint_path": "status",
    "singular_name": "device_status",
    "endpoint_description": "Gets the current status of a specific IoT device.",
}
print(generate_endpoint(params, "look up the device and return its status"))
```

`string.Template` uses `$name` placeholders, so the literal braces in the FastAPI path (`{item_id}`) need no escaping. For larger templates, Jinja2 is the standard choice. After generation, run a syntax check (`ast.parse`) and your linters before using the result.

## Practical Project: A Validated IoT Configuration Generator

Let's combine these patterns: a high-level description of a deployment goes in, and a validated configuration object comes out.

```python
class SensorConfig(BaseModel):
    sensor_type: Literal["temperature", "humidity", "pressure"]
    reading_interval_seconds: int = Field(ge=10, description="How often to sample, in seconds (minimum 10)")

class NetworkConfig(BaseModel):
    connection_type: Literal["WiFi", "Cellular", "LoRaWAN"]
    wifi_ssid: str | None = Field(default=None, description="Required when connection_type is WiFi")

class DeviceConfiguration(BaseModel):
    device_id_prefix: str
    location: str
    network: NetworkConfig
    sensors: list[SensorConfig]
    firmware_version: str = "1.0.0"

    @model_validator(mode="after")
    def check_rules(self):
        if self.network.connection_type == "WiFi" and not self.network.wifi_ssid:
            raise ValueError("wifi_ssid is required for WiFi connections")
        if not self.sensors:
            raise ValueError("at least one sensor must be configured")
        return self

class ConfigurationGenerator:
    def generate(self, requirements: str, max_attempts: int = 2) -> DeviceConfiguration:
        prompt = f"Generate an IoT device configuration for these requirements:\n<requirements>{requirements}</requirements>"
        for attempt in range(1, max_attempts + 1):
            try:
                # Parsing runs the Pydantic validators too, including the cross-field rules.
                response = client.responses.parse(model=MODEL, input=prompt, text_format=DeviceConfiguration)
            except ValidationError as e:
                print(f"Attempt {attempt} broke a business rule: {e.errors()[0]['msg']}")
                prompt += f"\n\nA previous attempt was invalid: {e.errors()[0]['msg']}. Correct it."
                continue
            if response.output_parsed is None:
                raise RuntimeError(f"No configuration returned (status: {response.status})")
            return response.output_parsed
        raise RuntimeError("Could not generate a valid configuration")

generator = ConfigurationGenerator()
config = generator.generate(
    "Deploy temperature and humidity sensors in our cold storage warehouse. "
    "They should report every 5 minutes and connect via our 'Warehouse-IoT' WiFi network."
)
print(config.model_dump_json(indent=2))   # reading_interval_seconds should be 300
```

The structure is guaranteed by the API, the business rules by Pydantic, and failures are retried with specific feedback, with a hard limit. Because the result is a typed object, the rest of your code (deployment scripts, a database insert) can use it with confidence.

## Conclusion

Structured output is the bridge between the probabilistic world of LLMs and the deterministic world of software.

Remember these key practices:
-   **Prefer Structured Outputs** (schema-enforced) over JSON mode, and JSON mode over plain prompting.
-   **Define the schema once in Pydantic** and let the SDK enforce and parse it.
-   **Validate business rules in code**, and retry with specific feedback, a bounded number of times.
-   **Constrain code generation** with explicit requirements, and test generated code like any other code.
-   **Don't use a model for deterministic work** such as template filling.

You are now equipped to build AI systems that don't just talk, but speak the precise, structured language your applications understand.

## Key Takeaways

-   Prompting for JSON can fail on syntax, JSON mode (`text={"format": {"type": "json_object"}}`) fixes syntax but not keys or types, and Structured Outputs constrains generation to your exact schema.
-   One Pydantic model can serve as schema, parser, and type across providers: `client.responses.parse(text_format=...)` with `output_parsed` for OpenAI, `messages.parse(output_format=...)` with `parsed_output` for Claude, and `response_schema` with `response.parsed` for Gemini.
-   A schema guarantees shape, not truth. Check `response.status` and a missing parsed result for truncation and refusals, and keep schemas to the simple JSON Schema subset providers support.
-   Put rules a schema cannot express (cross-field consistency such as "Critical means score at most 40") in a Pydantic `model_validator`, and retry a bounded number of times with the specific error message as feedback.
-   For code generation, name the library and version and list explicit requirements; treat the output as a draft to run, lint, and review.
-   Fill known values with `string.Template` or Jinja2, not a model; use the model only for the part that needs judgement, then check the result with `ast.parse`.

## Exercises

**1. Warm-up:** Write tests for `ValidatedAnalysis` that never call a model: build instances directly with `model_validate` and check that a Critical device with a score of 80 and a Healthy device with an issue both raise `ValidationError`, while a consistent Warning device passes. Add one new rule of your own (for example, a Warning device must list at least one issue) with a test for it. You are done when all your tests run in under a second with no API key set.

**2. Build:** Run `ConfigurationGenerator` on six requirement strings of your own, including one that names WiFi but no network name, one that asks for a 5-second interval, and one that asks for a sensor type the schema does not allow. Log for each request how many attempts it took, which validation messages came back, and whether it ended in a valid configuration or a `RuntimeError`. You are done when every request ends in exactly one of those two outcomes (no unhandled exception) and you have a table of attempts and outcomes.

**3. Stretch:** Make `generate_endpoint` self-checking. After substituting the model's body into the template, run `ast.parse` on the result; on a `SyntaxError`, retry up to two more times with the error line and message as feedback, and raise if all attempts fail. Run it 10 times and count first-try successes, successes after retry, and failures. You are done when the counts add up to 10 and every returned string parses with `ast.parse`.

# References and Further Reading

- Structured Outputs guide (OpenAI). https://platform.openai.com/docs/guides/structured-outputs
- Structured outputs (Claude documentation). https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Structured output (Gemini API documentation). https://ai.google.dev/gemini-api/docs/structured-output
- Pydantic documentation: validators. https://docs.pydantic.dev/latest/concepts/validators/
- 4. Structured Output (Taming LLMs). https://www.tamingllms.com/notebooks/structured_output.html

<!-- chapter-nav -->

---

[← Chapter 10: Advanced Prompting Strategies](10-advanced-prompting-strategies.md) | [Table of Contents](../README.md) | [Chapter 12: Domain-Specific Prompting →](12-domain-specific-prompting.md)
