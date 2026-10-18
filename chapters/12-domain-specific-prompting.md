# Chapter 12: Domain-Specific Prompting

So far, we have explored universal prompting strategies that work across a wide range of tasks. However, to unlock the highest levels of performance and create truly expert-level AI systems, we must move from general instructions to **domain-specific prompting**. This involves embedding deep, industry-specific knowledge, terminology, and workflows directly into your prompts.

Think of it as the difference between hiring a generalist and hiring a specialist. A generalist can give you a decent overview, but a specialist—a manufacturing engineer, a biomedical data analyst, a cybersecurity expert—will provide insights that are more nuanced, relevant, and actionable because they understand the specific context, constraints, and objectives of their field. This chapter will teach you how to "inject" that expertise into your AI.

### Learning Objectives

By the end of this chapter, you will be able to:

- Craft prompts tailored to specific industries like manufacturing, healthcare, and finance.
- Generate high-quality, domain-specific technical documentation.
- Build sophisticated code generation workflows that adhere to industry best practices.
- Design AI-powered data analysis systems that understand domain-specific KPIs.
- Develop effective, context-aware customer service automation for specialized products.
- Create detailed IoT device troubleshooting guides using embedded domain knowledge.

## The Power of Context: Generic vs. Domain-Specific Prompts

The core idea behind domain-specific prompting is to provide the AI with the rich context it needs to reason like an expert in a particular field. Let's see a direct comparison.

### A Generic Analysis

A generic prompt asks a general question and receives a general answer.

```python
import json
import openai

client = openai.OpenAI()
MODEL = "gpt-6-luna"

def ask(prompt: str, instructions: str | None = None) -> str:
    return client.responses.create(model=MODEL, instructions=instructions, input=prompt).output_text

def generic_analysis(data: str) -> str:
    """A generic prompt that lacks specific domain context."""
    return ask(f"Analyze this data and provide insights: {data}")
```

### A Domain-Specific Analysis

A domain-specific prompt imbues the AI with a persona and asks it to consider factors relevant to that domain.

```python
def domain_specific_analysis(data: str, domain: str) -> str:
    """A prompt that provides the AI with a specific expert role and context."""
    
    # Define expert personas for different domains
    domain_personas = {
        "manufacturing": "You are a manufacturing process engineer with 15 years of experience in lean manufacturing and quality control.",
        "healthcare": "You are a biomedical data analyst specializing in patient monitoring device data and regulatory compliance."
    }
    persona = domain_personas.get(domain, "You are a technical expert.")
    
    prompt = f"""
Analyze the following operational data.

<data>
{data}
</data>

Provide your analysis, focusing specifically on:
- Industry-specific Key Performance Indicators (KPIs).
- Potential safety and regulatory implications.
- Opportunities for operational efficiency improvements relevant to the {domain} sector.
- Actionable insights tailored to a {domain} context.
"""
    # The persona goes in the instructions (system prompt); the task and data go in the input.
    return ask(prompt, instructions=persona)
```

Let's compare the outputs using some sample factory data.

```python
factory_data = """
Production Line 3 Status:
- Output: 480 units/hour (target: 500)
- Defect Rate: 1.5% (target: <1%)
- Motor Temperature: 85°C (safe limit: 90°C)
"""

print("--- Generic Analysis ---")
print(generic_analysis(factory_data))

print("\n--- Manufacturing-Specific Analysis ---")
print(domain_specific_analysis(factory_data, "manufacturing"))
```

**Analysis Results:**

The generic analysis will correctly note that output is below target and the defect rate is high. The manufacturing-specific analysis will more likely discuss concepts like Overall Equipment Effectiveness (OEE), connect the reduced throughput to the motor running close to its limit, and frame the defect rate in quality-control terms. It speaks the language of the domain.

The persona is only part of the effect, though. Most of the improvement comes from the explicit list of what to focus on, which is domain knowledge *you* supplied. The more of your organization's real context you provide (targets, thresholds, past incidents, procedures), the more expert the answer becomes. Taken to its conclusion, that idea becomes retrieval-augmented generation (Chapter 27).

> **High-stakes domains.** In regulated fields such as healthcare, finance, and safety engineering, a domain persona makes the output *sound* authoritative, but it doesn't make it correct or compliant. Keep a qualified human in the loop, ground answers in approved sources, and say clearly in your product that the output is AI-generated.

## Domain 1: Generating Technical Documentation

AI is a powerful tool for generating documentation, but to create truly useful docs, you must provide it with the right technical and audience context.

### Generating API Documentation

Let's generate developer-focused documentation for an IoT API.

```python
def generate_api_documentation(api_spec: dict) -> str:
    """Generates comprehensive API documentation from a specification."""
    
    prompt = f"""
You are a senior technical writer specializing in creating clear, developer-friendly API documentation.

Generate a complete API reference document in Markdown format for the following API specification.

**API Specification:**

{json.dumps(api_spec, indent=2)}


**Documentation Requirements:**
1. **Overview:** Explain the API's purpose, base URL, version, and authentication method.
2. **Endpoints:** For each endpoint, detail the HTTP method, path, description, parameters, request body (if any), and example responses for both success (200 OK) and common errors (e.g., 404 Not Found, 401 Unauthorized).
3. **Code Examples:** Provide practical code examples for each endpoint in Python (using the `requests` library) and a `cURL` command.
4. **Accuracy:** Document only what the specification states. Where details are missing (request bodies, response fields, error codes), insert a clearly marked `TODO:` instead of inventing them.
"""
    return ask(prompt)

# Define a simple specification for our IoT API
iot_api_specification = {
    "name": "IoT Device Management API",
    "version": "v1.2",
    "baseURL": "https://api.myiot.com/v1",
    "authentication": "API Key in 'X-API-Key' header",
    "endpoints": [
        {
            "method": "GET",
            "path": "/devices/{device_id}",
            "description": "Retrieves the status and metadata for a specific device."
        },
        {
            "method": "POST",
            "path": "/devices/{device_id}/commands",
            "description": "Sends a command (e.g., 'reboot', 'update_firmware') to a device."
        }
    ]
}

api_docs = generate_api_documentation(iot_api_specification)
print("--- Generated API Documentation ---")
print(api_docs)
```

By providing the specification and clear instructions on the required sections, we get a well-structured document, not just a simple description. Note the accuracy requirement: this spec says nothing about response fields or error formats, and without that instruction a model will happily invent plausible ones. For real APIs, generate documentation from a complete machine-readable spec (such as an OpenAPI file) and have an engineer review it.

## Domain 2: Code Generation and Review

As we've seen, AI can write code. But to write *good* code, it needs to understand the patterns and constraints of the specific domain (e.g., web development, data processing, embedded systems).

### Generating Code with Domain-Specific Patterns

Let's generate a Python class for an MQTT client, a common component in IoT systems, and enforce IoT-specific best practices.

```python
def generate_iot_code(requirements: dict) -> str:
    """Generates code that follows specific IoT domain patterns."""
    
    # Define best practices for this domain
    iot_patterns = [
        "Implement automatic reconnection logic with exponential backoff.",
        "Handle potential network errors and connection losses gracefully.",
        "Use comprehensive logging for easy debugging in the field.",
        "Ensure all network operations are asynchronous (`async`/`await`).",
        "Include methods for clean connection teardown."
    ]
    
    patterns = "\n".join(f"- {p}" for p in iot_patterns)
    prompt = f"""
Generate a Python class for an MQTT client based on these requirements.

**Requirements:**
{json.dumps(requirements, indent=2)}

**Mandatory Implementation Patterns:**
{patterns}

Use the `aiomqtt` library and connect over TLS (port 8883). Read credentials from environment variables.
"""
    response = client.responses.create(
        model="gpt-6-sol",   # a stronger model is worth it for code
        instructions="You are an expert embedded systems developer specializing in IoT communication protocols.",
        input=prompt,
    )
    return response.output_text

mqtt_client_requirements = {
    "purpose": "Collects sensor data from topics and forwards it to a cloud API.",
    "broker_address": "mqtt.industrial-iot.com",
    "topics_to_subscribe": ["factory/+/temperature", "factory/+/pressure"],
    "authentication": "Username/Password"
}

generated_code = generate_iot_code(mqtt_client_requirements)
print("--- Generated IoT MQTT Client Code ---")
print(generated_code)
```

By explicitly listing the required engineering patterns, you guide the AI to produce code that is not just functional but also robust and maintainable in a real-world IoT environment.

### Performing a Domain-Specific Code Review

We can also use AI to review code, asking it to look for problems specific to a domain.

```python
def review_iot_code(code_snippet: str) -> str:
    """Performs a code review with a focus on IoT-specific issues."""
    
    iot_review_criteria = [
        "Correct handling of device disconnections and reconnections.",
        "Efficient use of network bandwidth and power, especially for battery-powered devices.",
        "Security of the communication channel (e.g., use of TLS).",
        "Robust error handling for intermittent connectivity.",
        "Scalability to handle hundreds or thousands of devices."
    ]
    
    criteria = "\n".join(f"- {c}" for c in iot_review_criteria)
    prompt = f"""
Analyze the following Python code for an IoT application.

<code>
{code_snippet}
</code>

**Review Focus:**
Evaluate the code against these IoT-specific best practices:
{criteria}

Provide your review in a structured format with sections for 'Strengths', 'Critical Issues', and 'Suggestions for Improvement'.
For each critical issue, quote the relevant line.
"""
    return ask(prompt, instructions="You are a senior IoT architect performing a code review.")

# A simple (and flawed) code snippet to be reviewed
sample_code = """
import paho.mqtt.client as mqtt
import time

def on_message(client, userdata, msg):
    print(f"Received data: {msg.payload.decode()}")

client = mqtt.Client()
client.on_message = on_message
client.connect("broker.hivemq.com", 1883, 60)
client.subscribe("sensors/temperature")

while True:
    client.loop()
    time.sleep(1)
"""

code_review = review_iot_code(sample_code)
print("\n--- Domain-Specific Code Review ---")
print(code_review)
```

## Domain 3: Data Analysis and Insights

When analyzing data, providing domain context transforms the AI from a simple data processor into a insightful analyst.

### Industrial Manufacturing Analytics

Let's analyze production data from the perspective of a quality control engineer. Following the lesson of Chapter 4, we **compute the metrics in code** and ask the model to interpret them, rather than asking it to do the arithmetic.

OEE (Overall Equipment Effectiveness) is the product of three ratios:

-   **Availability** = run time ÷ planned production time
-   **Performance** = (ideal cycle time × total units) ÷ run time
-   **Quality** = good units ÷ total units

```python
def compute_oee(data: dict) -> dict:
    """Standard OEE calculation. Deterministic, testable, and exact."""
    planned_minutes = data["shift_duration_hours"] * 60 - data["planned_downtime_minutes"]
    run_minutes = planned_minutes - data["unplanned_downtime_minutes"]
    availability = run_minutes / planned_minutes
    performance = (data["ideal_cycle_time_seconds"] * data["total_units_produced"]) / (run_minutes * 60)
    quality = (data["total_units_produced"] - data["defective_units"]) / data["total_units_produced"]
    return {
        "availability": round(availability, 3),
        "performance": round(performance, 3),
        "quality": round(quality, 3),
        "oee": round(availability * performance * quality, 3),
    }

def analyze_manufacturing_data(data: dict) -> str:
    """Interprets precomputed production metrics with a quality-engineering persona."""
    metrics = compute_oee(data)
    prompt = f"""
Analyze this shift's production report. The OEE figures were computed exactly; do not recalculate them.

<raw_data>
{json.dumps(data, indent=2)}
</raw_data>
<computed_metrics>
{json.dumps(metrics, indent=2)}
</computed_metrics>

Provide:
- **OEE interpretation:** Which of availability, performance, and quality is the main loss, compared with a typical world-class benchmark of about 85% OEE?
- **Likely causes:** Plausible causes of the main loss, labelled as hypotheses.
- **Actionable steps:** Specific, practical steps to improve quality and throughput.
- **Data needed:** What additional data (for example, defect counts over many shifts) would be needed to judge whether the process is in statistical control. Do not draw that conclusion from a single shift.
"""
    return ask(prompt, instructions="You are a Quality Control Engineer in a manufacturing plant.")

production_data = {
    "shift_duration_hours": 8,
    "planned_downtime_minutes": 30,
    "unplanned_downtime_minutes": 45,
    "ideal_cycle_time_seconds": 60,
    "total_units_produced": 400,
    "defective_units": 20,
}

print(compute_oee(production_data))
# {'availability': 0.9, 'performance': 0.988, 'quality': 0.95, 'oee': 0.844}
print("--- Manufacturing Data Analysis ---")
print(analyze_manufacturing_data(production_data))
```

The OEE for this shift is 84.4% (90% availability × 98.8% performance × 95% quality), and the model's job is to explain what that means and what to do about it, which is the part that benefits from domain knowledge. Notice also the last bullet: whether a process is "in statistical control" can only be judged from many samples over time, and asking a model to decide it from one shift summary would invite a confident fabrication.

## A Practical Project: Domain-Specific IoT Troubleshooting

Let's build a troubleshooting assistant that leverages deep domain knowledge about different types of IoT devices to provide expert-level guidance.

```python
class IoTTroubleshootingSystem:
    def __init__(self):
        self.client = openai.OpenAI()
        # This knowledge base could be loaded from a database or configuration file
        self.device_knowledge = {
            "pressure_transmitter": {
                "common_issues": ["Clogged impulse lines", "Diaphragm failure", "Calibration drift due to temperature changes"],
                "diagnostic_tools": ["Pressure calibrator", "Manometer", "Multimeter"],
                "safety_note": "Always isolate the device from process pressure before disconnecting."
            },
            "vibration_sensor": {
                "common_issues": ["Improper mounting", "Electromagnetic interference from motors", "Sensor overload"],
                "diagnostic_tools": ["Vibration analyzer", "Oscilloscope", "Torque wrench"],
                "safety_note": "Be aware of rotating machinery when approaching the sensor."
            }
        }

    def generate_troubleshooting_guide(self, device_type: str, issue_description: str) -> str:
        """Generates a detailed, domain-specific troubleshooting guide."""
        
        # Retrieve domain knowledge for the specific device type
        knowledge = self.device_knowledge.get(device_type, {})
        
        prompt = f"""
You are an expert field service technician creating a troubleshooting guide.

**Device Type:** {device_type}
**Reported Issue:** {issue_description}

**Your Knowledge Base for this Device Type:**
- Common Issues: {knowledge.get('common_issues', 'N/A')}
- Required Tools: {knowledge.get('diagnostic_tools', 'N/A')}
- Critical Safety Note: {knowledge.get('safety_note', 'Follow standard safety procedures.')}

Create a systematic, step-by-step troubleshooting guide. Structure it with the following sections:
1. **Initial Assessment & Safety:** What to check first and what safety precautions to take.
2. **Hypothesis 1 (Most Likely Cause):** State the most probable cause based on the issue and your knowledge.
3. **Verification Steps for Hypothesis 1:** List the specific actions to confirm or deny this cause.
4. **Hypothesis 2 (Next Likely Cause):** State the second most probable cause.
5. **Verification Steps for Hypothesis 2:** List the actions to test this hypothesis.
6. **Escalation Path:** When to stop and call for a specialist engineer.
"""
        
        return self.client.responses.create(model=MODEL, input=prompt).output_text

# --- Demo of the troubleshooting system ---
troubleshooter = IoTTroubleshootingSystem()
reported_problem = "The pressure transmitter is showing erratic readings that jump up and down, but the system pressure seems stable."

guide = troubleshooter.generate_troubleshooting_guide("pressure_transmitter", reported_problem)
print("--- Generated Troubleshooting Guide ---")
print(guide)
```

This final example shows how embedding a structured knowledge base directly into the prompt allows the AI to generate highly relevant, expert-level content that is far more useful than a generic response.

## Conclusion

Domain-specific prompting elevates you from a user of AI to an architect of AI-powered expertise. By embedding the context, vocabulary, and workflows of a specific field into your prompts, you can create applications that deliver specialized and significant value.

Remember these key best practices:
- **Give the AI a Persona:** Put the role in the system prompt, and spell out the priorities that role implies.
- **Provide Domain Context:** Include industry-specific rules, constraints, and knowledge; this matters more than the persona.
- **Compute, Then Interpret:** Calculate domain metrics in code and let the model explain them.
- **Use Domain Terminology:** Speak the language of the expert you want the AI to emulate.
- **Structure for the Domain:** Ask for outputs that match the formats used in that field (e.g., API docs, engineering reports, financial analyses).

By applying these techniques, you can transform a general-purpose language model into a team of specialized virtual experts, ready to tackle complex challenges across any industry.

# References and Further Reading

- LLM Prompt Library (battle-tested prompt templates). https://abilzerian.github.io/LLM-Prompt-Library/
- AwesomeGPTSystemPrompts (GitHub collection). https://github.com/AntreasAntoniou/AwesomeGPTSystemPrompts
- Prompt Engineering Guide (Prompting Guide). https://www.promptingguide.ai/
- Prompt Creator: Prompt Engineering for AI (Prompt Creator). https://promptcreator.ai/