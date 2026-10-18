# Chapter 9: Fundamental Prompt Engineering

In previous chapters, we learned how to make API calls to AI models. Now, we will learn the art and science of **prompt engineering**—the skill of crafting inputs that guide an AI to produce the exact output you desire. A well-engineered prompt is the difference between a generic, unhelpful response and a precise, actionable, and reliable one.

Think of an LLM as an incredibly talented but unguided intern. A vague request like "write about our product" will yield a generic essay. A specific prompt like "Write a three-paragraph marketing description for our new IoT temperature sensor, targeting industrial facility managers, focusing on its five-year battery life and durability, and end with a call to action to request a demo" will yield a powerful piece of marketing copy. This chapter is about learning how to give those perfect instructions.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Write prompts that consistently produce high-quality, structured results.
-   Master the core prompting patterns: zero-shot, few-shot, and chain-of-thought.
-   Use role-based prompting to get expert-level analysis for different domains.
-   Know when explicit chain-of-thought still helps with today's reasoning models.
-   Compare prompt versions with a small, scored evaluation instead of eyeballing outputs.
-   Build a smart IoT device diagnostic system using advanced prompting techniques.

## From Vague to Valuable: The Anatomy of a Great Prompt

The most common mistake in prompt engineering is vagueness. Let's see the difference.

### The Bad Prompt

A vague prompt lacks detail and context, forcing the AI to guess what you want.

```python
import openai

client = openai.OpenAI()
MODEL = "gpt-6-luna"

# A bad prompt: vague, unclear, and without direction.
bad_prompt = "Tell me about our IoT device."

response = client.responses.create(model=MODEL, input=bad_prompt)

# The result will be a generic, unhelpful summary.
print(response.output_text)
```

### The Good Prompt

An effective prompt is specific, providing the AI with all the necessary ingredients to produce a high-quality response.

```python
# A good prompt: specific, contextual, and structured.
good_prompt = """
Analyze the following IoT device log and classify its status.

**Device Log:**
"2024-07-10 14:32:15 - TEMP-007 - Battery: 12% - Signal: Weak - Status: Active"

**Instructions:**
1.  **Task:** Classify the device's overall health as 'Healthy', 'Warning', or 'Critical'.
2.  **Format:** Respond with a JSON object containing 'device_id', 'health_status', and 'reason'.
3.  **Context:** A battery level below 20% is a 'Warning'. A 'Weak' signal is also a 'Warning'.
"""

response = client.responses.create(
    model=MODEL,
    input=good_prompt,
    text={"format": {"type": "json_object"}},   # JSON mode: the output must be valid JSON
)

# The result will be a precise, structured, and actionable JSON object.
print(response.output_text)
```

(JSON mode guarantees valid JSON but not *which* keys you get; Chapter 11 shows how to enforce an exact schema.)

### The Five Elements of an Effective Prompt

Every great prompt should contain some or all of these five elements:

1.  **Task:** A clear and specific verb describing what you want the AI to do (e.g., "Summarize," "Translate," "Classify," "Generate code").
2.  **Context:** Background information the AI needs to complete the task successfully (e.g., the user's role, the source of the data, business rules).
3.  **Format:** The desired structure of the output (e.g., "JSON," "a bulleted list," "a three-paragraph summary").
4.  **Examples (Few-Shot):** One or more examples of the desired input/output format. This is one of the most powerful ways to guide the AI.
5.  **Role (Persona):** A persona for the AI to adopt (e.g., "You are an expert cybersecurity analyst," "You are a friendly customer service agent").

Two structural habits make all of these work better:

-   **Separate instructions from data with delimiters.** Wrap documents, logs, and user-supplied text in clear markers such as XML-style tags (`<log>…</log>`) or fenced blocks. The model can then tell what to *do* from what to *analyze*, which also makes prompt injection harder (Chapter 22).
-   **Explain the purpose, not just the rule.** "Keep it under 50 words because it is shown on a small device screen" works better than "Be brief!!!". Modern models generalize from reasons and tend to over-apply rules written in capitals.

## Pattern 1: Zero-Shot Prompting (Just Ask)

**Zero-shot prompting** is the simplest form: you ask the AI to perform a task without giving it any prior examples. This works surprisingly well for general knowledge and common tasks that the model has seen many times in its training data.

```python
def zero_shot_analysis(log_entry: str) -> str:
    """Analyzes a log entry using a simple zero-shot prompt."""
    
    prompt = f"""
Analyze this IoT log entry and provide a one-sentence summary of the key issue.

<log_entry>
{log_entry}
</log_entry>
"""
    
    response = client.responses.create(model=MODEL, input=prompt)
    return response.output_text.strip()

log = "2024-07-10 15:00:05 - PUMP-03 - Pressure reading of 250 PSI exceeds maximum threshold of 200 PSI."
print(zero_shot_analysis(log))
```

**When to use it:** For simple, well-defined tasks where the AI likely has extensive training data (e.g., summarization, simple classification, general questions).

## Pattern 2: Few-Shot Prompting (Show, Don't Just Tell)

**Few-shot prompting** is one of the most effective techniques for improving the accuracy and consistency of AI responses. Instead of just describing what you want, you provide a few examples of inputs and their corresponding desired outputs. This teaches the AI the exact pattern and format you expect.

```python
def few_shot_classifier(alert_text: str) -> str:
    """Classifies IoT alerts into predefined categories using examples."""
    
    prompt = f"""
Classify the following IoT alert into one of these categories: [Connectivity, Power, Sensor_Failure, Security].

---
Example 1:
Alert: "Device TEMP-01 failed to check in."
Category: Connectivity

Example 2:
Alert: "Device DOOR-05 battery level at 4%."
Category: Power

Example 3:
Alert: "Device CAM-02 reported anomalous reading outside of normal range."
Category: Sensor_Failure

Example 4:
Alert: "Multiple failed login attempts detected for GATEWAY-01."
Category: Security
---

Now, classify this alert. Reply with only the category name.
Alert: "{alert_text}"
Category:
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        reasoning={"effort": "low"},  # a simple task doesn't need deep reasoning
    )
    return response.output_text.strip()

print(f"Classification: {few_shot_classifier('Sensor HUMID-07 reporting negative humidity values.')}")
print(f"Classification: {few_shot_classifier('Device LIGHT-10 lost connection to WiFi network.')}")
```

**When to use it:** When you need a specific output format, when the task is novel, or when you need to improve the consistency and reliability of the output. Choose examples that are varied and cover the tricky cases (models copy the *pattern* of your examples closely, including accidental quirks), and validate the output in code: if the model returns something outside your category list, treat it as an error rather than passing it on.

## Pattern 3: Chain-of-Thought (CoT) Prompting

LLMs often make mistakes in complex reasoning when they jump straight to an answer. **Chain-of-thought prompting** asks the model to work through intermediate steps before concluding. With the non-reasoning models of 2022–2024, simply adding "Let's think step by step" measurably improved performance on logic, math, and multi-step problems.

> **How this changed with reasoning models.** Today's reasoning models (GPT-6, current Claude and Gemini models) already perform an internal chain of thought before answering; you control how much with the reasoning-effort setting. For them, "think step by step" adds little, and providers advise against micromanaging the model's internal reasoning. What still helps is the part of CoT that is really **task specification**: telling the model *which* steps your process requires (as in the example below) and asking for the conclusions of each step in the visible answer so a human can check them. Explicit CoT remains useful with small, fast, or non-reasoning models.

```python
def chain_of_thought_diagnosis(symptoms: str) -> str:
    """Diagnoses a complex IoT issue using chain-of-thought reasoning."""
    
    prompt = f"""
An IoT device is exhibiting the following symptoms:
{symptoms}

Diagnose the root cause using these steps, and show your conclusion for each step:
1. First, analyze each symptom individually.
2. Then, look for connections or correlations between the symptoms.
3. Based on the connections, formulate the most likely hypotheses for the root cause.
4. Finally, suggest a diagnostic action to confirm the most likely hypothesis.
"""
    
    response = client.responses.create(model=MODEL, input=prompt, reasoning={"effort": "medium"})
    return response.output_text

symptom_list = """
- Temperature readings are fluctuating wildly.
- Battery level is draining much faster than usual.
- The device disconnects from the network intermittently.
"""

print(chain_of_thought_diagnosis(symptom_list))
```
Spelling out the steps makes the model follow *your* diagnostic process, and asking for the conclusion of each step makes the answer easier for a human to check and debug. (A caution: a model's written explanation is not a faithful trace of how it actually reached the answer, so treat it as a useful summary, not as proof.)

## Pattern 4: Role-Based Prompting (The Persona Pattern)

Instructing the AI to adopt a specific **role** or **persona** is an effective way to tailor its response's *focus, vocabulary, and tone*. Asking for "a cybersecurity analyst's view" steers the model toward threats and attack surfaces; asking for "a logistics manager's view" steers it toward schedules and costs. Be realistic about what a persona does: research has found that labels like "You are a world-class expert" do little for factual accuracy on their own. The value comes from the *perspective and priorities* the role implies, so spell those out.

```python
def get_expert_analysis(problem_description: str, expert_role: str) -> str:
    """Gets an analysis of a problem from a specific expert's perspective."""
    
    response = client.responses.create(
        model=MODEL,
        instructions=f"You are a {expert_role}. Analyze situations strictly from that professional viewpoint.",
        input=f"""
Situation: {problem_description}

Provide your analysis, focusing on the key issues, risks, and recommended actions relevant to your role.
""",
    )
    return response.output_text

# Get analyses from multiple different experts on the same problem
roles_to_consult = [
    "Hardware Engineer specializing in GPS modules",
    "Network Engineer focused on data transmission",
    "Logistics and Operations Manager",
    "Cybersecurity Analyst concerned with signal spoofing"
]

situation = "A fleet of 500 IoT delivery drones is experiencing a 15% failure rate in their GPS modules."
for role in roles_to_consult:
    print(f"--- Analysis from a {role} ---")
    print(get_expert_analysis(situation, role))
    print("\n")
```

This technique is incredibly powerful for tackling a problem from multiple angles, revealing insights that a single, generic prompt might miss.

## Comparing Prompts with a Small Evaluation

Your first prompt is rarely your best one. Production-grade AI applications require testing and iteration, and judging a prompt by reading one or two outputs is how regressions slip through. Instead, compare prompt variants on a **set of labelled test cases** and score each output automatically. This is called an *evaluation* (or "eval"), and Chapter 24 develops it into a full testing practice.

```python
import json

# Each case: an input and the fields we expect the model to extract.
test_cases = [
    {"input": "Device PUMP-01 reports pressure at 300 PSI. The safe limit is 250 PSI.",
     "expected": {"device_id": "PUMP-01", "metric": "pressure", "value": 300}},
    {"input": "TEMP-07 at 92C, threshold 90C.",
     "expected": {"device_id": "TEMP-07", "metric": "temperature", "value": 92}},
    {"input": "Humidity sensor HUM-3 reading 18%; minimum allowed 20%.",
     "expected": {"device_id": "HUM-3", "metric": "humidity", "value": 18}},
]

variants = {
    "v1_minimal": "Summarize this alert in JSON: {input}",
    "v2_keys": "Extract device_id, metric (lowercase), and value (a number) from this alert. "
               "Respond with a JSON object with exactly those keys.\n\n<alert>{input}</alert>",
}

def score(output: str, expected: dict) -> bool:
    """A case passes only if the JSON parses and every expected field matches."""
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return False
    return all(data.get(key) == value for key, value in expected.items())

for name, template in variants.items():
    passed = 0
    for case in test_cases:
        output = client.responses.create(
            model=MODEL,
            input=template.format(input=case["input"]),
            text={"format": {"type": "json_object"}},
        ).output_text
        passed += score(output, case["expected"])
    print(f"{name}: {passed}/{len(test_cases)} cases passed")
```

The minimal variant usually produces valid JSON with unpredictable key names, so it fails the checks; the explicit variant passes. Three cases are only a start: a useful eval set has dozens to hundreds of cases, drawn from real inputs and including the hard edge cases. Because model outputs vary between runs, compare variants on the *pass rate over the whole set*, not on any single output.

## Putting It All Together: A Smart IoT Diagnostic System

Let's build a complete diagnostic system that uses all the patterns we've learned to provide a comprehensive analysis of an IoT device alert.

```python
import json
from datetime import datetime

CATEGORIES = ["Power", "Connectivity", "Sensor_Failure", "Software_Bug"]

class IoTDiagnosticSystem:
    def __init__(self):
        self.client = openai.OpenAI()

    def _ask(self, prompt: str, instructions: str | None = None, effort: str = "medium") -> str:
        response = self.client.responses.create(
            model=MODEL, instructions=instructions, input=prompt, reasoning={"effort": effort}
        )
        return response.output_text.strip()

    def generate_report(self, alert_log: str, device_context: dict) -> dict:
        """Generates a complete diagnostic report for an IoT alert."""
        
        # 1. Classify the alert using a few-shot prompt, then validate the label in code.
        classification_prompt = f"""
Classify this alert log into one of {CATEGORIES}. Reply with only the category.
Example 1: "Battery at 3%" -> Power
Example 2: "Failed to connect to WiFi" -> Connectivity
Log: "{alert_log}"
Category:
"""
        category = self._ask(classification_prompt, effort="low")
        if category not in CATEGORIES:
            category = "Unclassified"

        # 2. Get a detailed diagnosis by specifying the reasoning steps.
        diagnosis_prompt = f"""
An IoT device reported the following alert: "{alert_log}".
Device context: {json.dumps(device_context)}

Diagnose it in three steps, giving your conclusion for each:
1.  What is the immediate problem indicated by the alert?
2.  Considering the device context, what are the 2-3 most likely root causes?
3.  What is the recommended first action to take to verify the cause?
"""
        diagnosis = self._ask(diagnosis_prompt)

        # 3. Get a specialized perspective using role-based prompting.
        security_analysis = self._ask(
            f'Alert: "{alert_log}"\nContext: {json.dumps(device_context)}',
            instructions="You are a cybersecurity analyst. Assess IoT alerts only for security implications "
                         "(compromise, spoofing, denial of service) and say plainly if there are none.",
        )

        # 4. Assemble the final report
        report = {
            "alert": alert_log,
            "category": category,
            "diagnosis_reasoning": diagnosis,
            "security_assessment": security_analysis,
            "report_generated_at": datetime.now().isoformat()
        }
        return report

# --- Demo of the complete system ---
system = IoTDiagnosticSystem()

test_alert = "Device MOTION-07 is spamming the server with thousands of 'motion detected' events per second."
test_context = {
    "device_type": "PIR Motion Sensor",
    "firmware_version": "v1.2.1",
    "last_firmware_update": "2024-06-01",
    "location": "Main entrance"
}

full_report = system.generate_report(test_alert, test_context)

print(json.dumps(full_report, indent=2))
```

This final example demonstrates how these fundamental prompting patterns are not used in isolation but are combined to build a sophisticated, multi-faceted analysis workflow.

## Conclusion

Prompt engineering is the new essential skill for developers in the age of AI. It is a craft that blends clear instructions, contextual awareness, and structured thinking. By mastering the patterns in this chapter—zero-shot, few-shot, chain-of-thought, and role-based prompting—you have gained the ability to command AI models with precision and predictability.

Remember the core principle: **garbage in, garbage out.** A vague prompt yields a vague response. A specific, well-structured, context-rich prompt is the key to getting the most out of any LLM. Treat prompts like code: version them, test them against a set of cases, and measure before you change them. You are now equipped to build more reliable, accurate, and powerful AI features.

## Key Takeaways

-   A good prompt states the task, context, and output format, and adds examples and a role where they help. Wrap logs and user text in delimiters such as `<log>...</log>` so the model can tell instructions from data.
-   Explain why a rule exists ("under 50 words because it shows on a device screen") instead of shouting it in capitals; modern models generalize from reasons.
-   Few-shot examples teach the exact pattern and label set you want, quirks included, so pick varied examples that cover the tricky cases and reject any label outside your list in code.
-   Reasoning models already think internally, so "think step by step" adds little. Spell out the steps of *your* process and ask for each step's conclusion in the visible answer, and do not treat that explanation as a faithful trace of how the model decided.
-   A persona changes focus, vocabulary, and tone, not factual accuracy. Put it in `instructions` and spell out the priorities the role implies.
-   Compare prompt variants with a small eval: labelled cases, an automatic `score()` function, and the pass rate over the whole set rather than one eyeballed output.

## Exercises

**1. Warm-up:** Take the `few_shot_classifier` and write 10 labelled alerts of your own (at least two per category, including one ambiguous case). Run each alert through the classifier and print the accuracy and the list of misclassified alerts. You are done when you can report a number such as "9/10 correct" and name the alert it got wrong.

**2. Build:** Extend the "Comparing Prompts" eval to 12 test cases, including at least three awkward inputs (a unit like "212F", a value written in words, a device ID in lowercase). Add a third variant, `v3_fewshot`, that includes two worked examples, and run every variant three times. You are done when the script prints the pass rate of each variant averaged over the three runs, and you can say whether `v3_fewshot` beats `v2_keys` by more than the run-to-run variation.

**3. Stretch:** Harden `IoTDiagnosticSystem.generate_report` so that a failure in one step does not lose the whole report. Wrap each model call so a raised API error (or an empty reply) records an `"error"` entry for that field while the other steps still run, and log how often the classifier returned `Unclassified`. You are done when a test that makes the security call raise an exception (for example by patching `_ask`) still gets back a report with a valid `category` and `diagnosis_reasoning`.

# References and Further Reading

- Prompt Engineering: The Definitive Step-By-Step How to Guide (Plain English). https://ai.plainenglish.io/prompt-engineering-the-definitive-step-by-step-how-to-guide-fb7c5eea1900
- Advanced Prompt Engineering for Content Creators – Full Handbook (freeCodeCamp). https://www.freecodecamp.org/news/advanced-prompt-engineering-handbook
- Best practices for LLM prompt engineering (Palantir Docs). https://www.palantir.com/docs/foundry/aip/best-practices-prompt-engineering
- Mastering Prompt Engineering for Large Language Models (Medium). https://medium.com/@nasdag/mastering-prompt-engineering-for-large-language-models-1afafb52c44b
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022). https://arxiv.org/abs/2201.11903
- Prompt engineering overview (Claude documentation). https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview
- Prompt engineering guide (OpenAI documentation). https://platform.openai.com/docs/guides/prompt-engineering