# Chapter 4: Understanding AI Capabilities and Limitations

Before we build more complex applications, we must take a crucial step back and develop a deep, practical understanding of what AI can—and, more importantly, *cannot*—do reliably. The most common mistake new AI developers make is treating Large Language Models (LLMs) as infallible, all-knowing oracles. They are not.

LLMs are incredibly powerful pattern-matching engines, but they are also prone to making mistakes, fabricating information, and misunderstanding context in ways that can be both subtle and significant. This chapter is your guide to becoming a discerning AI developer. We will learn to appreciate the magic while also recognizing the sleight of hand, equipping you to build applications that are not just powerful, but also safe and reliable.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Identify the core tasks where modern AI models excel.
-   Recognize the common failure modes of LLMs, especially **hallucinations**.
-   Develop a critical mindset for when to trust AI output and when to be skeptical.
-   Implement practical safeguards and best practices in your applications to mitigate AI's weaknesses.
-   Build a simple but responsible AI assistant that embodies these safety principles.

## Where AI Shines: Core Strengths

Modern LLMs are remarkably proficient at a range of tasks centered around the statistical patterns of language.

### 1. Understanding and Generating Text

This is AI's home turf. It can summarize dense articles, translate languages, rephrase text in different styles, and extract structured information from unstructured prose.

```python
import openai
import os
from dotenv import load_dotenv

# Assumes your OPENAI_API_KEY is set in a .env file
load_dotenv()
client = openai.OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")

def ask(prompt: str, instructions: str | None = None) -> str:
    """A small helper we'll reuse throughout this chapter."""
    response = client.responses.create(model=MODEL, instructions=instructions, input=prompt)
    return response.output_text

def demonstrate_text_abilities():
    """Demonstrates AI's core text processing capabilities."""
    
    tasks = {
        "Summarization": "Summarize this paragraph in one sentence: The new IoT deployment includes 500 temperature sensors and 200 humidity sensors, all reporting back to a central server every 5 minutes. The goal is to monitor the server farm's environment to prevent overheating and maintain optimal hardware longevity.",
        "Translation": "Translate this to Spanish: The pressure sensor is reporting a critical failure.",
        "Information Extraction": "Extract the device ID, error type, and timestamp from this log entry: 2024-01-20T14:30:15Z - DEVICE_ID=TEMP-042 - ERROR=READ_TIMEOUT",
        "Code Fixing": "Fix the syntax error in this Python code: def my_func(x) print(x + 1)"
    }

    for task_name, prompt in tasks.items():
        print(f"--- {task_name} ---")
        print(f"AI Response: {ask(prompt)}\n")

demonstrate_text_abilities()
```
The AI handles these with ease because they are fundamentally pattern-based language tasks.

### 2. Recognizing Patterns in Data

LLMs can identify trends, anomalies, and categories in data sets, making them excellent for tasks like log analysis or sentiment classification.

```python
def demonstrate_pattern_recognition():
    """Shows AI's ability to spot anomalies in a series of log entries."""
    log_data = """
    10:00 - sensor_01 - temp: 72.1
    10:05 - sensor_01 - temp: 72.3
    10:10 - sensor_01 - temp: 95.8
    10:15 - sensor_01 - temp: 72.5
    10:20 - sensor_01 - temp: 72.4
    """
    
    analysis = ask(f"Analyze these logs and identify any anomalies:\n{log_data}")
    print("--- Pattern Recognition ---")
    print(f"AI Analysis: {analysis}")

demonstrate_pattern_recognition()
```
The AI identifies the temperature spike because it deviates sharply from the surrounding readings. (Note that we don't label the anomaly in the data; doing so would test reading, not detection.) For five readings this is impressive; for a million readings a statistical rule (for example, a z-score threshold) is faster, cheaper, and more reliable. A good pattern is to let code flag candidates and use the LLM to explain them.

### 3. Generating Code

One of the most powerful applications for developers is code generation. You can describe a function in plain English, and the AI will write the code for you.

```python
def demonstrate_code_generation():
    """Demonstrates AI writing a Python function from a description."""
    prompt = "Write a Python function called `is_safe_temperature` that takes a temperature in Celsius and returns True if it's between 0 and 100, and False otherwise."
    
    print("--- Code Generation ---")
    print(ask(prompt, instructions="You are a helpful coding assistant that writes clean Python code."))

demonstrate_code_generation()
```
> **Warning:** AI-generated code should always be treated as a draft. It requires careful review, testing, and security analysis by a human developer before being used in production.

## Where AI Stumbles: Inherent Limitations

Understanding an AI's weaknesses is more important than knowing its strengths. Building around these limitations is the hallmark of a great AI engineer.

### 1. Mathematical and Precise Logical Reasoning

LLMs are not calculators or formal logic engines. They produce answers by predicting tokens, not by executing arithmetic. Modern reasoning models are far better at math than their predecessors (a current flagship will almost always get `127 × 83` right), but "almost always" is not good enough for a billing system, a dosage calculation, or a control loop. Errors become more likely with long numbers, many steps, or large tables of figures, and when they happen they look just as confident as correct answers.

```python
def demonstrate_math_check():
    """Compares the model's arithmetic with Python's."""
    math_problem = "What is 48,213 multiplied by 7,919? Reply with only the number."
    
    print("--- Math Check ---")
    print(f"Problem: {math_problem}")
    print(f"AI's Answer: {ask(math_problem)}")
    print(f"Correct Answer: {48_213 * 7_919:,}")
```

**The Safeguard: AI for Language, Code for Math**

A robust pattern is to use the AI to understand the *intent* of the problem and translate it into an expression, which you then evaluate with ordinary, deterministic code. Never pass model output to `eval()`: a model can be tricked (Chapter 22) into producing `__import__('os').system(...)` just as easily as `127 * 83`. Instead, use an evaluator that only understands arithmetic:

```python
import ast
import operator

ALLOWED_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.USub: operator.neg,
}  # No ** on purpose: 9**9**9 would hang the process.

def evaluate_arithmetic(expression: str) -> float:
    """Evaluates +, -, *, / on numbers only. Anything else raises ValueError."""
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[type(node.op)](walk(node.operand))
        raise ValueError(f"Disallowed expression: {ast.dump(node)}")
    return walk(ast.parse(expression, mode="eval").body)

def safe_math_solver(problem: str) -> str:
    """Uses AI to translate a word problem into arithmetic, then computes it in code."""
    expression = ask(
        f"Convert this problem into a single arithmetic expression: '{problem}'",
        instructions="Return only the expression, using numbers and + - * / ( ). No words, no code fences.",
    ).strip()
    print(f"AI generated this expression: `{expression}`")

    try:
        return f"Calculated result: {evaluate_arithmetic(expression)}"
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"Could not evaluate the AI's expression: {e}"

print(safe_math_solver("What is 127 multiplied by 83?"))
```
This hybrid approach uses the strengths of both systems: the LLM's language understanding and Python's computational reliability. For anything more complex than arithmetic, the same idea scales up as **tool calling** (Chapter 14), where the model asks your code to run a calculator, a database query, or a script in a proper sandbox.

### 2. Access to Real-Time Information

LLMs are trained on a static dataset. Their knowledge has a "cutoff date," and a bare model call has no way to check today's weather, your database, or the current time.

```python
def demonstrate_timeliness_limit():
    """Shows AI's lack of real-time knowledge."""
    prompt = "What is the current temperature in New York City?"
    
    print("--- Timeliness Limitation ---")
    print(f"AI's Response: {ask(prompt)}")
    print("⚠️ A good model will state its limitation; a weaker one may invent an answer.")
```
To give a model current information, you must supply it: either fetch the data yourself and put it in the prompt, or give the model a **tool** it can call (Chapter 14). Most providers now offer built-in web search tools, so "the model can't access the internet" is really "the model can only access what you deliberately connect it to."

## The Specter of AI Hallucinations

The most dangerous failure mode of an LLM is the **hallucination**. This is when the model generates information that is plausible, confident, and completely false.

Hallucinations happen because the AI's goal is to predict the next most statistically likely token, not to state the truth. If a false statement is constructed from plausible word patterns, the AI may generate it.

```python
def demonstrate_hallucination():
    """Prompts the AI about a non-existent entity to elicit a hallucination."""
    
    # The "Raspberry Pi Model Z" does not exist.
    prompt = "What are the key features of the Raspberry Pi Model Z, released in 2023?"
    
    print("--- Hallucination Probe ---")
    print(f"Prompt (about a fictional product): {prompt}")
    print(f"AI's Response: {ask(prompt)}")
```
Current frontier models will often (correctly) say they know of no such product, especially for well-known brands. Hallucinations have not disappeared, though; they have moved to less obvious territory: specific numbers, citations and URLs, details of niche products, internal company facts, and anything the question *presupposes* is true. A weaker model, or a more obscure topic, will happily invent specifications that sound entirely plausible. That is why you must **never blindly trust specific, factual claims from an LLM without verification**, and why grounding answers in sources you supply (Chapter 27) is the main defence.

## Building Safeguards into Your Applications

As responsible developers, we must build systems that anticipate and mitigate these weaknesses.

### 1. Requesting Confidence Levels

A simple technique is to instruct the AI to state its own confidence level and explain it. Treat the number with caution: self-reported confidence is poorly calibrated (models are often confidently wrong), so it is useful as a prompt for the model to surface caveats and as a rough triage signal, not as a probability you can rely on.

```python
def get_response_with_confidence(prompt: str) -> str:
    """Instructs the AI to include a confidence note in its response."""
    instructions = (
        "After your main response, add a line 'Confidence: low|medium|high' and one sentence "
        "on what you are unsure about and what should be checked."
    )
    return ask(prompt, instructions=instructions)

print(get_response_with_confidence("What is the average power consumption of a LoRaWAN gateway?"))
```

### 2. Implementing a "Verification Layer"

For critical information, you can use a two-step AI process. First, get an answer. Second, ask another AI call to act as a "fact-checker" on the first response.

```python
import json

def self_verifying_ai_call(prompt: str):
    """A two-step process where one AI call checks the output of another."""
    
    # Step 1: Get the initial answer
    initial_response = ask(prompt)
    
    # Step 2: Ask a second call to identify claims that need verification
    verification_prompt = f"""
    Please review the following text and list any specific factual claims (like numbers, dates, names, or technical specs) that should be verified from an external source.

    <text_to_review>
    {initial_response}
    </text_to_review>
    """
    
    verification_response = ask(verification_prompt, instructions="You are a careful fact-checking assistant.")

    return {
        "original_answer": initial_response,
        "claims_to_verify": verification_response
    }

result = self_verifying_ai_call("What are the main differences between Zigbee and Z-Wave for home automation?")
print(json.dumps(result, indent=2))
```
This process helps flag claims for human review. Be clear about its limits: the second call uses the same knowledge as the first, so it is good at *finding* checkable claims but cannot *verify* them. Real verification needs an external source of truth, such as a search tool, your own database, or a document you supply.

### 3. Building a Safe Assistant Class

Let's encapsulate these principles into a `SmartSafeAssistant` class that automatically applies different safety protocols based on the type of question.

```python
import re

class SmartSafeAssistant:
    def classify_question(self, question: str) -> str:
        """A deliberately simple rule-based classifier for question types.

        Keyword rules are brittle; Chapter 11 shows how to classify with an LLM and a fixed label set.
        """
        q_lower = question.lower()
        if any(keyword in q_lower for keyword in ["calculate", "what is", "multiply", "plus"]) and any(char.isdigit() for char in q_lower):
            return "math"
        if any(keyword in q_lower for keyword in ["today", "now", "current"]):
            return "current_info"
        return "general"

    def ask(self, question: str) -> str:
        """Processes a question using the appropriate safe handler."""
        question_type = self.classify_question(question)
        print(f"-> Detected question type: {question_type}")

        if question_type == "math":
            return self._handle_math(question)
        elif question_type == "current_info":
            return self._handle_current_info()
        else:
            return self._handle_general(question)

    def _handle_math(self, question: str) -> str:
        # Implements the safe "translate to an expression, compute in code" pattern
        return safe_math_solver(question) # Assuming safe_math_solver is defined as above

    def _handle_current_info(self) -> str:
        return "I'm sorry, but I do not have access to live, real-time information. Please consult a dedicated service for current data."
        
    def _handle_general(self, question: str) -> str:
        # For general questions, we add a disclaimer.
        return ask(question) + "\n\n*Please note: For critical applications, always verify important information from an authoritative source.*"

# --- Demo of the SmartSafeAssistant ---
assistant = SmartSafeAssistant()

questions_to_test = [
    "What is 45 * 193?",
    "What's the weather like right now?",
    "Explain the concept of mesh networking in IoT."
]

for q in questions_to_test:
    print(f"\n--- Asking: '{q}' ---")
    response = assistant.ask(q)
    print(f"Safe Response: {response}")
```

This simple class demonstrates a powerful concept: building a layer of logic *around* the LLM to enforce safety and reliability, rather than just trusting the model's raw output.

## Conclusion: Developing an AI-Savvy Mindset

You now have a realistic and practical understanding of what AI can and cannot do. This knowledge is your most valuable asset as an AI developer.

-   **Trust AI for:** Understanding and transforming language: summarization, extraction, classification, drafting, code suggestions, and brainstorming.
-   **Be Skeptical of AI for:** Exact computation, anything that needs current or private data you haven't supplied, and specific, verifiable facts (numbers, names, citations).
-   **Your Strategy:** Use AI as a powerful reasoning engine and language tool, but couple it with reliable, deterministic systems (like your own code or trusted APIs) for tasks that require precision and accuracy.

By embracing this mindset, you'll avoid common pitfalls and build applications that are not only intelligent but also trustworthy. You are now prepared to dive into more advanced topics, armed with the critical judgment needed to build responsibly.

## Key Takeaways

-   LLMs are strong at pattern-based language work: summarizing, translating, extracting fields from logs, explaining anomalies, and drafting code. Treat generated code as a draft that needs review and tests, and for large datasets let a statistical rule flag candidates and the LLM explain them.
-   For exact computation, have the model translate the problem into an expression and compute it in code with a restricted evaluator like `evaluate_arithmetic`. Never pass model output to `eval()`; tool calling (Chapter 14) is the general form of this pattern.
-   A bare model call knows nothing after its training cutoff and nothing about your private data. Supply current information in the prompt or give the model a tool to fetch it.
-   Hallucinations now concentrate in specific numbers, citations and URLs, niche product details, and questions that presuppose something false. Verify specific factual claims against an external source; grounding answers in documents you supply (Chapter 27) is the main defence.
-   Self-reported confidence is poorly calibrated, and a second "fact-checker" call shares the first call's knowledge. Both help you find what to check, but neither verifies anything.
-   Put your own logic around the model: route each request by type, as `SmartSafeAssistant` does, so math goes to code and live-data questions get an honest answer instead of a guess.

## Exercises

**1. Warm-up:** Measure how arithmetic accuracy changes with problem size instead of trusting one example. Generate 20 random multiplication problems, from 2-digit by 2-digit up to 6-digit by 6-digit, send each with "Reply with only the number.", and compare the reply with Python's result. Count any reply that does not parse as a number as wrong. You are done when you have a small table of accuracy by digit count for your model.

**2. Build:** Write tests for `evaluate_arithmetic`: valid expressions such as `48213 * 7919` and `-3 + 4 / 2` return the right value, and inputs such as `__import__('os').system('ls')`, `2 ** 10`, `abs(-1)`, `x + 1` and `"a" * 3` raise `ValueError`. Decide what should happen for `True + 1` and make your test enforce it. Then run `safe_math_solver` on ten IoT word problems with known answers (for example, "120 sensors each report every 5 minutes; how many readings per day?"). You are done when all evaluator tests pass and you can report how many of the ten problems came back exactly right, with each failure classified as a wrong expression or a rejected expression.

**3. Stretch:** Build a small hallucination probe set of ten IoT questions: five about real, checkable facts (such as the LoRaWAN frequency band used in Europe) and five that presuppose a product or standard that does not exist. Run each question three times through `get_response_with_confidence`, and label every answer by hand as correct, admitted not knowing, or fabricated. You are done when you have a table of fabrication rates for both halves and have checked whether any fabricated answer was labelled "Confidence: high".

# References and Further Reading

- Understanding The Limitations Of AI (Artificial Intelligence). https://medium.com/@marklevisebook/understanding-the-limitations-of-ai-artificial-intelligence-a264c1e0b8ab
- AI’s limitations: What artificial intelligence can’t do. https://lumenalta.com/insights/ai-limitations-what-artificial-intelligence-can-t-do
- 14 Things AI Can — and Can't Do. GeeksforGeeks. https://www.geeksforgeeks.org/14-things-ai-can-and-cant-do/
- Limitations of AI: What’s Holding Artificial Intelligence Back in 2025? https://visionx.io/blog/limitations-of-ai/

<!-- chapter-nav -->

---

[← Chapter 3: Setting Up Your Development Environment](03-development-environment.md) | [Table of Contents](../README.md) | [Chapter 5: A Developer's Guide to the OpenAI API →](05-openai-api-complete-guide.md)
