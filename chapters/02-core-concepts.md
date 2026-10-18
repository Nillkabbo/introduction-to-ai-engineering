# Chapter 2: The Engine Room of AI: Tokens, Embeddings, and Context Windows

In the last chapter, we had our first conversation with an AI. It felt simple, almost magical. But as developers, we know that magic is just well-executed technology. To build powerful, reliable, and cost-effective AI applications, we need to look under the hood. We don't need to become mechanics, but we do need to understand the basic "physics" that govern how these models operate.

Think of an LLM as a high-performance engine. To drive it well, you need to understand three things:

1.  **The Fuel (Tokens):** How the engine consumes fuel and what it costs. For an LLM, the fuel is text, measured in units called **tokens**.
2.  **The Transmission (Embeddings):** How the engine translates your intent into motion. For an LLM, this is the process of turning words into meaningful numbers called **embeddings**.
3.  **The Gas Tank (Context Window):** How far you can go on a single trip before needing to refuel. For an LLM, this is its short-term memory, known as the **context window**.

Mastering these three concepts is the key to moving from simple prompts to sophisticated AI engineering.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Explain what tokens are and how they directly impact your API costs.
-   Use `tiktoken` to count tokens and estimate the cost of an AI call before you make it.
-   Grasp how embeddings allow an AI to understand the *meaning* of words, not just the words themselves.
-   Build a semantic search tool that finds related concepts using embeddings.
-   Understand the limitations of an LLM's "memory" and how to manage a conversation's context window.

## Tokens: The Currency of Language Models

The first and most important thing to understand is that LLMs don't see words or characters. They see the world in **tokens**. A token is a chunk of text, which could be a whole word, part of a word, a number, or a piece of punctuation.

Let's see this in action. The sentence "Hello world!" is broken down by the AI into three distinct tokens: `["Hello", " world", "!"]`.

This might seem like a minor detail, but it is the single most important factor affecting the cost and performance of your application.

### Why Tokens Matter: The Cost of Conversation

AI providers don't charge you per request or per word; they charge you **per token**. This applies to both the tokens you send in your prompt *and* the tokens the AI generates in its response.

A short prompt is cheap. A long one is expensive. Let's make this tangible. We can use a library from OpenAI called `tiktoken` to see exactly how an LLM "sees" our text.

First, install the library:
```bash
pip install tiktoken
```

Now, let's count the tokens in a few different strings.

```python
import tiktoken

# o200k_base is the tokenizer used by OpenAI's recent model families.
# (tiktoken.encoding_for_model() only knows model names that existed when your
# tiktoken version was released, so naming the encoding directly is more robust.)
encoder = tiktoken.get_encoding("o200k_base")

text_to_analyze = "This is a simple sentence."

# The .encode() method converts our string into a list of token integers.
tokens = encoder.encode(text_to_analyze)

print(f"Text: '{text_to_analyze}'")
print(f"Token count: {len(tokens)}")
print(f"Tokens: {[encoder.decode([t]) for t in tokens]}")
```

**Output:**

```
Text: 'This is a simple sentence.'
Token count: 6
Tokens: ['This', ' is', ' a', ' simple', ' sentence', '.']
```

The text was broken into six tokens. Notice that the spaces belong to the tokens that follow them. Different providers use different tokenizers, so the same text can be a slightly different number of tokens on Claude or Gemini.

> **Estimating vs. measuring.** A local tokenizer like `tiktoken` is great for quick estimates. For exact numbers, use what the provider reports: every API response includes a `usage` object with the billed input and output tokens, and providers offer counting endpoints you can call *before* sending a request (for example `client.responses.input_tokens.count(model=..., input=...)` on OpenAI and `client.messages.count_tokens(...)` on Anthropic).

Now, let's build a simple cost calculator to see the financial implications.

```python
# Example price: $0.10 per 1 million input tokens.
# Prices change often; always check the provider's pricing page.
PRICE_PER_MILLION_INPUT_TOKENS = 0.10

def calculate_prompt_cost(text: str):
    """Estimates the input cost of sending a prompt."""
    token_count = len(encoder.encode(text))
    cost = token_count / 1_000_000 * PRICE_PER_MILLION_INPUT_TOKENS
    
    print(f"Text: '{text}'")
    print(f"Tokens: {token_count}")
    print(f"Estimated Cost: ${cost:.8f}")
    print("-" * 20)

# Compare a wordy prompt with a concise one.
vague_prompt = "Please help me, my IoT device is not working as I expected and I believe there might be an issue with the sensor readings."
concise_prompt = "Help interpret an abnormal IoT sensor reading."

calculate_prompt_cost(vague_prompt)
calculate_prompt_cost(concise_prompt)
```

The wordy prompt is roughly three times as many tokens as the concise one. On a single call the difference is a fraction of a cent; across millions of calls, or when the extra text is a long document pasted into every request, it adds up.

> **Key Takeaway:** Tokens are the unit of cost and latency. Remove text that doesn't help the model, but never cut context it actually needs: a confused model that needs a retry costs more than a few extra tokens. In practice, the big savings come from controlling *large* inputs (conversation history, retrieved documents) and output length, not from shaving words off a one-line instruction. Chapter 21 covers this in depth.

## Embeddings: How AI Understands Meaning

Now that we know how AI *counts* text, let's explore how it *understands* it. The magic behind this is a concept called **embeddings**.

An embedding is a way of turning a token or a piece of text into a list of numbers—a vector—that captures its semantic meaning.

Think of it like a map. On a world map, we can represent any city with two numbers: latitude and longitude. Cities that are geographically close, like London and Paris, will have similar coordinates.

Embeddings do the same thing, but for meaning. They place words in a high-dimensional "meaning space."

-   The vector for "cat" will be very close to the vector for "kitten."
-   The vector for "happy" will be very close to the vector for "joyful."
-   The vector for "car" will be far away from the vector for "planet."

This allows the AI to understand relationships between words. In classic word-embedding models (like word2vec), even the *relationship* between vectors was meaningful: the vector from "man" to "woman" is very similar to the vector from "king" to "queen." Modern embedding models work on whole sentences and paragraphs, which is what makes them useful for search.

### Creating Your First Embedding

Let's get the embedding vector for a piece of text. The `text-embedding-3-small` model is a great, cost-effective choice for this.

```python
import openai

client = openai.OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

def get_embedding(text: str) -> list[float]:
    """Generates an embedding vector for a given piece of text."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

# Get the embedding for a simple word
cat_embedding = get_embedding("cat")

print(f"The word 'cat' is represented by a list of {len(cat_embedding)} numbers.")
print(f"Here are the first 5 numbers: {cat_embedding[:5]}")
```

The output is a list of 1536 numbers. We don't care about the specific values; we only care about comparing them to other vectors.

### A Practical Use Case: Semantic Search

The most powerful application of embeddings is **semantic search**—finding things based on meaning, not just keywords. Let's build a tool that can find the most relevant troubleshooting step for a user's problem.

```python
import numpy as np

def find_most_similar(query: str, options: list[str]) -> str:
    """Finds the option most semantically similar to the query."""
    
    # 1. Get embeddings for the query and all options in a single API call
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query] + options
    )
    
    # 2. Separate the query embedding from the option embeddings
    query_embedding = response.data[0].embedding
    option_embeddings = [item.embedding for item in response.data[1:]]
    
    # 3. Calculate the similarity between the query and each option.
    # Cosine similarity is the dot product of normalized vectors. OpenAI's
    # embeddings are already normalized to length 1, so a plain dot product works.
    # For other providers, normalize first (divide each vector by its norm).
    similarities = [np.dot(query_embedding, opt_emb) for opt_emb in option_embeddings]
    
    # 4. Find the index of the highest similarity score
    best_option_index = np.argmax(similarities)
    
    return options[best_option_index]

# Our knowledge base of potential solutions
possible_solutions = [
    "Check for direct sunlight hitting the sensor.",
    "Restart the device's WiFi router.",
    "Recalibrate the sensor against a known temperature source.",
    "Replace the device's battery.",
]

# A user's problem, described in their own words
user_problem = "The thermostat reading is way too high, it says 95 degrees in here!"

# Find the best solution
best_solution = find_most_similar(user_problem, possible_solutions)

print(f"User's Problem: '{user_problem}'")
print(f"Most Relevant Solution: '{best_solution}'")
```

Notice that the user's problem shares almost no keywords with any of the solutions, so a keyword search would find nothing. An embedding search compares meaning instead: "the reading is way too high" is close to ideas like an external heat source or a miscalibrated sensor, so one of those two solutions should rank first. Which one wins depends on the embedding model, and that's the honest lesson: semantic search *ranks* candidates by similarity; it doesn't *reason* about which fix is correct. In real systems you retrieve the top few candidates and let a model or a person decide (Chapter 27).

## Context Windows: The AI's Limited Memory

The final core concept is the **context window**. This is arguably the most important limitation to understand when building applications.

An LLM's context window is its **short-term memory**. It's the maximum number of tokens the model can consider at one time when generating a response. Everything—the system prompt, the user prompts, and the assistant's own past responses—must fit within this window.

The model itself has no memory between calls. Each API request is stateless, and "memory" is simply the history *you* send. If a request exceeds the context window, the API rejects it with an error; nothing is silently forgotten. Deciding what to keep, drop, or summarize is your application's job.

### Context Window Sizes

The size of the context window is a key feature that differentiates models. As of October 2026, roughly one million tokens (on the order of 2,000 pages of text) is common across the flagship models from OpenAI, Anthropic, and Google, while some smaller or older models offer 128K–400K. Check the provider's model page for the exact limit, and note that the maximum *output* length is a separate, much smaller limit.

A huge window does not make context free:

-   **Cost and latency grow with input size.** You pay for every token you send, on every call.
-   **Quality can degrade with very long inputs.** Models are better at using information near the beginning and end of a long prompt than information buried in the middle (the "lost in the middle" effect). Sending only the relevant parts usually gives better answers than sending everything.

### A Simple Memory Management Strategy

A common strategy for managing long conversations is a "sliding window" of memory. We always keep the system prompt, but drop the oldest user/assistant messages once the history exceeds a token budget. The budget should leave room for the model's reply.

```python
def trim_conversation_history(
    messages: list[dict],
    max_input_tokens: int = 100_000,
    reserve_output_tokens: int = 4_000,
) -> list[dict]:
    """Returns a copy of the history that fits within the token budget,
    leaving room for the model's reply."""
    def count(msg: dict) -> int:
        return len(encoder.encode(msg["content"]))

    system = [m for m in messages[:1] if m["role"] == "system"]
    history = messages[len(system):]

    budget = max_input_tokens - reserve_output_tokens - sum(count(m) for m in system)
    if history and count(history[-1]) > budget:
        # Never silently drop the user's latest question.
        raise ValueError("The newest message alone exceeds the token budget.")

    kept: list[dict] = []
    # Walk backwards from the newest message, keeping as many as fit.
    for msg in reversed(history):
        if count(msg) > budget:
            break
        kept.append(msg)
        budget -= count(msg)

    return system + list(reversed(kept))
```

This returns a new list instead of modifying the caller's, and always keeps the most recent messages. More advanced strategies summarize older turns instead of discarding them, or store facts in a database and retrieve them when relevant (Chapter 27). Some providers now also offer server-side helpers for this, such as Anthropic's automatic context compaction.

## Key Takeaways

-   **Tokens are the unit of cost and latency.** You pay for every input and output token, and different providers tokenize the same text differently. Use `tiktoken` with `o200k_base` for quick estimates, and the response's `usage` object or the provider's token-counting endpoint for exact numbers.
-   **Save tokens where it counts.** The big savings come from controlling large inputs (conversation history, pasted documents) and output length, not from shaving words off a one-line instruction. Never cut context the model actually needs.
-   **Embeddings capture meaning as vectors.** `text-embedding-3-small` returns 1536 numbers per text, and you compare texts with cosine similarity (a plain dot product for OpenAI's normalized vectors; normalize first for other providers). Embed many texts in one API call.
-   **Semantic search ranks, it does not reason.** It finds matches that share no keywords, but the top hit is not guaranteed to be the right fix, so retrieve the top few candidates and let a model or a person decide.
-   **Every request is stateless.** The model only knows what you send in that call. Exceeding the context window gets the request rejected rather than silently forgotten, and your code decides what to keep, drop, or summarize.
-   **A large window is not free.** Cost and latency grow with input size and quality drops for information buried in the middle, so trim history with a sliding window that keeps the system prompt, returns a copy, and leaves room for the reply.

These three ideas—cost, meaning, and memory—are the levers you will constantly be pulling and balancing as an AI developer. In the next chapter, we will get our hands dirty and set up a professional development environment so you can start building real applications on this foundation.

## Exercises

**1. Warm-up:** Use `o200k_base` to count tokens for five IoT strings: a plain English status sentence, a pretty-printed JSON telemetry payload (for example `{"device_id": "TEMP-042", "temp_c": 85.2, "status": "WARNING"}` with indentation), the same payload as compact JSON with no spaces, a device UUID, and the status sentence translated into another language. Print a table of characters, tokens, and characters per token for each. Then send one of them through `client.responses.create` and compare `response.usage.input_tokens` with your estimate. You are done when you can explain which format is cheapest and why the billed count differs from your local count.

**2. Build:** Extend `find_most_similar` into `top_k(query, options, k=3)` that returns `(option, score)` pairs sorted by score, and grow the knowledge base to about eight troubleshooting steps. Write a labelled test set of eight user complaints, each paired with the step you consider correct, and embed the knowledge base only once instead of on every query. You are done when the script prints top-1 and top-3 hit rates for your test set and you have looked at every miss to decide whether the query, the knowledge base wording, or the expected label was the problem.

**3. Stretch:** Write tests (plain `assert` statements are fine) for `trim_conversation_history`: the system prompt is always kept, the caller's list is not modified, the result plus `reserve_output_tokens` fits the budget, the newest message is always included, and an oversized newest message raises `ValueError`. Then add a `summarize` option: instead of discarding the dropped older messages, replace them with a single model-written summary message placed right after the system prompt. You are done when all your tests pass (use a fake summarizer function in the tests so they need no API call) and a 50-turn sample conversation stays under budget while the summary still mentions the device ID from turn 1.

# References and Further Reading

- AI Tokens Explained: Understanding Context Windows and Processing. https://medium.com/@jimcanary/ai-tokens-explained-understanding-context-windows-and-processing-a99ca2dd9142
- What Are AI Tokens and Context Windows (And Why Should You Care)? https://simple.ai/p/tokens-and-context-windows
- From Tokens to Context Windows: Simplifying AI Jargon. Technology Policy Institute. https://techpolicyinstitute.org/publications/artificial-intelligence/from-tokens-to-context-windows-simplifying-ai-jargon/
- AI Tokens Explained - What They Are and Why They Matter. https://zenvanriel.nl/ai-engineer-blog/ai-tokens-explained-what-they-are-and-why-they-matter/
- Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023). https://arxiv.org/abs/2307.03172
- tiktoken (OpenAI's tokenizer library). https://github.com/openai/tiktoken

<!-- chapter-nav -->

---

[← Chapter 1: Welcome to AI Engineering](01-ai-engineering.md) | [Table of Contents](../README.md) | [Chapter 3: Setting Up Your Development Environment →](03-development-environment.md)
