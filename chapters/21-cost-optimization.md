# Chapter 21: AI Application Cost Optimization

When you first begin building with AI, the focus is on what's possible. The creative energy is high, and the results can feel magical. Then the first monthly invoice from your AI provider arrives, and a different kind of magic trick appears: making your budget disappear.

A startup I worked with learned this the hard way. They built a customer service bot that was very effective at solving user problems, and even more effective at burning through their runway: their monthly AI bill grew past $50,000. The problem wasn't that AI is inherently expensive; it was that they treated model calls like any other web service call, without considering their unique cost structure.

This chapter is about the levers that actually move an AI bill. Applied in the right order (measure first, then the "free" savings, then trade-offs), they routinely cut costs by half or more without hurting quality.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Understand what you are billed for, including cached input and reasoning tokens.
-   Measure costs from the provider's reported usage instead of estimates.
-   Use the two biggest levers available today: **prompt caching** and **Batch APIs**.
-   Control output and reasoning costs, and route requests to the cheapest model that works.
-   Use response caching (including semantic caching) safely.
-   Calculate the return on investment (ROI) of an AI feature.

## Understanding the AI Cost Structure

AI API pricing is consumption-based, and the unit is the **token** (Chapter 2). A request's cost has several components, each priced per million tokens:

-   **Input tokens:** everything you send (system prompt, tool definitions, conversation history, documents, the user's message).
-   **Cached input tokens:** input that matches a prefix the provider has recently processed, billed at a large discount (commonly 90% off or more). Some providers also charge a small premium the first time a prefix is written to the cache.
-   **Output tokens:** everything the model generates, typically 4–8× the price of input.
-   **Reasoning tokens:** on reasoning models, the internal "thinking" tokens. You don't see them (or see only a summary), but **they are billed as output tokens**, and on hard problems they can outnumber the visible answer many times over.

Prices differ enormously between model tiers, often by a factor of 20 to 50 between a provider's cheapest and most capable model, and they change frequently. Keep prices in configuration and check the providers' pricing pages; never hard-code them deep in application logic.

### Measure from Usage, Not Estimates

Every API response reports exactly what you were billed for. Use those numbers, not a local tokenizer, which is only an approximation (and wrong for other providers' models).

```python
import openai

client = openai.OpenAI()
MODEL = "gpt-6-luna"

# Illustrative prices in USD per 1M tokens (verify on the providers' pricing pages).
PRICES = {
    "gpt-6-luna":        {"input": 0.10, "cached_input": 0.01, "output": 0.50},
    "claude-haiku-5-5":  {"input": 0.10, "cached_input": 0.01, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "cached_input": 0.20, "output": 10.00},
    "claude-opus-5-5":   {"input": 4.00, "cached_input": 0.20, "output": 20.00},
}

def openai_call_cost(response) -> float:
    """Exact cost of one Responses API call, from its usage report."""
    p = PRICES[MODEL]
    usage = response.usage
    cached = usage.input_tokens_details.cached_tokens       # included in input_tokens
    uncached = usage.input_tokens - cached
    return (uncached * p["input"] + cached * p["cached_input"] + usage.output_tokens * p["output"]) / 1_000_000

response = client.responses.create(model=MODEL, input="Summarize: GW-03 rebooted 4 times overnight; uplink RSSI -95 dBm.")
print(f"Output tokens: {response.usage.output_tokens} "
      f"(of which reasoning: {response.usage.output_tokens_details.reasoning_tokens})")
print(f"Cost: ${openai_call_cost(response):.6f}")
```

Each provider reports usage slightly differently (for example, Anthropic reports `cache_read_input_tokens` and `cache_creation_input_tokens` separately from `input_tokens`), so write one small cost function per provider. Then **log the cost of every call** with the feature, user, and model that caused it (Chapter 23). The first time you do this, you will usually find that one or two features, or one runaway loop, account for most of the bill. Optimize those first.

### Where the Money Hides

-   **Context accumulation:** Chat applications resend the whole history every turn, so the input cost of a conversation grows roughly quadratically with its length.
-   **Bloated prompts:** Long system prompts, many tool definitions, and too many retrieved documents are paid for on *every* call.
-   **Uncapped reasoning and output:** High reasoning effort on simple tasks, and verbose answers no one reads.
-   **Retries and agent loops:** A failing validation retried five times, or an agent that wanders for 30 steps, multiplies the cost of a task.
-   **The wrong model:** A frontier model doing a job a small model does equally well.

## Lever 1: Prompt Caching (Provider-Side)

This is usually the largest and easiest saving available today. When many requests share the same *prefix* (the same long system prompt, tool definitions, few-shot examples, or document), the provider can reuse its processing of that prefix. Cached input is billed at a fraction of the normal price and processed faster.

-   **OpenAI** applies caching automatically to sufficiently long prompts; check `usage.input_tokens_details.cached_tokens` to see it working.
-   **Anthropic** caches when you ask: add `cache_control` (Chapter 6 shows the code).
-   **Google** offers implicit caching on recent models plus explicit cache objects for large, reused content.

Caching matches the prefix **exactly**, so the rule is: **stable content first, variable content last.**

```python
# Good: identical prefix on every call -> cacheable
SYSTEM_PROMPT = open("support_playbook.md").read()   # thousands of tokens, rarely changes

def answer_ticket(ticket_text: str) -> str:
    return client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,       # stable: first
        input=ticket_text,                # variable: last
    ).output_text

# Bad: a timestamp at the top changes the prefix on every call -> nothing is ever cached
# instructions = f"Current time: {datetime.now()}\n\n{SYSTEM_PROMPT}"
```

Common silent cache-breakers: timestamps or request IDs near the start of the prompt, tool lists in a non-deterministic order, and JSON serialized without sorted keys. Verify with the usage fields that your cache hit rate is what you expect.

## Lever 2: Batch APIs for Non-Urgent Work

OpenAI, Anthropic, and Google all offer **Batch APIs**: you upload many requests at once, they are processed asynchronously (results typically within 24 hours, often much sooner), and you pay roughly **50% of the normal price**. Batch jobs also have separate, larger rate limits.

Anything that doesn't need an answer in seconds is a candidate: nightly report generation, classifying or summarizing a backlog of tickets or logs, generating embeddings for a document collection, and running your evaluation suite (Chapter 24).

```python
import json

# 1. Write one request per line (JSONL). custom_id lets you match results to inputs.
log_entries = {"log-001": "GW-03 rebooted 4 times overnight", "log-002": "TEMP-12 reading -999.0"}
with open("batch_input.jsonl", "w") as f:
    for custom_id, text in log_entries.items():
        f.write(json.dumps({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {"model": MODEL, "input": f"Classify this IoT log as Power, Connectivity, or Sensor_Fault: {text}"},
        }) + "\n")

# 2. Upload the file and create the batch.
batch_file = client.files.create(file=open("batch_input.jsonl", "rb"), purpose="batch")
batch = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/responses", completion_window="24h")
print(batch.id, batch.status)

# 3. Later: when client.batches.retrieve(batch.id).status == "completed",
#    download the output file (client.files.content(batch.output_file_id)) and match lines by custom_id.
```

## Lever 3: Control Output and Reasoning

Output tokens are the most expensive tokens, and reasoning tokens are output tokens.

-   **Set reasoning effort per task.** Use low effort for extraction, classification, and routing, and reserve high effort for genuinely hard problems. This is often the single biggest cost lever *within* one model, so measure quality at each level instead of defaulting to the highest.
-   **Ask for the length you need.** "Answer in at most three sentences" or a compact JSON schema (Chapter 11) beats letting the model write an essay.
-   **Cap output with `max_output_tokens`** as a safety net against runaway generations (leaving enough room for reasoning, Chapter 5).

## Lever 4: Trim the Input You Send on Every Call

Concise wording matters far less than the large blocks of context you resend:

-   **Conversation history:** keep a window of recent turns and summarize older ones (Chapter 2).
-   **Retrieved documents:** retrieve fewer, better chunks (reranking, Chapter 27) instead of stuffing ten "just in case."
-   **Tool definitions:** send only the tools relevant to the current task.

What *doesn't* help much: shaving polite words from a one-line instruction, or replacing newlines with `|` characters. Formatting characters are cheap, and clear structure often improves accuracy, so a "compressed" prompt that confuses the model costs more in retries than it saves.

## Lever 5: Route to the Cheapest Model That Works

Not every task needs the most capable model. A **model router** sends each request to the cheapest model that handles it well.

```python
class SmartModelRouter:
    """Routes requests by task type. The tiers and model names come from configuration."""
    TIERS = {
        "simple":   {"model": "gpt-6-luna", "effort": "low"},      # classification, extraction, short answers
        "moderate": {"model": "gpt-6-luna", "effort": "medium"},   # summaries, standard Q&A
        "complex":  {"model": "gpt-6-sol",  "effort": "high"},     # multi-step reasoning, planning, code
    }

    def classify_task(self, task_type: str) -> str:
        # Route on information you already have (the feature or endpoint) whenever possible.
        return {"tag_log": "simple", "summarize": "moderate", "plan_maintenance": "complex"}.get(task_type, "moderate")

    def run(self, task_type: str, prompt: str) -> str:
        tier = self.TIERS[self.classify_task(task_type)]
        return client.responses.create(
            model=tier["model"], input=prompt, reasoning={"effort": tier["effort"]}
        ).output_text

router = SmartModelRouter()
print(router.run("tag_log", "Tag this log line: 'TEMP-12 reading -999.0'"))
```

Routing on the *feature* (which endpoint or job the request comes from) is far more reliable than guessing complexity from keywords in the prompt. When you do need to classify free-form requests, use a small model with a fixed label set. Two cautions from practice: before building a multi-model cascade, compare against the simpler option of one capable model at lower reasoning effort, which is often just as cheap per *completed* task; and a "try cheap, fall back to expensive if validation fails" loop only saves money if the cheap model passes most of the time and your validation is reliable.

## Lever 6: Response Caching, Done Safely

If identical requests recur (FAQ answers, repeated report sections), an application-level cache can skip the call entirely (Chapter 8 shows a Redis implementation keyed on every parameter that affects the output).

**Semantic caching** extends this to paraphrases: embed the query, find a previous query with high cosine similarity, and return its answer. It can raise hit rates, but it is easy to get dangerously wrong: queries that differ in one crucial detail are often nearly identical in embedding space.

```python
import numpy as np

class SemanticCache:
    """Embedding-based cache with a guard for entity mismatches."""
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.entries: list[tuple[np.ndarray, str, str]] = []   # (embedding, query, response)

    def _embed(self, text: str) -> np.ndarray:
        data = client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding
        return np.array(data)   # OpenAI embeddings are unit-length, so dot product = cosine similarity

    @staticmethod
    def _entities(text: str) -> set[str]:
        # Tokens containing digits: device IDs, dates, amounts. "T-101" and "T-102" must never share an answer.
        return {tok.strip("?.,!").upper() for tok in text.split() if any(c.isdigit() for c in tok)}

    def get(self, query: str) -> str | None:
        if not self.entries:
            return None
        q = self._embed(query)
        scores = [float(q @ emb) for emb, _, _ in self.entries]
        best = int(np.argmax(scores))
        _, cached_query, cached_response = self.entries[best]
        if scores[best] >= self.threshold and self._entities(query) == self._entities(cached_query):
            return cached_response
        return None

    def set(self, query: str, response: str) -> None:
        self.entries.append((self._embed(query), query, response))
```

Note that every lookup costs an embedding call (cheap, but not free) and adds latency. In production, store the vectors in a vector index (Chapter 27), give entries an expiry, use semantic caching only for content that doesn't depend on the user or on fast-changing data, and measure the false-hit rate before turning it on.

## Calculating Return on Investment (ROI)

Cost optimization is meaningless without understanding the value the AI feature provides.

`ROI = (Value Gained - Total Cost) / Total Cost`

```python
import json

class ROICalculator:
    def __init__(self, human_analyst_hourly_rate: float = 75.0):
        self.hourly_rate = human_analyst_hourly_rate

    def calculate_feature_roi(self, monthly_api_cost: float, monthly_other_costs: float,
                              hours_saved_per_month: float, revenue_increase_per_month: float) -> dict:
        cost = monthly_api_cost + monthly_other_costs     # include hosting, tooling, and maintenance time
        value = hours_saved_per_month * self.hourly_rate + revenue_increase_per_month
        return {
            "monthly_cost": cost,
            "monthly_value": value,
            "monthly_net_gain": value - cost,
            "roi_percentage": (value - cost) / cost * 100 if cost > 0 else float("inf"),
        }

roi = ROICalculator().calculate_feature_roi(
    monthly_api_cost=1500,             # after optimization
    monthly_other_costs=2000,          # hosting, monitoring, a share of engineering time
    hours_saved_per_month=80,          # analysts no longer triage logs by hand
    revenue_increase_per_month=5000,   # estimated value of prevented downtime
)
print(json.dumps(roi, indent=2))
```

With these numbers the feature returns a positive ROI even after counting the non-API costs, which people often forget. The hard part is not the formula but honest inputs: measure hours saved and downtime prevented, rather than assuming them.

## Conclusion

Cost optimization is about maximizing value per dollar, not just cutting spending. Work through the levers in order:

1.  **Measure** cost per call, per feature, and per completed task from the providers' usage reports.
2.  **Cache prompt prefixes:** stable content first, variable content last.
3.  **Batch** everything that isn't interactive, at about half the price.
4.  **Control output and reasoning effort** per task.
5.  **Trim the large inputs** (history, retrieved context, tool lists), not the small words.
6.  **Route** each task to the cheapest model that passes your evals.
7.  **Cache responses** where repeats are acceptable, and semantic caching only with safeguards.

The startup that was spending $50,000 a month? By measuring first and then applying these techniques, they reduced their bill to under $8,000 while improving their service's responsiveness.

## Key Takeaways

-   Compute cost from the usage each response reports (uncached input, cached input, and output tokens, with reasoning tokens billed as output), not from a local tokenizer. Write one small cost function per provider and log the cost of every call with its feature, user, and model.
-   Prompt caching matches the prefix exactly, so put stable content (system prompt, tool definitions, examples) first and variable content last. Timestamps near the top, tools in random order, and unsorted JSON silently break it; confirm hits with `usage.input_tokens_details.cached_tokens`.
-   Batch APIs cost about half the normal price and suit anything that isn't interactive: nightly reports, backlog classification, embeddings, and eval runs. Use `custom_id` to match results back to inputs.
-   Set reasoning effort per task, ask for the length you need, and cap runaway generations with `max_output_tokens`. On the input side, trim the big blocks you resend (history, retrieved chunks, tool lists) rather than polite words.
-   Route on the feature or endpoint a request comes from, to the cheapest model that passes your evals. Before building a cascade, compare it with one capable model at lower reasoning effort, measured per completed task.
-   Semantic caching needs a high similarity threshold, an entity guard so "T-101" never gets "T-102"'s answer, expiry, and a measured false-hit rate. ROI only means something if it counts non-API costs and uses measured, not assumed, value.

## Exercises

**1. Warm-up:** Use `answer_ticket` with a system prompt long enough to be cached (a few thousand tokens), call it three times with different tickets, and print `cached_tokens` and `openai_call_cost` for each call. Then move a timestamp to the front of the instructions, as in the "bad" example, and repeat. You are done when you can show cached tokens above zero on the second and third calls of the good layout, zero for the bad layout, and the cost difference between the two.

**2. Build:** Change `openai_call_cost` to take the model name as a parameter, add a price entry for `gpt-6-sol`, and write a `log_call_cost(response, model, feature, user_id)` helper that appends one JSON line per call to a file. Send 20 requests through `SmartModelRouter` across the `tag_log`, `summarize`, and `plan_maintenance` features, then write a short script that reads the file and prints total cost, cost per feature, and each feature's share of the bill. Done means the per-feature totals add up exactly to the sum of the individual call costs, and you can say which feature to optimize first.

**3. Stretch:** Measure the false-hit rate of `SemanticCache` before trusting it. Build a labelled set of at least 30 query pairs: paraphrases that should share an answer, and near-duplicates that must not (different device IDs, different time ranges, and negations such as "Is T-101 online?" versus "Is T-101 offline?"). Report hits and false hits at thresholds of 0.90, 0.95, and 0.98, with and without the entity guard. You are done when you can name the threshold with zero false hits on your set, and list the kinds of near-duplicates the entity guard does not catch.

# References and Further Reading

- Prompt caching (OpenAI). https://platform.openai.com/docs/guides/prompt-caching
- Batch API (OpenAI). https://platform.openai.com/docs/guides/batch
- Prompt caching (Claude documentation). https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Batch processing (Claude documentation). https://platform.claude.com/docs/en/build-with-claude/batch-processing
- Context caching (Gemini API). https://ai.google.dev/gemini-api/docs/caching
- [Three proven strategies for optimizing AI costs (Google Cloud, 2024)](https://cloud.google.com/transform/three-proven-strategies-for-optimizing-ai-costs)
