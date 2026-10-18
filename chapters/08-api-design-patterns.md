# Chapter 8: API Design Patterns and Best Practices

Building a simple AI application is one thing; building a robust, scalable, and resilient one is an entirely different engineering challenge. As you move from prototypes to production, your application will face the harsh realities of the real world: network failures, API rate limits, service outages, and unpredictable user load.

This chapter is your guide to building production-grade AI systems. We will explore a set of battle-tested design patterns that address the unique challenges of working with third-party AI APIs. These patterns are not just about making your code work; they are about making it work reliably, efficiently, and cost-effectively under pressure. By mastering these techniques, you will be able to build applications that are not only intelligent but also enterprise-ready.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Implement robust retry strategies with exponential backoff to handle transient API failures.
-   Design and build rate limiters that respect different AI provider quotas.
-   Choose the right response mode (blocking, streaming, or batch) for different use cases.
-   Cache responses safely to reduce latency and API costs.
-   Design a multi-provider architecture with automatic failover for high availability.

## The Foundation: Robust Error Handling

Before we explore advanced patterns, we must master the basics. A production application never crashes because of a failed API call; it anticipates failure and handles it gracefully.

### From Naive to Nuanced Error Handling

A simple `try...except` block is the first step, but it's not enough.

```python
# A naive approach: catches everything, but tells you nothing.
def naive_ai_call(prompt: str):
    try:
        ...  # API call
    except Exception as e:
        print(f"Something went wrong: {e}")
        return None
```

A much better approach is to catch *specific* error types that the API library provides. This allows you to react differently to different kinds of failures. For example, a `RateLimitError` is temporary and can be retried, while an `AuthenticationError` is permanent and should not be.

```python
import openai

client = openai.OpenAI()
MODEL = "gpt-6-luna"

def better_error_handling(prompt: str) -> str | None:
    """Handles specific, known exceptions from the OpenAI library."""
    try:
        response = client.responses.create(model=MODEL, input=prompt)
        return response.output_text
    except openai.RateLimitError as e:
        print(f"Rate limit exceeded. Please wait before trying again. Details: {e}")
    except openai.AuthenticationError as e:
        print(f"Authentication failed. Check your API key. Details: {e}")
    except openai.BadRequestError as e:
        print(f"The request was invalid (fix the code, don't retry): {e}")
    except openai.APIError as e:
        print(f"An OpenAI API error occurred: {e}")
    return None
```

Create the client once and reuse it: it holds a connection pool, so creating a new client per call wastes time on new TCP/TLS handshakes.

## Pattern 1: Retries with Exponential Backoff

Network glitches and temporary service overloads are common. A simple retry can often resolve the issue. However, retrying immediately can make the problem worse, contributing to a "thundering herd" that overwhelms the service.

The professional solution is **exponential backoff with jitter**.
-   **Exponential Backoff:** The delay between retries increases exponentially (e.g., 1s, 2s, 4s, 8s).
-   **Jitter:** A small, random amount of time is added to the delay to prevent multiple clients from retrying at the exact same moment.

**Start with what the SDK already does.** The official OpenAI, Anthropic, and Google SDKs all retry transient failures (connection errors, timeouts, 429 rate limits, and 5xx server errors) with exponential backoff, by default twice, and they respect the server's `Retry-After` hint. Often all you need is to tune that:

```python
client = openai.OpenAI(max_retries=4, timeout=60.0)    # same idea: anthropic.Anthropic(max_retries=4)
```

Add your own retry layer only when you need behavior the SDK doesn't provide, for example retrying a *whole operation* (call the model, then validate its output) or one policy shared across providers. If you do, turn the SDK's retries off for that client, otherwise the two layers multiply (3 attempts × 3 attempts = 9 calls). The `tenacity` library provides a clean, declarative way to do this.

```bash
pip install tenacity
```

```python
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import openai

# This client does no retries of its own; tenacity owns the retry policy.
no_retry_client = openai.OpenAI(max_retries=0)

TRANSIENT_ERRORS = (
    openai.RateLimitError,         # 429
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,    # 5xx
)

@retry(
    stop=stop_after_attempt(4),
    # Exponential backoff with full jitter: random wait between 0 and 1s, 2s, 4s, ... capped at 30s.
    wait=wait_random_exponential(multiplier=1, max=30),
    retry=retry_if_exception_type(TRANSIENT_ERRORS),
)
def robust_ai_call(prompt: str) -> str:
    """An AI call with an explicit retry policy for transient errors only."""
    response = no_retry_client.responses.create(model=MODEL, input=prompt)
    return response.output_text

try:
    print(robust_ai_call("Write a short poem about resilient code."))
except Exception as e:
    print(f"The API call failed even after several retries: {e}")
```
Notice what is *not* retried: authentication errors, invalid requests (400), and content-policy refusals. Retrying those only wastes time and money.

## Pattern 2: Intelligent Rate Limiting

Every AI provider imposes rate limits: a maximum number of requests per minute (RPM) and, usually more important for LLMs, tokens per minute (TPM), both of which depend on your account tier. Exceeding them results in `RateLimitError` (HTTP 429) exceptions. A well-behaved application anticipates and respects these limits *before* making a call. Providers also report your remaining quota in response headers (for example `x-ratelimit-remaining-tokens`), which you can use to adapt.

A common way to do this is with the **token bucket algorithm**. Imagine each user has a bucket that can hold a certain number of "request tokens." The bucket is refilled at a constant rate. To make a request, you must take a token from the bucket. If the bucket is empty, you must wait.

```python
import time

class TokenBucketRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.capacity = requests_per_minute
        self.tokens_per_second = requests_per_minute / 60.0
        self.tokens = float(self.capacity)
        self.last_update_time = time.monotonic()

    def can_proceed(self) -> bool:
        """Checks if a request can be made, consuming a token if so."""
        now = time.monotonic()
        
        # Add new tokens that have accrued since the last update
        elapsed_time = now - self.last_update_time
        self.tokens += elapsed_time * self.tokens_per_second
        self.tokens = min(self.capacity, self.tokens) # Cap at max capacity
        
        self.last_update_time = now
        
        # Check if there are enough tokens to proceed
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

# Example: Simulate calls for a user on a free tier (e.g., 20 RPM)
limiter = TokenBucketRateLimiter(requests_per_minute=20)

for i in range(25):
    if limiter.can_proceed():
        print(f"Request #{i+1}: Allowed")
    else:
        print(f"Request #{i+1}: Denied. Must wait.")
    time.sleep(0.1) # Simulate rapid requests
```
By checking `limiter.can_proceed()` before each API call, your application can throttle itself to avoid hitting API rate limits. The same algorithm works for token limits: make the bucket's capacity your TPM limit and have each request take its estimated token count instead of `1`. This in-memory version protects a single process; when several servers share one API quota, keep the bucket in a shared store such as Redis (Chapter 18).

## Pattern 3: Choosing a Response Mode (Blocking, Streaming, or Batch)

Not all requests are the same, and there are three ways to get a response:

-   **Blocking (non-streaming):** Wait for the full response, then use it. Best for short answers and for anything your code must parse as a whole (JSON, a classification label).
-   **Streaming:** Receive text as it is generated. Best for long-form output shown to a person, because the user sees progress within a second instead of staring at a spinner.
-   **Batch API:** Submit many requests as one asynchronous job and collect the results later, typically within 24 hours, at roughly **half the price**. Best for offline work such as nightly report generation, classifying a backlog, or running evaluations (Chapter 21). OpenAI, Anthropic, and Google all offer one.

Here's a function that picks between the first two for interactive use.

```python
from typing import Iterator

def adaptive_ai_response(prompt: str) -> str | Iterator[str]:
    """Streams long-form requests; returns short answers in one piece."""
    
    # Heuristic: If the prompt asks for long-form content, stream it.
    streaming_keywords = ['write', 'explain', 'describe', 'story', 'report', 'detailed']
    should_stream = any(keyword in prompt.lower() for keyword in streaming_keywords)
    
    if should_stream:
        print("-> Using streaming response.")
        stream = client.responses.create(model=MODEL, input=prompt, stream=True)
        return (event.delta for event in stream if event.type == "response.output_text.delta")
    
    print("-> Using blocking response.")
    return client.responses.create(model=MODEL, input=prompt).output_text

# Test with a short query
result = adaptive_ai_response("What is the capital of Japan?")
print(result)

# Test with a long query
stream_result = adaptive_ai_response("Write a short story about a brave knight.")
for chunk in stream_result:
    print(chunk, end="", flush=True)
print()
```

## Pattern 4: Intelligent Caching

API calls are the slowest and most expensive part of most AI applications. If many users ask the same thing, an application-level response cache can return a stored answer instantly and for free. For production, a distributed cache like **Redis** lets every instance of your application share the same cache.

Two rules make response caching safe:

-   **The key must include everything that affects the output**: the model, the instructions, the full input, and generation settings. Otherwise a cached answer to one request is served for a different one.
-   **Only cache where a repeated answer is acceptable.** FAQ-style answers and expensive, deterministic transformations are good candidates. Personalized or time-sensitive answers are not, and neither are cases where users expect variety.

### Caching with Redis

First, install the Redis Python client: `pip install redis`.

```python
import hashlib
import json
import redis

class RedisAICache:
    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl: int = 3600):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = default_ttl  # 1 hour

    def _create_key(self, request: dict) -> str:
        """A stable key over every parameter that affects the output."""
        canonical = json.dumps(request, sort_keys=True)  # sort_keys: same dict -> same string
        return f"ai_cache:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def get(self, request: dict) -> str | None:
        return self.redis_client.get(self._create_key(request))

    def set(self, request: dict, response: str) -> None:
        self.redis_client.setex(self._create_key(request), self.default_ttl, response)

cache = RedisAICache()

def cached_ai_call(prompt: str, instructions: str = "You are a concise IoT expert.") -> str:
    """Checks the cache before calling the model."""
    request = {"model": MODEL, "instructions": instructions, "input": prompt}

    cached = cache.get(request)
    if cached is not None:
        print("-> Cache hit")
        return cached

    print("-> Cache miss. Calling API...")
    response = no_retry_client.responses.create(**request).output_text  # or robust_ai_call
    cache.set(request, response)
    return response

# The first call will be slow and hit the API.
print(cached_ai_call("What is Zigbee?"))
# The second call will be instantaneous.
print(cached_ai_call("What is Zigbee?"))
```

This is different from **provider-side prompt caching** (Chapters 6 and 21), where the provider reuses the processed *prefix* of your prompt (for example a long system prompt or document) and bills those tokens at a large discount, while still generating a fresh answer. The two complement each other: application caching skips the call entirely; prompt caching makes the calls you do make cheaper.

## Pattern 5: Multi-Provider Failover and Load Balancing

Relying on a single AI provider is risky. Services can experience outages or performance degradation. A resilient architecture uses multiple providers and can automatically switch between them.

### An Abstraction Layer for AI Providers

First, we create a common interface for different providers. Each provider gets a small function with the same signature, so the rest of the application never sees provider-specific syntax.

```python
from enum import Enum

import anthropic
import openai
from google import genai

class AIProviderEnum(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

class MultiProviderClient:
    """One `call()` method in front of three different SDKs."""

    def __init__(self):
        # Short timeouts and one retry each: the router below handles failover.
        self.openai = openai.OpenAI(max_retries=1, timeout=30.0)
        self.anthropic = anthropic.Anthropic(max_retries=1, timeout=30.0)
        self.google = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY

    def call(self, provider: AIProviderEnum, prompt: str) -> str:
        """Makes a call using the specified provider's API syntax."""
        print(f"--> Calling {provider.value}...")

        if provider == AIProviderEnum.OPENAI:
            return self.openai.responses.create(model="gpt-6-luna", input=prompt).output_text

        if provider == AIProviderEnum.ANTHROPIC:
            message = self.anthropic.messages.create(
                model="claude-sonnet-5-5",
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in message.content if b.type == "text")

        if provider == AIProviderEnum.GOOGLE:
            return self.google.models.generate_content(model="gemini-3.8-flash", contents=prompt).text

        raise ValueError(f"Unknown provider: {provider}")
```

### A Simple Failover Router

Now, we can build a router that tries providers in a preferred order until one succeeds.

```python
class FailoverRouter:
    def __init__(self, client: MultiProviderClient):
        self.client = client
        # Define the preferred order of providers
        self.provider_preference = [
            AIProviderEnum.OPENAI, 
            AIProviderEnum.ANTHROPIC, 
            AIProviderEnum.GOOGLE
        ]

    def call_with_failover(self, prompt: str) -> str:
        """Tries each provider in order until one succeeds."""
        for provider in self.provider_preference:
            try:
                response = self.client.call(provider, prompt)
                if response:
                    print(f"Success with {provider.value}!")
                    return response
            except Exception as e:
                print(f"Provider {provider.value} failed: {e}. Trying next...")
        
        return "Error: All AI providers are currently unavailable."

# --- Usage ---
multi_client = MultiProviderClient()
router = FailoverRouter(multi_client)
result = router.call_with_failover("Explain the concept of an API.")
print(f"Final Result: {result}")
```

This simple failover logic can greatly increase the availability of your application. A few cautions from production experience:

-   **Fail over only on errors that another provider can fix**: outages, timeouts, and rate limits. A 400 "invalid request" from your own bug will fail everywhere.
-   **Different models give different answers.** Prompts tuned for one model may behave differently on another, so evaluate your fallback models on your test set (Chapter 24) before you rely on them.
-   **Check your data agreements.** Sending a request to a second provider means sending your users' data to a second company.

More advanced routers add health checks, latency tracking, and circuit breakers (stop calling a provider for a while after repeated failures). Instead of building all of this yourself, you can also use an **LLM gateway** (open-source proxies such as LiteLLM, or the gateway features of cloud platforms) that offers one API in front of many providers.

## Conclusion: Building for the Real World

You have now learned five essential patterns for building production-ready AI applications. These techniques—retries, rate limiting, adaptive responses, caching, and failover—are the building blocks of robust and scalable systems.

By incorporating these patterns into your development workflow, you move from being a user of AI APIs to being an architect of intelligent systems. You are now prepared to build applications that can withstand the rigors of a production environment.

## Key Takeaways

-   **Catch specific SDK exceptions.** Retry transient errors (`RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError`) and fail fast on permanent ones such as `AuthenticationError` and `BadRequestError`. Create the client once and reuse it.
-   **Use the SDK's built-in retries first.** The official SDKs already retry twice with backoff, so tune `max_retries` and `timeout` before adding anything. If you add your own layer (for example `tenacity` with exponential backoff and jitter), set `max_retries=0` on that client so the two layers do not multiply.
-   **Throttle yourself with a token bucket** sized for both requests per minute and tokens per minute, and move the bucket to a shared store such as Redis when several processes share one quota.
-   **Pick the response mode per use case:** block for short or machine-parsed answers, stream long text shown to people, and use the providers' Batch APIs (about half price, results within about 24 hours) for offline work.
-   **Cache responses only where repeats are acceptable**, with a key that includes every input that affects the output (model, instructions, input, and generation settings). Provider-side prompt caching is a separate saving that complements it.
-   **Fail over across providers behind one `call()` interface**, but only on errors another provider can fix (outages, timeouts, rate limits), and evaluate the fallback models on your test set before you rely on them.

## Exercises

**1. Warm-up:** Write a test for your retry policy without calling any API. Apply the same `@retry` settings as `robust_ai_call` (with a near-zero wait so the test is fast) to a fake function that raises `openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))` on its first two calls and then succeeds, and to a second fake that raises `ValueError` as a stand-in for a permanent error. Done means assertions pass showing the first function was called exactly 3 times and returned its value, and the second was called exactly once and its exception propagated.

**2. Build:** Extend `TokenBucketRateLimiter` so each request takes a `cost`, add a `wait_and_acquire(cost)` method that sleeps until enough capacity has accrued, and raise an error immediately if `cost` is larger than the bucket's capacity. Combine two buckets, one for requests per minute and one for tokens per minute, estimating each request's tokens with the counting approach from Chapter 2. Done means a simulation of 100 fake requests of random sizes against limits of 60 RPM and 10,000 TPM records a timestamp and cost per request, and a check over that log confirms the token bucket's guarantee: in any window of `t` seconds, the requests (and tokens) sent never exceed the bucket's capacity plus `t` times its refill rate.

**3. Stretch:** Make `FailoverRouter` selective: fail over only on each SDK's transient errors (connection errors, timeouts, rate limits, 5xx) and re-raise everything else immediately. Add a circuit breaker that skips a provider for 60 seconds after 3 consecutive failures, and make the clock injectable so you can test the cooldown without sleeping. Done means that pointing the OpenAI client at an unreachable address (for example `base_url="http://127.0.0.1:9"`) makes requests fail over to Anthropic, the log shows OpenAI being skipped after the third failure until the cooldown ends, and a deliberately invalid request raises at once without trying the other providers.

# References and Further Reading

- 5 Essential API Design Patterns for Successful AI Model Implementation. https://dev.to/stellaacharoiro/5-essential-api-design-patterns-for-successful-ai-model-implementation-2dkk
- Beyond the Gang of Four: Practical Design Patterns for Modern AI Systems. https://www.infoq.com/articles/practical-design-patterns-modern-ai-systems/
- Generative AI Design Patterns: A Comprehensive Guide. https://towardsdatascience.com/generative-ai-design-patterns-a-comprehensive-guide-41425a40d7d0
- AI Patterns: A Structured Approach to Artificial Intelligence Development. https://medium.com/@xavier.mehaut/ai-patterns-a-structured-approach-to-artificial-intelligence-development-9d97903c8e34
- Exponential Backoff And Jitter (AWS Architecture Blog). https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
- Tenacity documentation. https://tenacity.readthedocs.io/

<!-- chapter-nav -->

---

[← Chapter 7: Mastering the Google Gemini API](07-google-gemini-api-guide.md) | [Table of Contents](../README.md) | [Chapter 9: Fundamental Prompt Engineering →](09-fundamental-prompt-engineering.md)
