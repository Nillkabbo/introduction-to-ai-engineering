# Chapter 15: Building Production-Ready Agents

So far, we have built agents in the clean, predictable environment of our development machines. We have explored architectures, given them tools, and taught them to reason. But the journey from a clever prototype to a robust, reliable production system is one of the most challenging transitions in software engineering.

A production agent isn't just a smart algorithm; it's a piece of critical infrastructure. It must be resilient to failure, observable when things go wrong, and scalable as demand grows. It must handle messy real-world data and unpredictable edge cases without crashing.

In this chapter, we will bridge that gap. We will take off our lab coats and put on our hard hats, focusing on the patterns and practices required to build agents that can be trusted to run 24/7.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Evaluate and choose the right agent framework for your project's needs.
-   Design and implement robust state management systems that survive restarts.
-   Build communication protocols for multi-agent coordination.
-   Instrument your agents with comprehensive logging, metrics, and tracing.
-   Architect a complete, production-grade agent for an industrial automation use case.
-   Implement strategies for graceful failure handling and recovery.

## The Production Gauntlet: From Prototype to Resilient System

The differences between a prototype and a production agent are stark. A prototype lives in memory and is forgiven for its failures. A production agent must assume that failure is inevitable and be built to withstand it.

Let's look at the contrast. Here's a simple prototype agent. It works, but it's brittle.

```python
# A simple prototype agent. DO NOT USE IN PRODUCTION.
class PrototypeAgent:
    def __init__(self):
        self.memory = []  # In-memory state is lost on restart!
    
    def process(self, task: str) -> str:
        # No error handling, no logging, no persistence.
        # A single API failure or process crash erases everything.
        print(f"Processing task: {task}")
        result = f"Completed: {task}" 
        self.memory.append(result)
        return result
```

Now, consider the architectural skeleton of a production-grade agent. It's designed from the ground up for resilience and observability.

```python
import json
import logging
import time

class ProductionAgent:
    """An agent architected for production reliability."""
    
    def __init__(self, agent_id: str, state_store, metrics_collector):
        self.agent_id = agent_id
        self.state_store = state_store          # For persistence (async interface, below)
        self.metrics = metrics_collector        # For monitoring
        self.logger = logging.getLogger(f"agent.{agent_id}")   # configure JSON handlers at startup
        self.state: dict = {}

    async def initialize(self):
        """Loads persisted state; call once on startup (constructors can't await)."""
        self.state = await self.state_store.load(self.agent_id) or {"tasks_completed": 0}

    async def process(self, task: dict):
        """Processes a task with full production safeguards."""
        task_id = task.get("id", "unknown_task")
        start = time.monotonic()
        
        self.logger.info(json.dumps({"event": "task_started", "task_id": task_id}))
        
        try:
            # Main logic would go here
            result = f"Completed task: {task.get('description')}"

            # Record metrics for success
            self.metrics.record_success(agent_id=self.agent_id, duration=time.monotonic() - start)
            # Persist state after successful operation
            self.state["tasks_completed"] += 1
            await self.state_store.save(self.agent_id, self.state)
            
            self.logger.info(json.dumps({"event": "task_completed", "task_id": task_id}))
            return {"status": "success", "result": result}
            
        except Exception as e:
            # Record metrics for failure
            self.metrics.record_failure(agent_id=self.agent_id, error_type=type(e).__name__)
            self.logger.error(
                json.dumps({"event": "task_failed", "task_id": task_id, "error": str(e)}),
                exc_info=True # Include stack trace
            )
            return {"status": "error", "error": str(e)}
```

This chapter is about building the pieces—`StateStore`, `MetricsCollector`, and more—that make this robust architecture possible.

## Choosing Your Foundation: Agent Frameworks

While you can build an agent from scratch (and Chapter 14's loop is most of an agent), frameworks provide tested components for the agent loop, memory, persistence, tracing, and human-in-the-loop steps. The landscape moves fast, and several frameworks popular in 2024 have since been redesigned or retired, so it pays to understand the categories rather than memorize APIs:

| Category | Examples (as of October 2026) | Use when |
| --- | --- | --- |
| **Provider agent SDKs** | OpenAI Agents SDK, Claude Agent SDK, Google Agent Development Kit (ADK) | You mostly use one provider and want its best-supported loop, tools, and tracing. |
| **Graph/state frameworks** | LangGraph, and LangChain 1.x (built on LangGraph) | You need explicit control over state, branching, checkpoints, and resumable human approval. |
| **Multi-agent / team frameworks** | CrewAI, Microsoft Agent Framework (successor to AutoGen and Semantic Kernel) | You want role-based teams of agents with built-in coordination. |
| **Typed, lightweight libraries** | PydanticAI and similar | You want type-safe tools and outputs with minimal abstraction. |
| **Managed agent platforms** | Hosted agent services from the major providers and clouds | You want the provider to run the loop and the sandbox where tools execute. |

Whatever you choose, keep two things in your own hands: your **tool implementations** (with their permission checks), and your **evaluations** (Chapter 24). Those outlive any framework migration.

### LangChain 1.x: A Single-Agent Example

LangChain is one of the most widely used open-source frameworks, with a vast ecosystem of integrations. Version 1.0 (late 2025) simplified the agent API to a single `create_agent` function built on LangGraph. The older `AgentExecutor`-style code found in many tutorials has moved to the `langchain-classic` package.

```python
# pip install langchain langchain-openai
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# 1. Tools are plain, typed Python functions; the docstring becomes the description.
def check_iot_status(device_id: str) -> str:
    """Checks the current status of a specific IoT device."""
    return f"Device {device_id} is online."

def restart_device(device_id: str) -> str:
    """Restarts a specific IoT device."""
    return f"Device {device_id} restart initiated."

# 2. Assemble the agent. The checkpointer gives it per-conversation memory.
agent = create_agent(
    model="openai:gpt-6-luna",
    tools=[check_iot_status, restart_device],
    system_prompt="You are an industrial automation specialist. Restart devices only when they are online and unresponsive.",
    checkpointer=InMemorySaver(),   # use a database-backed checkpointer in production
)

# 3. Run it. The thread_id identifies the conversation, so follow-ups remember context.
config = {"configurable": {"thread_id": "operator-42"}}
result = agent.invoke({"messages": [{"role": "user", "content": "Check sensor_001."}]}, config)
print(result["messages"][-1].content)
```

**Best for:** Agents that need many integrations, persistent conversations, and LangGraph features such as pausing for human approval and resuming later.

### Multi-Agent Frameworks

**Microsoft Agent Framework** is the successor to **AutoGen**, which Microsoft moved to maintenance mode in late 2025 (a community fork continues as AG2). The idea that AutoGen popularized, specialized agents that solve a problem by exchanging messages, lives on. For example:

1.  A `MonitorAgent` detects an anomaly.
2.  It sends a message to a `DiagnosticAgent`: "I've detected a pressure spike on Pump-07."
3.  The `DiagnosticAgent` analyzes the problem and replies with a root cause and a suggested fix.
4.  An `OperatorAgent` executes the fix (for example, by calling a `restart_pump` tool), after human approval.

**Best for:** Problems that genuinely decompose into separate roles. Chapter 16 discusses when multiple agents help and when they mostly add cost.

### CrewAI: Task-Oriented Agent Teams

CrewAI focuses on orchestrating teams of agents to accomplish a set of tasks in a structured, hierarchical way. It formalizes the concepts of roles, goals, and sequential task execution.

**Philosophy:** A "crew" of agents with defined roles works together like a project team to complete a mission, one task at a time.

A typical CrewAI workflow involves:

1.  **Defining Agents:** You create agents with specific `roles` (e.g., "Safety Officer"), `goals` (e.g., "Ensure all operations meet safety standards"), and `tools`.
2.  **Defining Tasks:** You create a list of tasks, which can have dependencies on each other.
3.  **Forming a Crew:** You assemble the agents and tasks into a `Crew`.
4.  **Executing:** The crew manager assigns tasks to the most suitable agents in sequence, passing the results of one task as context to the next.

**Best for:** Process automation, workflows with clear steps, and simulating the output of a human team (e.g., a research report generated by a "Researcher" and an "Editor" agent).

## The Agent's Brain: State Management and Persistence

An agent's state—its memory, current task, and configuration—is its most valuable asset. In production, this state cannot live only in memory. It must be persisted to an external store to survive restarts, crashes, and scaling events.

The solution is to use a `StateStore`, an abstraction for saving and loading agent data.

### Designing a StateStore

We start by defining a simple, abstract interface that any storage backend can implement.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class StateStore(ABC):
    """Abstract base class for a key-value state storage system."""
    
    @abstractmethod
    async def save(self, agent_id: str, state: Dict[str, Any]) -> bool:
        """Saves the state for a given agent."""
        pass

    @abstractmethod
    async def load(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Loads the state for a given agent."""
        pass

    @abstractmethod
    async def delete(self, agent_id: str) -> bool:
        """Deletes the state for a given agent."""
        pass
```

### Concrete Implementations

Now we can create concrete implementations for different backends, choosing the right one based on our application's needs.

For local development or simple applications, a file-based store is sufficient.

```python
import aiofiles
import os

class FileStateStore(StateStore):
    """A simple file-based state store for development."""
    def __init__(self, base_path: str = "./agent_states"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    async def save(self, agent_id: str, state: Dict[str, Any]) -> bool:
        file_path = os.path.join(self.base_path, f"{agent_id}.json")
        try:
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(state, indent=2, default=str))
            return True
        except Exception:
            # Add logging here
            return False

    # ... implement load and delete ...
```

For high-performance production systems, a dedicated in-memory database like Redis is a superior choice.

```python
import redis.asyncio as aioredis

class RedisStateStore(StateStore):
    """A shared, fast state store using Redis."""
    def __init__(self, redis_url: str = "redis://localhost"):
        self.redis = aioredis.from_url(redis_url, decode_responses=True)

    async def save(self, agent_id: str, state: Dict[str, Any]) -> bool:
        try:
            # JSON, not pickle: unpickling data an attacker can modify executes arbitrary code.
            await self.redis.set(f"agent:state:{agent_id}", json.dumps(state, default=str))
            return True
        except aioredis.RedisError:
            # Add logging here
            return False

    async def load(self, agent_id: str) -> Optional[Dict[str, Any]]:
        raw = await self.redis.get(f"agent:state:{agent_id}")
        return json.loads(raw) if raw else None

    async def delete(self, agent_id: str) -> bool:
        return bool(await self.redis.delete(f"agent:state:{agent_id}"))
```

Redis is fast but, depending on configuration, can lose recent writes on a crash. For state you can't afford to lose (an agent halfway through a multi-step task with real-world side effects), use a durable database, or a durable-execution engine (Chapter 28).

By coding our agent against the `StateStore` interface, we can easily switch between backends without changing the agent's core logic.

### Stateful Agent Implementation

A stateful agent loads its state upon initialization and periodically checkpoints its state back to the store.

```python
from datetime import datetime

class StatefulAgent:
    def __init__(self, agent_id: str, state_store: StateStore):
        self.agent_id = agent_id
        self.state_store = state_store
        self.state = {} # Will be loaded in initialize

    async def initialize(self):
        """Loads state from the store or creates a new one."""
        self.state = await self.state_store.load(self.agent_id)
        if not self.state:
            print(f"No existing state for {self.agent_id}. Creating new state.")
            self.state = {"created_at": datetime.now().isoformat(), "history": []}
            await self._checkpoint()
        else:
            print(f"Successfully loaded state for {self.agent_id}.")
    
    async def _checkpoint(self):
        """Saves the current state to the store."""
        print(f"Checkpointing state for {self.agent_id}...")
        await self.state_store.save(self.agent_id, self.state)
```

## Agents in Conversation: Communication Protocols

In many systems, agents need to coordinate. This requires a robust communication protocol. Just like with state, we can define an abstract interface for sending and receiving messages.

```python
# A standard message format using Pydantic for validation
from pydantic import BaseModel, Field
import uuid

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    recipient: str
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

# Abstract Communication Protocol
class CommunicationProtocol(ABC):
    @abstractmethod
    async def send(self, message: Message) -> bool:
        pass
    
    @abstractmethod
    async def receive(self, agent_id: str, timeout: int = 1) -> Optional[Message]:
        pass
```

A common and scalable implementation uses a message broker. Note that Redis Pub/Sub is fire-and-forget (a message sent while an agent is offline is lost); for messages that must be delivered, use Redis Streams, RabbitMQ, or Kafka, which store messages until they are acknowledged.

Between organizations and vendors, open protocols are emerging for this layer: **MCP** (Chapter 14) standardizes how agents reach tools and data, and the **Agent2Agent (A2A)** protocol, started by Google and now an open project, standardizes how agents built by different teams discover and delegate work to one another.

## Guardrails Every Production Agent Needs

Beyond persistence and messaging, agents need limits that ordinary services don't:

-   **Step and cost budgets.** Cap the number of model calls and tool calls per task, and the total tokens or dollars, and stop gracefully when a budget runs out.
-   **Idempotent tools.** An agent (or a retry) may call the same tool twice. Design write operations so that repeating them is harmless, for example by passing a unique request ID the backend uses to ignore duplicates.
-   **Timeouts on everything**, including tools, so one hung API doesn't freeze the agent.
-   **Human approval** for irreversible or high-impact actions (Chapters 14 and 28).
-   **Evaluation before deployment.** Agents fail in new ways when prompts, tools, or models change; run a set of realistic tasks and check the outcomes before every release (Chapter 24).

## Seeing in the Dark: Production Observability

When an agent is running 24/7, you need to know what it's doing, how it's performing, and what went wrong when it fails. This is **observability**, and it rests on three pillars.

1.  **Logging:** Structured, machine-readable logs are essential. Instead of printing plain text, log JSON objects. This allows you to easily search, filter, and analyze logs in a tool like Elasticsearch or Datadog.

    ```python
    import structlog
    
    # Configure structlog for JSON output at the start of your application
    # ...
    
    log = structlog.get_logger()
    log.info("task_started", agent_id="agent_001", task_id="t123")
    # Output: {"event": "task_started", "agent_id": "agent_001", ...}
    ```

2.  **Metrics:** Metrics are numeric measurements of your agent's health. They answer questions like "How many tasks are processed per second?" or "What is the error rate?". Using a library like `prometheus-client`, you can expose these metrics for a monitoring system like Prometheus to scrape.

    ```python
    from prometheus_client import Counter, Histogram

    TASKS_PROCESSED = Counter(
        'agent_tasks_processed_total', 
        'Total tasks processed by an agent',
        ['agent_id', 'status'] # Labels to slice the data
    )
    
    TASK_DURATION = Histogram(
        'agent_task_duration_seconds',
        'Time taken to process a task',
        ['agent_id']
    )
    
    # In your agent's process method:
    # TASKS_PROCESSED.labels(agent_id=self.agent_id, status='success').inc()
    # TASK_DURATION.labels(agent_id=self.agent_id).observe(duration_in_seconds)
    ```

3.  **Tracing:** Tracing follows a single request as it flows through multiple agents or services. If an `OrchestratorAgent` sends a task to a `WorkerAgent`, a distributed trace ties their logs and metrics together, giving you a complete picture of that one operation. Tools like OpenTelemetry are the industry standard for implementing tracing.

## Conclusion: Engineering for Reliability

Building a production-ready agent is a discipline that blends AI prompting with robust software engineering. It requires thinking beyond the "happy path" and designing for a world where networks fail, APIs time out, and processes restart.

By choosing the right framework, externalizing state, establishing clear communication channels, and building a comprehensive observability stack, you can create agents that are not just intelligent, but also dependable. These are the agents that can be trusted to automate critical business processes, manage complex systems, and deliver value reliably and at scale.

# References and Further Reading

- Building AI Agents the Right Way: Design Principles for Agentic AI (GoPubby): https://ai.gopubby.com/building-ai-agents-the-right-way-design-principles-for-agentic-ai-47d1b92f0124
- The Ultimate Guide to Building AI Agents in 2025 (Medium): https://medium.com/@divyanshbhatiajm19/the-ultimate-guide-to-building-ai-agents-in-2025-from-concept-to-deployment-121da166562e
- A practical guide to building agents (OpenAI, 2025): https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- Building effective agents (Anthropic, 2024): https://www.anthropic.com/engineering/building-effective-agents
- LangChain v1 migration guide: https://docs.langchain.com/oss/python/migrate/langchain-v1
- Building AI Agents: 14 Best Practices I Learned the Hard Way (Product Compass): https://www.productcompass.pm/p/building-ai-agents-best-practices