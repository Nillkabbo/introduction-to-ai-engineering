# Introduction to AI Engineering (for software engineers)
By Obinna Okechukwu

## What you learn

After completing this book, you will:
- Be able to implement production AI systems with confidence
- Have deep understanding of all major AI APIs
- Have a good understanding of common prompt engineering techniques
- Be able to design and deploy AI agents
- Be able to make informed architectural decisions
- Be able to work proficiently with multimodal AI applications


## How to use this book

Each chapter ends with **Key Takeaways** to review what matters and three graded **Exercises** (Warm-up, Build, Stretch) that extend the chapter's running IoT example. Each exercise states what "done" looks like, so you can check your own work. Links at the bottom of every chapter take you to the previous and next chapters.

## Book Structure

### Part 1: Foundations (Chapters 0-4)

#### [Chapter 0: Introduction](chapters/INTRODUCTION.md)

#### [Chapter 1: Welcome to AI engineering](chapters/01-ai-engineering.md)
- What are LLMs and how do they work?
- The paradigm shift: describing vs. instructing
- Your first AI call in Python (OpenAI Responses API)
- The structure of a conversation (roles: system/developer, user, assistant)
- Practical example: IoT status interpreter
- A simple mental model: tokens, embeddings, next-token prediction
- The landscape of AI models and reasoning models

#### [Chapter 2: Core Concepts - Tokens, Embeddings, and Context](chapters/02-core-concepts.md)
- Tokens and cost calculation
- Estimating tokens with tiktoken vs. exact counts from provider usage
- Embeddings and semantic search
- Context windows, statelessness, and long-context limits
- Practical example: semantic search for IoT troubleshooting
- Trimming conversation history safely

#### [Chapter 3: Setting Up Your Development Environment](chapters/03-development-environment.md)
- Python environment setup (venv or uv)
- Essential libraries: openai, python-dotenv
- API key security and keeping model names in configuration
- Project structure and virtual environments
- Building a command-line AI chatbot
- Creating requirements.txt

#### [Chapter 4: Understanding AI Capabilities and Limitations](chapters/04-ai-capabilities-limitations.md)
- Core strengths of LLMs
- Common failure modes (hallucinations, exact computation, missing real-time data)
- Safeguards: code for math (without eval), verification, grounding
- Example: SmartSafeAssistant

### Part 2: API Mastery (Chapters 5-8)

#### [Chapter 5: OpenAI API Complete Guide](chapters/05-openai-api-complete-guide.md)
- The Responses API, and how it maps to Chat Completions
- Instructions (system prompts) and conversation history
- Reasoning effort, output limits, and when temperature applies
- Streaming responses
- A complete function-calling loop (parallel calls, turn limits)
- Vision, PDFs, speech generation, and transcription

#### [Chapter 6: Anthropic Claude API Mastery](chapters/06-anthropic-claude-api-mastery.md)
- Claude API basics and reading responses by content block
- Effort, thinking, and system prompts
- Long-context and PDF analysis, prompt caching
- Vision and tool use (including parallel tool calls)
- Structured outputs
- Choosing models and providers with measurable criteria

#### [Chapter 7: Google AI (Gemini) API Guide](chapters/07-google-gemini-api-guide.md)
- The google-genai SDK (and the deprecated google-generativeai)
- System instructions, thinking levels, and chat sessions
- Video, audio, and PDF analysis with the Files API
- Automatic function calling, and when to disable it
- Building a predictive maintenance assistant

#### [Chapter 8: API Design Patterns and Best Practices](chapters/08-api-design-patterns.md)
- Error handling and retries (without stacking SDK retries)
- Rate limiting for requests and tokens
- Blocking, streaming, and Batch API responses
- Safe response caching
- Multi-provider failover with real provider calls

### Part 3: Prompt Engineering Mastery (Chapters 9-12)

#### [Chapter 9: Fundamental Prompt Engineering](chapters/09-fundamental-prompt-engineering.md)
- Anatomy of effective prompts and delimiters for untrusted input
- Zero-shot, few-shot, and chain-of-thought prompting (and how reasoning models change CoT)
- Role-based prompting
- Comparing prompts with a small, scored evaluation
- Practical IoT diagnostic system

#### [Chapter 10: Advanced Prompting Strategies](chapters/10-advanced-prompting-strategies.md)
- Self-consistency with constrained answers and agreement rates
- Tree-of-thought prompting for open-ended problems
- ReAct (Reasoning and Acting) with robust parsing
- Multi-Agent Debate, and its limits
- Reflexion with structured critique and external feedback
- Prompt chaining with validation gates

#### [Chapter 11: Structured Output Generation](chapters/11-structured-output-generation.md)
- Prompting for JSON, JSON mode, and schema-enforced Structured Outputs
- Structured Outputs with OpenAI, Anthropic, and Gemini via Pydantic
- Business-rule validation and bounded retries
- Constrained code generation, and deterministic templating
- Example: IoT configuration generator

#### [Chapter 12: Domain-Specific Prompting](chapters/12-domain-specific-prompting.md)
- Domain context vs. personas
- Technical documentation generation without invented details
- Domain-specific code generation and review
- Data analysis: compute metrics (OEE) in code, interpret with AI
- Troubleshooting guides with an embedded knowledge base

### Part 4: Building AI Agents (Chapters 13-16)

#### [Chapter 13: Introduction to AI Agents](chapters/13-introduction-to-ai-agents.md)
- Chatbots, workflows, and agents
- The Perceive-Think-Act loop
- Agent components: perception, memory, reasoning, action
- Building an autonomous IoT agent with an explicit action set

#### [Chapter 14: Tool Use and Function Calling](chapters/14-tool-use-function-calling.md)
- The tool-use loop
- Complete implementations for OpenAI, Claude, and Gemini
- Strict schemas, permissions, and human approval
- The Model Context Protocol (MCP)
- Tool results as an attack surface, and real sandboxing

#### [Chapter 15: Building Production Agents](chapters/15-building-production-agents.md)
- The agent framework landscape (provider SDKs, LangChain/LangGraph, multi-agent frameworks)
- State management and persistence
- Context engineering (compaction, just-in-time retrieval) and long-term agent memory
- Communication protocols (message brokers, MCP, A2A)
- Guardrails: budgets, idempotency, timeouts, evaluation
- Observability: logging, metrics, tracing

#### [Chapter 16: Multi-Agent Systems](chapters/16-multi-agent-systems.md)
- When multiple agents help, and when they hurt
- Multi-agent collaboration patterns (orchestrator-worker)
- Message passing and coordination
- Consensus mechanisms
- Hierarchical agent structures

### Part 5: Python Web Applications (Chapters 17-19)

#### [Chapter 17: Building AI-Powered Web Applications with Python](chapters/17-building-ai-powered-web-applications.md)
- Flask and FastAPI for AI endpoints (with async clients)
- Secure file uploads (images, audio, PDFs)
- Streaming responses and SSE
- WebSockets for real-time updates
- Example: IoT device management dashboard architecture

#### [Chapter 18: Real-Time AI Applications](chapters/18-real-time-ai-applications.md)
- Keeping LLMs off the critical path
- WebSocket integration
- Background task processing with Celery
- Bounded, expiring multi-level caching
- Rate limiting and priority queues
- Example: Real-time IoT anomaly detection

#### [Chapter 19: AI Application Architecture Patterns](chapters/19-ai-application-architecture-patterns.md)
- Monolith vs. microservices
- Event-driven architectures
- Database patterns (Postgres and pgvector first, polyglot when needed)
- Validated configuration for models and versioned prompts
- Example: Scalable IoT analytics platform

### Part 6: Production Systems (Chapters 20-25)

#### [Chapter 20: Scaling AI Applications](chapters/20-scaling-ai-applications.md)
- Provider rate limits as the first bottleneck
- Stateless servers, load balancing, and when affinity helps
- Queue-based architectures
- Database design for AI workloads
- Multi-level and semantic caching (and their risks)
- Example: Global IoT platform architecture and data residency

#### [Chapter 21: Cost Optimization](chapters/21-cost-optimization.md)
- Token, cached-token, and reasoning-token costs
- Measuring cost from provider usage reports
- Prompt caching and Batch APIs
- Reasoning effort, output control, and input trimming
- Model routing and safe semantic caching
- ROI calculation

#### [Chapter 22: Security and Safety](chapters/22-security-safety.md)
- API key management and secret storage
- Direct and indirect prompt injection, and the lethal trifecta
- Layered defences: least privilege, isolation, constrained outputs, human approval
- Data minimization, pseudonymization, and content moderation
- Audit logging and compliance (GDPR, HIPAA, EU AI Act)
- Example: Safer healthcare IoT pipeline

#### [Chapter 23: Monitoring and Observability](chapters/23-monitoring-observability.md)
- Structured logging
- Metrics and KPIs (and label cardinality)
- Distributed tracing with OpenTelemetry GenAI conventions
- Measuring quality, not just uptime
- Alerting on cost spikes, error rates, and quality proxies
- Example: IoT system health dashboard

#### [Chapter 24: Testing AI Systems](chapters/24-testing-ai-systems.md)
- Designing for testability with injected clients
- Unit and workflow tests with fake model clients
- Evals on golden datasets: exact checks and calibrated LLM judges
- Running evals in CI
- Load testing AI endpoints

#### [Chapter 25: Deployment and DevOps](chapters/25-deployment-devops.md)
- CI/CD for AI applications, with eval quality gates
- Environment management and model pinning
- Blue-green deployments
- Feature flags for models and prompts
- Rollback strategies
- Example: Metric-gated IoT firmware canary with AI-written reports

### Part 7: Advanced Topics (Chapters 26-30)

#### [Chapter 26: Fine-Tuning and Custom Models](chapters/26-fine-tuning-custom-models.md)
- When to fine-tune vs. prompt engineering or RAG
- The 2026 platform landscape (hosted, cloud, open-weight LoRA)
- Data preparation for fine-tuning
- Running and evaluating a fine-tuning job against a prompted baseline
- A cautionary example: facts belong in lookups and RAG

#### [Chapter 27: RAG (Retrieval Augmented Generation)](chapters/27-rag-retrieval-augmented-generation.md)
- RAG workflow: indexing, retrieval, grounded generation with citations
- Vector databases, metadata, and relevance thresholds
- Building a RAG-powered assistant
- Production RAG: hybrid search, reranking, chunking, access control, evaluation
- RAG vs. long context vs. fine-tuning

#### [Chapter 28: AI Workflows and Orchestration](chapters/28-ai-workflows-orchestration.md)
- From chains to workflows (DAGs)
- Building a workflow orchestrator
- Retries, failure propagation, and idempotency
- Human-in-the-loop patterns and durable execution
- Example: Automated IoT incident response

#### [Chapter 29: Emerging Patterns and Future Trends](chapters/29-emerging-patterns-future-trends.md)
- Reasoning models and test-time compute
- Agents in production, MCP, and A2A
- Multimodal and real-time AI
- Open-weight, small, on-device, and edge models
- Federated learning and its privacy safeguards
- Quantum computing, in perspective

#### [Chapter 30: Building Your AI Product](chapters/30-building-ai-product.md)
- From technology to solution: product thinking
- The Lean AI Canvas (and choosing the right AI core)
- MVPs and data flywheels
- Ethical and security review
- Launching and iterating AI products

> **Edition note:** This book was reviewed and updated in October 2026. See [docs/REVIEW-2026-10.md](docs/REVIEW-2026-10.md) for what changed and why.


## Additional Resources

### General AI Engineering & LLMs
- [GenAI Handbook](https://genai-handbook.github.io/) – A living, textbook-style roadmap for learning modern AI, LLMs, and generative models, with curated links to the best blogs, videos, and courses.
- [The 2025 AI Engineering Reading List (Latent.Space)](https://www.latent.space/p/2025-papers) – A practical, annotated list of 50+ must-read papers, blogs, and models across LLMs, prompting, RAG, agents, codegen, vision, and more.
- [Full End-to-End Pipeline for LLM Apps (Rohan's Bytes)](https://www.rohan-paul.com/p/full-end-to-end-pipeline-for-developing) – A 2024–2025 guide to building, deploying, and monitoring LLM-powered applications, with technical best practices and industry case studies.

### Prompt Engineering
- [Prompt Engineering Guide](https://www.promptingguide.ai/) – Comprehensive guide to prompting techniques, from basics to advanced strategies like chain-of-thought, tree-of-thought, and multi-agent systems.
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) – Official best practices and examples for crafting effective prompts.
- [Prompt Engineering Mastery: The Complete Guide](https://www.aifire.co/p/the-ultimate-guide-for-mastering-ai-engineering-in-2024) – A step-by-step roadmap for mastering prompt engineering and LLMs, with practical resources and project ideas.
- [Claude Prompt Engineering Guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) – Anthropic’s guide to prompt design for Claude models.

### AI Agents & Autonomous Systems
- [Awesome AI Agents (GitHub)](https://github.com/jim-schwoebel/awesome_ai_agents) – A massive, regularly updated list of 1,500+ resources, tools, frameworks, datasets, and courses for building and learning about AI agents.
- [A Survey on LLM-based Autonomous Agents (arXiv)](https://arxiv.org/abs/2309.07864) – Comprehensive academic survey of agent architectures, tool use, evaluation, and future directions.
- [LangChain Documentation](https://python.langchain.com/docs/) – The most popular open-source framework for building LLM-powered agents, tool use, and RAG systems.
- [LlamaIndex Documentation](https://docs.llamaindex.ai/) – Framework for building data-augmented LLM applications and agentic workflows.

### Agent Standards & Security
- [Model Context Protocol](https://modelcontextprotocol.io) – The open standard for connecting AI applications to tools and data.
- [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) – The most critical security risks for LLM-based applications.
- [Building Effective Agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents) – Practical patterns for workflows and agents.

### RAG (Retrieval-Augmented Generation)
- [Deconstructing RAG (LangChain Blog)](https://blog.langchain.dev/deconstructing-rag/) – Practical guide to RAG architectures, vector databases, and best practices.
- [RAG Course (DeepLearning.AI)](https://learn.deeplearning.ai/courses/advanced-rag) – Free video course on advanced RAG techniques and evaluation.

### Benchmarks, Datasets, and Evaluation
- [HELM: Holistic Evaluation of Language Models (Stanford)](https://crfm.stanford.edu/helm/latest/) – A living benchmark for LLMs, covering knowledge, reasoning, safety, and more.
- [Hugging Face Datasets](https://huggingface.co/datasets) – Thousands of open datasets for LLM training, fine-tuning, and evaluation.

### Community & News
- [Latent.Space Newsletter](https://www.latent.space/) – Weekly deep dives and news for AI engineers.
- [AI Fire Academy](https://www.aifire.co/) – Practical guides, workflows, and a community for mastering AI engineering.
---
