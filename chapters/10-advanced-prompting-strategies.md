# Chapter 10: Advanced Prompting Strategies

In the last chapter, we covered the fundamentals of prompt engineering: be specific, provide context, and use patterns like few-shot examples and explicit reasoning steps. These techniques are the bedrock of effective AI interaction and will serve you well for most use cases.

But what happens when a problem is so complex, ambiguous, or high-stakes that a single prompt isn't enough? This is where advanced strategies come in. They are not different ways of phrasing a question; they are **multi-call workflows** that structure how the model explores, checks, and refines its answers. This chapter introduces the most important ones, explains what the research actually shows about each, and shows how to implement them.

> **Reasoning models changed the cost-benefit math.** Many of these techniques were invented in 2022–2023 to squeeze better reasoning out of models that answered in a single pass. Today's reasoning models already explore and check internally, controlled by a reasoning-effort setting. Before building a multi-call workflow, try the simplest option first: one call to a capable model at higher reasoning effort. Use the strategies below when you need what a single call can't give you: measurable agreement, external information, independent perspectives, or verification against an outside signal. Every strategy multiplies your cost and latency, so measure whether it actually improves results on your task (Chapter 24).

### Learning Objectives

By the end of this chapter, you will be able to:

-   Implement **Self-Consistency** to make answers to well-defined questions more reliable.
-   Use **Tree-of-Thought (ToT)**-style prompting to explore multiple approaches to an open problem.
-   Apply the **ReAct (Reasoning and Acting)** pattern to gather information step by step.
-   Use **Multi-Agent Debate** to surface competing perspectives, and know its limits.
-   Implement **Reflexion**-style self-improvement loops, and understand why external feedback matters.
-   Build **Prompt Chains** that break a complex task into a sequence of focused, verifiable steps.

## The Limits of Simple Prompts

Consider this complex IoT system failure:

```python
complex_problem = """
A smart factory's assembly line has stopped. Key symptoms include:
- Conveyor belt motors in Zone C are offline.
- Vibration sensors in the adjacent Zone B are showing erratic, high-amplitude readings.
- Network latency for all devices in Zone C has increased by 500ms.
- The issue began an hour after the morning shift started.
- No new firmware has been deployed recently.
"""
```

A single response might latch onto one symptom (say, the network latency) and miss the explanation that connects all of them. The strategies below give the model structured ways to explore, gather facts, and check its work.

We'll use one small helper throughout:

```python
import openai

client = openai.OpenAI()
MODEL = "gpt-6-luna"

def ask(prompt: str, instructions: str | None = None, effort: str = "medium") -> str:
    response = client.responses.create(
        model=MODEL, instructions=instructions, input=prompt, reasoning={"effort": effort}
    )
    return response.output_text.strip()
```

## Strategy 1: Self-Consistency

**The idea:** Ask the same question several times and take the most common answer. Because the model's outputs vary from run to run, independent attempts make different mistakes, and the answer that most attempts agree on is more likely to be right. The original research (Wang et al., 2022) showed large gains on arithmetic and logic benchmarks, where each attempt reasons its own way to a short, checkable final answer.

The detail that makes or breaks this technique: **you can only vote on answers you can compare.** Two free-text diagnoses are almost never identical strings, so counting them finds no consensus. Constrain the final answer to a fixed set of labels (or a number), and vote on that.

```python
from collections import Counter

ROOT_CAUSES = [
    "sensor_fault",
    "installation_near_heat_source",
    "calibration_error",
    "firmware_bug",
    "power_supply_issue",
]

def self_consistency_diagnosis(problem: str, num_runs: int = 5) -> tuple[str, float]:
    """Samples several diagnoses and returns the majority label with its agreement rate."""
    prompt = f"""
Diagnose the most likely root cause of this IoT issue.
Think it through, then end your reply with a final line of the form:
ROOT_CAUSE: <one of {ROOT_CAUSES}>

Problem: {problem}
"""
    votes = []
    for _ in range(num_runs):
        reply = ask(prompt)
        final_line = reply.strip().splitlines()[-1]
        label = final_line.removeprefix("ROOT_CAUSE:").strip()
        if label in ROOT_CAUSES:       # discard malformed answers instead of guessing
            votes.append(label)

    if not votes:
        return "no_valid_answer", 0.0
    label, count = Counter(votes).most_common(1)[0]
    return label, count / num_runs

device_issue = "A smart thermostat reads 10°F higher than the actual room temperature. It was installed last week and has the latest firmware."
label, agreement = self_consistency_diagnosis(device_issue)
print(f"Consensus: {label} ({agreement:.0%} of runs agreed)")
```

The agreement rate is a useful by-product: if only 2 of 5 runs agree, the question is genuinely ambiguous, and that's a good signal to ask for more information or escalate to a human. (Chapter 11 shows how to enforce the label format with structured outputs instead of parsing the last line.)

**When to use it:** Questions with a single correct, comparable answer (a category, a number, yes/no) where an error is costly. It multiplies cost by the number of runs.

## Strategy 2: Tree-of-Thought (ToT) Prompting

**The idea:** Complex problems are rarely solved in a straight line. Experts consider several approaches, develop the promising ones, and abandon dead ends. The original Tree-of-Thoughts work (Yao et al., 2023) implemented this as a *search*: generate candidate "thoughts," score them, expand the best ones, and backtrack when a branch fails.

Full tree search is expensive and rarely needed in applications. A practical, simplified version (breadth-first with a single level) captures most of the benefit for open-ended diagnosis and planning:

1.  **Generate:** Brainstorm several distinct approaches.
2.  **Develop:** Elaborate each approach independently.
3.  **Evaluate and synthesize:** Compare the developed approaches critically and combine the best elements.

```python
import re

class TreeOfThoughtSolver:
    def solve(self, problem: str, num_paths: int = 3) -> str:
        approaches = self._generate_approaches(problem, num_paths)
        print("Approaches:", *approaches, sep="\n- ")

        developed = [self._develop(problem, approach) for approach in approaches]
        return self._evaluate_and_synthesize(problem, developed)

    def _generate_approaches(self, problem: str, n: int) -> list[str]:
        reply = ask(f"""
Problem: {problem}
List {n} distinct, high-level approaches to diagnosing this problem, one per line, no numbering.
""", effort="low")
        lines = [re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip() for line in reply.splitlines()]
        return [line for line in lines if line][:n]   # strip list markers; keep exactly n

    def _develop(self, problem: str, approach: str) -> str:
        return ask(f"""
Problem: {problem}
Approach: {approach}
Develop this approach: the specific investigation steps, what evidence would confirm or rule it out, and its main weakness.
""")

    def _evaluate_and_synthesize(self, problem: str, developed: list[str]) -> str:
        paths = "\n\n".join(f"<path id='{i + 1}'>\n{p}\n</path>" for i, p in enumerate(developed))
        return ask(f"""
Problem: {problem}

Here are several independently developed diagnostic approaches:
{paths}

Act as a critical reviewer. Which approach best explains ALL the symptoms? Point out where the others fall short,
then produce a single prioritized action plan that keeps the strongest elements.
""", effort="high")

solution = TreeOfThoughtSolver().solve(complex_problem)
print("\n--- Final Plan ---\n", solution)
```

Developing each approach in a *separate* call matters: each branch is explored without being anchored by the others, which is where the diversity comes from.

**When to use it:** Open-ended problems with several plausible directions (diagnosis, design, planning) where premature commitment is the main risk.

## Strategy 3: ReAct (Reasoning and Acting)

**The idea:** Many problems can't be solved with the information in the prompt. ReAct (Yao et al., 2022) interleaves **reasoning** ("what do I need to know next?") with **actions** (calling a tool) and **observations** (the tool's result), in a loop, until the model has enough to answer.

ReAct is the conceptual ancestor of today's agents. In production you implement it with the providers' native **tool calling** (Chapters 5, 6 and 14), which handles the action format for you. The text-based version below is still worth building once, because it shows exactly what the loop does.

```python
class ReActAgent:
    ACTION_PATTERN = re.compile(r"^Action:\s*(\w+)", re.MULTILINE)
    FINAL_PATTERN = re.compile(r"^Final Answer:\s*(.+)", re.MULTILINE | re.DOTALL)

    def __init__(self, tools: dict):
        self.tools = tools

    def run(self, problem: str, max_steps: int = 6) -> str:
        transcript = ""
        tool_list = "\n".join(f"- {name}: {fn.__doc__}" for name, fn in self.tools.items())

        for step in range(1, max_steps + 1):
            reply = ask(f"""
You are an IoT diagnostic agent. Solve the problem by alternating Thought and Action steps.

Problem: {problem}

Available tools (they take no arguments):
{tool_list}

Respond with EITHER
Thought: <your reasoning>
Action: <one tool name>
OR, once you have enough evidence,
Thought: <your reasoning>
Final Answer: <the root cause and recommended action>

History so far:
{transcript or "(none)"}
""")
            print(f"\n--- Step {step} ---\n{reply}")

            final = self.FINAL_PATTERN.search(reply)
            if final:
                return final.group(1).strip()

            action = self.ACTION_PATTERN.search(reply)
            if action is None:
                observation = "Invalid format. Reply with 'Action: <tool>' or 'Final Answer: ...'."
            elif action.group(1) not in self.tools:
                observation = f"Unknown tool '{action.group(1)}'. Choose from: {', '.join(self.tools)}."
            else:
                observation = self.tools[action.group(1)]()
            print(f"Observation: {observation}")
            transcript += f"{reply}\nObservation: {observation}\n"

        return "Stopped: no final answer within the step limit."

# --- Dummy tools for the demo ---
def check_network_logs():
    """Network latency and connectivity events for Zone C."""
    return "High latency in Zone C starting at 09:05 AM; switch SW-C1 logged repeated power-cycle events."

def check_motor_status():
    """Current status of the Zone C conveyor motors."""
    return "All Zone C motors report 'offline'. Last heartbeat 09:02 AM. Motor controllers show undervoltage faults."

def review_maintenance_log():
    """Recent maintenance and facility events."""
    return "No maintenance scheduled. Facilities log: power grid switchover to backup feed at 09:02 AM."

agent = ReActAgent({
    "check_network_logs": check_network_logs,
    "check_motor_status": check_motor_status,
    "review_maintenance_log": review_maintenance_log,
})
print("\nResult:", agent.run(complex_problem))
```

The robustness details are what turn a fragile demo into a working loop: the format is parsed with patterns rather than `split()`, format errors and unknown tools are fed back as observations so the model can correct itself, there is an explicit **finish** action, and the loop has a hard step limit.

**When to use it:** Problems where the model must gather information incrementally. This is the foundation of autonomous agents (Part 4).

## Strategy 4: Multi-Agent Debate

**The idea:** Several "agents" (here, the same model with different role prompts) analyze a problem, read each other's arguments, and revise their positions over a few rounds; a final step synthesizes the result. The original work (Du et al., 2023) reported gains in factuality and reasoning.

Keep your expectations calibrated. Later studies found that, for the same total compute, debate often does no better than simpler methods such as self-consistency, and agents built on the same model share the same blind spots, so debate does not remove bias, it only surfaces different framings. Where debate shines is **decision support**: making trade-offs explicit (opportunity vs. risk vs. feasibility) so a human can see the strongest case for each side.

```python
class MultiAgentDebate:
    AGENTS = {
        "optimist": "an Opportunity Analyst. Focus on benefits, upside, and best-case outcomes.",
        "skeptic": "a Risk Analyst. Focus on risks, failure modes, hidden costs, and worst-case outcomes.",
        "pragmatist": "a Practical Engineer. Focus on feasibility, resources, timelines, and real-world constraints.",
    }

    def run_debate(self, problem: str, rounds: int = 2) -> str:
        views = {
            name: ask(f"Problem: {problem}\n\nGive your initial analysis and recommendation.",
                      instructions=f"You are {role}")
            for name, role in self.AGENTS.items()
        }

        for round_number in range(1, rounds + 1):
            previous = dict(views)   # everyone responds to the SAME previous round
            for name, role in self.AGENTS.items():
                others = "\n\n".join(f"<{other}>\n{view}\n</{other}>"
                                     for other, view in previous.items() if other != name)
                views[name] = ask(f"""
Problem: {problem}

Your previous position:
{previous[name]}

The other analysts said:
{others}

Respond to their strongest points. Concede where they are right, rebut where they are wrong,
and state your updated recommendation.
""", instructions=f"You are {role}")
            print(f"Round {round_number} complete.")

        positions = "\n\n".join(f"<{name}>\n{view}\n</{name}>" for name, view in views.items())
        return ask(f"""
Problem: {problem}

Final positions after a structured debate:
{positions}

Write a balanced recommendation: the key benefits, the key risks and how to mitigate them,
the practical constraints, and a clear decision with the conditions under which it should be revisited.
""", effort="high")

decision = "Should we implement an AI-powered predictive maintenance system in our factory? Consider costs, benefits, risks, and implementation challenges."
print(MultiAgentDebate().run_debate(decision))
```

Note `previous = dict(views)`: without that snapshot, later agents in a round would see earlier agents' *new* positions while earlier agents saw only old ones, making the debate order-dependent.

**When to use it:** Decisions with genuine trade-offs, where the value is in making competing considerations explicit. Avoid it for questions with a single correct answer, where self-consistency is cheaper and at least as effective.

## Strategy 5: Reflexion

**The idea:** Generate a solution, critique it, and use the critique to produce a better one. Reflexion (Shinn et al., 2023) formalized this for agents, and its key ingredient is easy to overlook: the reflections were driven by **external feedback**, such as failing unit tests or a task environment reporting failure. Self-critique without any outside signal is much weaker; research has shown that models often cannot reliably find their own reasoning errors, and sometimes "fix" correct answers into wrong ones (Huang et al., 2023).

So the most effective version of this loop uses a real check wherever you have one: run the generated code's tests, validate the JSON against a schema, check the numbers with code, or compare against a reference document. Where no external check exists, a structured critique against explicit criteria is still useful for completeness and clarity, as long as you don't treat it as proof of correctness.

```python
import json

class ReflexionAgent:
    def solve(self, problem: str, criteria: list[str], max_attempts: int = 3) -> str:
        reflections: list[str] = []        # local to this call, so nothing leaks between problems
        solution = ""

        for attempt in range(1, max_attempts + 1):
            lessons = "\n".join(f"- {r}" for r in reflections) or "(none yet)"
            solution = ask(f"""
Problem: {problem}

Lessons from previous attempts:
{lessons}

Write your best solution, addressing every lesson above.
""")

            verdict = self._critique(problem, solution, criteria)
            print(f"Attempt {attempt}: passes={verdict['passes']} issues={verdict['issues']}")
            if verdict["passes"]:
                return solution
            reflections.extend(verdict["issues"])

        return solution   # best effort after max_attempts; flag for human review in a real system

    def _critique(self, problem: str, solution: str, criteria: list[str]) -> dict:
        reply = client.responses.create(
            model=MODEL,
            input=f"""
Evaluate the solution strictly against each criterion.

Problem: {problem}
Criteria: {json.dumps(criteria)}
<solution>
{solution}
</solution>

Return JSON: {{"passes": true or false, "issues": ["specific, actionable problem", ...]}}.
"passes" is true only if every criterion is fully met.
""",
            text={"format": {"type": "json_object"}},
        ).output_text
        verdict = json.loads(reply)
        return {"passes": bool(verdict.get("passes")), "issues": list(verdict.get("issues", []))}

criteria = [
    "Covers device authentication and credential rotation",
    "Covers network segmentation for IoT devices",
    "Addresses data privacy for residents, including retention limits",
    "Includes an incident response process",
    "Gives a rough cost tier (low/medium/high) for each measure",
]
plan = ReflexionAgent().solve(
    "Design an IoT security strategy for a mid-sized smart city that balances functionality, privacy, and cost.",
    criteria,
)
print(plan)
```

Compared with a keyword check ("does the critique mention 'missing'?"), a structured verdict against explicit criteria gives a clear stop condition and concrete, actionable feedback for the next attempt.

**When to use it:** Tasks with checkable quality criteria, and above all tasks where you can plug in a real external check (tests, validators, reference data).

## Strategy 6: Prompt Chaining

**The idea:** Instead of one prompt that tries to do everything, split the task into a sequence of focused steps, where each step's output becomes the next step's input. This is the most widely used advanced technique in production, because it makes systems **more reliable, easier to debug, and cheaper** at the same time:

-   Each prompt has one job, so it is easier to get right and to test.
-   You can **validate between steps** in ordinary code and stop early when something is wrong.
-   You can use a small, cheap model for simple steps and a strong model only where needed.

Let's turn a raw maintenance log into a customer-ready incident notice in three steps, with a code check between them.

```python
def extract_incident(raw_log: str) -> dict:
    """Step 1: extraction. Structured output we can validate."""
    reply = client.responses.create(
        model=MODEL,
        reasoning={"effort": "low"},
        input=f"""
Extract the incident from this log as JSON with keys:
device_id (string), start_time (ISO 8601), impact (string), customer_facing (boolean).

<log>
{raw_log}
</log>
""",
        text={"format": {"type": "json_object"}},
    ).output_text
    return json.loads(reply)

def validate_incident(incident: dict) -> list[str]:
    """Gate: deterministic checks in plain code, not in a prompt."""
    problems = [f"missing {key}" for key in ("device_id", "start_time", "impact", "customer_facing")
                if key not in incident]
    if not incident.get("customer_facing", False):
        problems.append("not customer-facing: no notice needed")
    return problems

def draft_notice(incident: dict) -> str:
    """Step 2: generation, grounded only in the validated fields."""
    return ask(f"""
Write a short, calm service notice for customers based ONLY on these facts:
{json.dumps(incident)}
Do not speculate about causes. Include the start time and what customers may notice.
""")

def review_notice(notice: str, incident: dict) -> str:
    """Step 3: a focused review pass."""
    return ask(f"""
Check this customer notice against the facts. Fix any statement not supported by the facts,
remove internal jargon, and return only the corrected notice.

Facts: {json.dumps(incident)}
<notice>
{notice}
</notice>
""")

raw_log = "2026-10-12T07:41:03Z GW-17 uplink lost; 340 smart meters in district 4 not reporting; billing dashboards show stale data."
incident = extract_incident(raw_log)
problems = validate_incident(incident)
if problems:
    print("Chain stopped:", problems)
else:
    print(review_notice(draft_notice(incident), incident))
```

Notice what the chain buys you: if the extraction misses a field, the chain stops with a clear reason instead of producing a confident but wrong notice; and each step can be tested on its own (Chapter 24).

**When to use it:** Almost any multi-part task with a predictable sequence of steps. If you can draw the steps as a flowchart, prefer a chain over one giant prompt, and over an autonomous agent (Chapter 13).

## Conclusion

You have moved beyond single prompts to multi-call reasoning workflows. Each comes with a cost multiplier, so reach for them when they solve a specific problem:

-   **Self-Consistency:** Majority voting over comparable answers; the agreement rate also measures uncertainty.
-   **Tree-of-Thought:** Explore distinct approaches independently, then evaluate and combine.
-   **ReAct:** Interleave reasoning with tool use to gather information; the basis of agents.
-   **Multi-Agent Debate:** Make trade-offs explicit for decisions; not a substitute for verification.
-   **Reflexion:** Iterate with critique, ideally driven by an external check.
-   **Prompt Chaining:** Focused steps with validation in between; the workhorse of production systems.

These strategies combine well: a chain can include a self-consistency vote at its riskiest step, or a ReAct loop to gather facts before a Tree-of-Thought planning step. Combined with measurement, they are how you engineer with AI rather than just prompting it.

## Key Takeaways

-   Try one call to a capable model at higher reasoning effort before building any multi-call workflow; every strategy here multiplies cost and latency, so measure that it actually helps on your task.
-   Self-consistency only works on answers you can compare: force the final answer into a fixed label set or a number, discard malformed votes, and treat a low agreement rate as a signal to escalate.
-   In Tree-of-Thought and debate, develop each branch or agent in a separate call (and snapshot the previous round) so the paths are not anchored on each other; debate surfaces trade-offs but does not remove the shared blind spots of one model.
-   A robust ReAct loop parses actions with patterns, feeds format errors and unknown tools back as observations, has an explicit finish action, and stops at a hard step limit. In production, use native tool calling instead of text parsing.
-   Reflexion is strongest with an external signal (tests, a schema, a reference document); pure self-critique can turn correct answers into wrong ones, so use a structured verdict against explicit criteria and a bounded number of attempts.
-   Prompt chaining is the production workhorse: one job per step, deterministic validation in plain code between steps, and an early stop with a clear reason when a check fails.

## Exercises

**1. Warm-up:** Run `self_consistency_diagnosis` with `num_runs=7` on the thermostat issue and on a deliberately vague issue of your own ("the sensor is acting weird"). Also count how many replies were discarded as malformed. You are done when you can show the agreement rate and the malformed count for both issues, and the vague issue has the lower agreement.

**2. Build:** Give `ReflexionAgent` a real external check. Ask it to write a Python function `adc_to_celsius(raw: int) -> float` for a sensor whose 12-bit reading maps linearly from 0 to 4095 onto -40 to 125 degrees Celsius, then replace `_critique` with code that runs the generated function against five `assert` cases (including both ends of the range and an out-of-range value that must raise `ValueError`) and returns the failing cases as the issues. Run the generated code in a separate subprocess with a timeout, not with `exec()` in your own process. You are done when the loop either passes all five asserts or stops after `max_attempts`, and prints the attempt number and the failing asserts for every attempt.

**3. Stretch:** Measure whether self-consistency is worth its cost. Write 10 device issues with a known correct label from `ROOT_CAUSES`, then compare three setups: a single call at `effort="medium"`, a single call at `effort="high"`, and self-consistency with 5 runs at `effort="medium"`. Record accuracy, number of API calls, and total wall-clock time for each. You are done when you have a small table of all three numbers per setup and a one-sentence recommendation backed by it.

# References and Further Reading

- Self-Consistency Improves Chain of Thought Reasoning in Language Models (Wang et al., 2022). https://arxiv.org/abs/2203.11171
- Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023). https://arxiv.org/abs/2305.10601
- ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022). https://arxiv.org/abs/2210.03629
- Improving Factuality and Reasoning in Language Models through Multiagent Debate (Du et al., 2023). https://arxiv.org/abs/2305.14325
- Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023). https://arxiv.org/abs/2303.11366
- Large Language Models Cannot Self-Correct Reasoning Yet (Huang et al., 2023). https://arxiv.org/abs/2310.01798
- Building effective agents: prompt chaining and other workflow patterns (Anthropic, 2024). https://www.anthropic.com/engineering/building-effective-agents
