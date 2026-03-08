# Subagent Architecture — Design Review

## Overview

ollama-cli currently runs a single chat loop: one model, one conversation, sequential tool execution. This proposal adds multi-agent support so the orchestrator can break down complex tasks, delegate to subagents running in parallel, and manage context efficiently.

## Problem

1. Complex tasks require multiple steps (search, read, analyze, write) that block each other sequentially
2. Different subtasks benefit from different models (coding vs. general reasoning)
3. Tool results are verbose and fill the context window quickly — a single web search can consume thousands of tokens

## Architecture

Three components:

```
Orchestrator                    Subagent                     Mailbox File
    |                              |                              |
    |-- spawn(task, model, tools)->|                              |
    |                              |-- {step: "think", ...} ---->|
    |                              |-- {step: "act", ...} ------>|
    |                              |-- {step: "observe", ...} -->|
    |                              |-- {step: "think", ...} ---->|
    |                              |-- {step: "act", ...} ------>|
    |                              |-- {step: "observe", ...} -->|
    |                              |                              |
    |                              |  (self-summarize)            |
    |                              |-- {step: "done",             |
    |                              |    summary: "...",           |
    |                              |    status: "success"} ----->|
    |                              |                              |
    |<--- read ONLY summary -------+------------------------------|
    |                              |                              |
    |  (summary enters context)    |  (full trace stays on disk)  |
```

### Orchestrator (main process)

- Talks to the human
- Breaks down tasks into subtasks
- Decides which model each subagent should use (in auto mode)
- Writes prompts for subagents, hints which tools are relevant
- Reads mailbox summaries — never the full execution trace
- Can stop or forget subagents

### Subagent (subprocess)

- Runs a ReACT (Reason + Act) state machine in a separate process
- Each step is logged to its mailbox file in structured format
- Has access to a scoped set of tools chosen by the orchestrator
- On completion, self-summarizes its results before exiting
- Checks for stop signals between each ReACT step

```
THINK -> ACT -> OBSERVE -> THINK -> ... -> DONE
  |                                          |
  +-------- check stop signal <--------------+
```

### Mailbox (file-based, not a process)

- Directory: `~/.ollama-cli/mailbox/`
- One JSONL file per agent: `agent-001.jsonl`
- Each line is a structured step:

```json
{"step": "think", "content": "I need to search for weather data", "timestamp": 1709827200}
{"step": "act", "tool": "web_search", "params": {"query": "weather Cork"}, "timestamp": 1709827201}
{"step": "observe", "content": "Found temperature 9C, cloudy...", "timestamp": 1709827205}
{"step": "done", "summary": "Cork is 9C, cloudy with light rain.", "status": "success", "timestamp": 1709827210}
```

- Stop signal: orchestrator writes `{"signal": "stop"}` to the agent's file
- Subagent checks for stop signal before each step

## Orchestrator Actions

| Action | When | Effect |
|--------|------|--------|
| Read summary | Subagent done | Summary enters orchestrator context |
| Stop | Taking too long / wrong direction | Write `{signal: "stop"}`, subagent self-summarizes what it has so far, then exits |
| Forget | Result not needed anymore | Don't read summary at all, just delete/archive the mailbox file |
| Peek | Debugging / user asks | Read full trace without loading into orchestrator context, show to user directly |

## Context Management

The core design principle: **the orchestrator only ever sees summaries, never raw execution traces.**

- Subagent runs web search, gets 3000 chars of page content, reasons over it in 5 ReACT steps — that's ~10k tokens of trace
- Orchestrator receives: `"Cork is 9C, cloudy with light rain. 5-day forecast shows highs of 9-10C."` — ~30 tokens
- Full trace remains in the mailbox JSONL for debugging or user inspection via Peek

This keeps the orchestrator's context window clean and allows it to coordinate many subagents without overflow.

## Constraints

- **Ollama is single-GPU**: requests serialize on one GPU. Parallel subagents are most useful when they use different models or when one is waiting on I/O (web fetch, file read) while another does inference.
- **Subagent model selection**: orchestrator picks the model. Coding tasks get `qwen2.5-coder:14b`, general tasks get `mistral:latest`, etc. In manual model mode, all subagents use the user's selected model.

## Implementation Plan

1. **Mailbox helpers** — read/write/signal functions for the JSONL mailbox directory
2. **Subagent ReACT loop** — single subprocess, state machine, writes to mailbox, checks stop signals
3. **Orchestrator integration** — spawn one subagent from the main chat loop, read its summary
4. **Multi-agent** — spawn multiple subagents, coordinate results, present to user
