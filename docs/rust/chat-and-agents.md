# Chat, agents, and tools

## Chat vs agent

- **`Chat`** — one model call per `prompt` / `execute` / `stream` / `stream_*`. Tool callbacks are **not** run automatically; you can inspect `tool_calls` and drive follow-up turns yourself.
- **`Agent`**
  - **Unary** `prompt` / `execute` / `prompt_parts` / `prompt_content`: exactly **one** model call; registered tools are **not** run by the runtime.
  - **Streaming** `stream` / `stream_prompt` / …: runs the automatic **tool loop** until the model returns without tool calls or `with_max_tool_round_trips` (default **8**) is exceeded.

## Tool registration

- Tools are registered **per** chat or agent session.
- Tool calls are passed through the runtime request to the model.
- When tools run on **`Agent`**, local session tools are preferred before a runtime-level tool executor.
- Each tool’s `ToolDefinition::execution_mode` controls concurrency with other tools in the same model turn. If any invoked tool in a turn is **`Sequential`**, all tool calls in that turn run **sequentially** in model order.

## Context compression (agent streaming)

On **`Agent`**, `with_context_compressor` (Rust) registers an async closure that runs **once per streaming tool-loop round**, immediately before each model call. It receives the accumulated `ChatMessage` list and an optional best-effort input-token estimate.

The hook is **not** used for unary `prompt` / `execute`.

In **JavaScript**, use `AgentSession.registerContextCompressor` before `streamPrompt` / `streamPromptWithContent`.

For full tables and API notes, see the [README “Tool Calling” section](https://github.com/yetanother.ai/xlai/blob/main/README.md#tool-calling).
