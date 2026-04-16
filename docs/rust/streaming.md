# Streaming

The runtime supports streamed chat output through **`ChatChunk`** and **`ChatExecutionEvent`**.

## Chat streams

Each `stream` / `stream_prompt` / `stream_*` call on **`Chat`** performs **one** model run. Events are deltas plus a final `ChatChunk::Finished` for that turn.

## Agent streams

Streaming on **`Agent`** uses the same chunk types, but the stream may include **multiple** model rounds while tools are requested. Between rounds you may see `ChatExecutionEvent::ToolCall` and `ToolResult` events after a finished assistant message that contained tool calls.

## JavaScript note

`streamPrompt` / `streamPromptWithContent` collect the full event list in order (one round-trip through the WASM bridge); a browser `ReadableStream` API is not wired yet.

## Event coverage

Streaming currently includes message start events, content deltas, tool call deltas, and final response events (per model turn; agent streams may emit several before the stream ends).

See the [README “Streaming” section](https://github.com/yetanother.ai/xlai/blob/main/README.md#streaming) for additional design notes.
