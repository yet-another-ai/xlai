You may answer directly or request tool execution. Respond with exactly one valid JSON value matching the required schema. If you need tools, set `assistant_response` to null and fill `tool_calls`. If you can answer directly, set `tool_calls` to [] and put the user-facing reply in `assistant_response`. Do not add markdown fences, prose, or any text outside the JSON.

Available tools:
{{ tool_specs }}
