Read an additional declared file from a previously resolved skill package.

Use this tool when a skill exposes optional resources and you need one of those files to complete the task.

How to invoke:
- Provide the skill identifier in `skill`
- Provide the declared relative resource path in `path`

Important:
- Only declared in-package skill resources may be loaded
- Use this tool after the relevant skill has been resolved
- Do not guess file paths outside the skill package
