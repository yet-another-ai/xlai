Respond with exactly one valid JSON value that matches the provided JSON Schema. Do not add markdown fences, prose, or any text before or after the JSON.

JSON Schema:
{{ schema }}
{% if name %}

Schema name: {{ name }}
{% endif %}
{% if description %}

Schema description: {{ description }}
{% endif %}
