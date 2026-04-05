Respond with output that matches the provided Lark grammar exactly. Do not add markdown fences, prose, or any text outside the grammar-constrained output.

Lark Grammar:
{{ grammar }}
{% if name %}

Schema name: {{ name }}
{% endif %}
{% if description %}

Schema description: {{ description }}
{% endif %}
