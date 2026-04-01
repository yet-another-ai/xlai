The following skills are available in this session. Use them when they match the user's request:

{% for skill in skills %}
- `{{ skill.name }}`: {{ skill.description }}
{% endfor %}
