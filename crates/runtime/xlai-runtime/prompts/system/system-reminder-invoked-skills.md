The following skills were invoked in this session. Continue to follow their guidance:

{% for skill in skills %}
- `{{ skill.name }}`: {{ skill.description }}
{% endfor %}
