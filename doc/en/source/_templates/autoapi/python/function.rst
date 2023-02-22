{% if obj.display %}
.. py:function:: {{ obj.short_name }}({{ obj.args | replace("qulacs_core", "qulacs") }}){% if obj.return_annotation is not none %} -> {{ obj.return_annotation | replace("qulacs_core", "qulacs") }}{% endif %}

{% for (args, return_annotation) in obj.overloads %}
              {{ obj.short_name }}({{ args | replace("qulacs_core", "qulacs") }}){% if return_annotation is not none %} -> {{ return_annotation | replace("qulacs_core", "qulacs")}}{% endif %}

{% endfor %}
   {% for property in obj.properties %}
   :{{ property }}:
   {% endfor %}

   {% if obj.docstring %}
   {{ obj.docstring|indent(3) }}
   {% endif %}
{% endif %}
