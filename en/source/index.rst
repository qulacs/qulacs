Qulacs Documentation
======================================

.. meta::
   :description lang=en: qulacs documentation

Qulacs is a fast quantum circuit simulator for simulating large, noisy, or parametric quantum circuits.
Implemented in C/C++ and with python interface, Qulacs achieved both high speed circuit simulation and high usability.

Qulacs was developed in Prof. `Fujii's group <http://quantphys.org/wp/qinfp/>`_. Maintained and developing new features by `QunaSys <http://www.qunasys.com/>`_.

:doc:`intro/0_about`.


`Github repository
<https://github.com/qulacs/qulacs>`_ |
`Slack Community <https://join.slack.com/t/qulacs/shared_invite/enQtNzY1OTM5MDYxMjAxLWM1ZDc3MzdiNjZhZjdmYTQ5MTJiOTEzZjI3ZjAwZTg0OGFiNjcxY2VjZWRjMWY0YjE5ZTViOWQzZTliYzdmYzY>`_ | 
`Study Material (Japanese) <https://dojo.qulacs.org/>`_ 



Get Started
-----------
::
  
   pip install qulacs

Detailed instructions about installation in :doc:`intro/1_install`.
Having trouble starting? see :doc:`intro/2_faq`.
Basic usage of qulacs in Python and C++: :doc:`intro/3_usage`.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Get Started

   intro/0_about
   intro/1_install
   intro/2_faq
   intro/3_usage

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   
   intro/4.1_python_tutorial
   intro/4.2_cpp_tutorial

.. toctree::
   :maxdepth: 2
   :caption: User Manual
   
   guide/2.1_states
   guide/2.2_gates
   guide/2.3_circuits
   guide/2.4_parametric


.. toctree::
   :maxdepth: 2
   :caption: API reference
   :hidden:

   pyRef/modules
   api/cpp_library_root
   


.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Contributing

   write/0_readme
   write/1_contents


