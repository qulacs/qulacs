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



.. * **Tutorials**: 
   :doc:`intro/3_python_tutorial`
   :doc:`intro/4_cpp_tutorial`.


.. toctree::
   :maxdepth: 1
   :caption: Tutorial
   
   intro/4.1_python_tutorial
   intro/4.2_cpp_tutorial

.. Advanced features
   ----------------------------------
   Brief description of the Advanced feature
   * **User Manual**: :doc:`guide/0_root`.
   * **Advanced Tutorials**: :doc:`guide/2_python`. :doc:`guide/3_cpp`.
   * **Qulacs Cookbooks**: :doc:`apply/0_root`.
   * **C++ API reference**: :doc:`api/cpp_library_root`.


.. comment
.. toctree::
   :maxdepth: 2
   :caption: Guides
   guide/0_root
   guide/1_concepts
   guide/2_python
   guide/3_cpp
   
..   guide/4_config

.. toctree::
   :maxdepth:2
   :caption: API reference
   :hidden:

   api/cpp_library_root


.. How to contribute to this document
   ----------------------------------
   We come contributions to Qulacs Documentation!
   * **Contribution Guide**: :doc:`write/0_readme`.
   * **TODOs**: :doc:`write/1_contents`.
   * **Internationalization**: :doc:`write/2_intl`.


.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Contributing

   write/0_readme
   write/1_contents


