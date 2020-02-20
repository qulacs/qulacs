Qulacs Documentation
======================================

.. meta::
   :description lang=en: qulacs documentation

Qulacs is a fast quantum circuit simulator for simulating large, noisy, or parametric quantum circuits.
Implemented in C/C++ and with python interface, Qulacs achieved both high speed circuit simulation and high usability.

Qulacs was developed in Prof. `Fujii's group <http://quantphys.org/wp/qinfp/>`_. Maintained and developing new features by `QunaSys <http://www.qunasys.com/>`_.

:doc:`intro/About_Qulacs`.


`Github repository
<https://github.com/qulacs/qulacs>`_ |
`Slack Community <https://join.slack.com/t/qulacs/shared_invite/enQtNzY1OTM5MDYxMjAxLWM1ZDc3MzdiNjZhZjdmYTQ5MTJiOTEzZjI3ZjAwZTg0OGFiNjcxY2VjZWRjMWY0YjE5ZTViOWQzZTliYzdmYzY>`_ | 
`Study Material (Japanese) <https://dojo.qulacs.org/>`_ 



Quick-Start
-----------
::
  
   pip install qulacs

More detail instruction in :doc:`intro/HowToInstall`.
Having trouble starting? see :doc:`intro/TroubleShooting`.


.. toctree::
   :maxdepth: 2
   
   :caption: First steps

   intro/About_Qulacs
   intro/HowToInstall
   intro/TroubleShooting

Getting statred with Qulacs
-----------------------------------

* **Installation**: :doc:`intro/install`.

* **Tutorials**: :doc:`intro/Tutorial_python_first` | :doc:`intro/Tutorial_CPP`.


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting started
   
   intro/install
   intro/Tutorial_python_first
   intro/Tutorial_CPP


Advanced features
----------------------------------

Brief description of the Advanced feature

* **User Manual**: :doc:`guide/0_root`.

* **Advanced Tutorials**: :doc:`guide/2_python`. :doc:`guide/3_cpp`.

* **Qulacs Cookbooks**: :doc:`apply/0_root`.

* **C++ API reference**: :doc:`api/cpp_library_root`.



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Advanced features
   
   guide/0_root
   guide/1_concepts
   guide/2_python
   guide/3_cpp
   guide/4_config
   apply/0_root
   api/cpp_library_root


How to contribute to this document
----------------------------------

We come contributions to Qulacs Documentation!

* **Contribution Guide**: :doc:`write/readme`.

* **TODOs**

* **Internationalization**

.. toctree::
   :maxdepth: 1
   :hidden:
   :glob:
   :caption: How to write this doc
   
   write/readme
   write/contents
   write/intl


