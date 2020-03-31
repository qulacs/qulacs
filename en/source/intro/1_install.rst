
Installation
----------------

If you encounter some troubles, see `troubleshooting`_.

Requirements
~~~~~~~~~~~~

-  python 2.7 or 3.x
-  gcc/g++ >= 7.0.0 or VisualStudio 2017
-  cmake >= 2.8
-  git

C++ Library(cppsim)
~~~~~~~~~~~~~~~~~~~

GCC
^^^

::

   git clone https://github.com/qulacs/qulacs.git
   cd qulacs
   ./script/build_gcc.sh

MSVC
^^^^

::

   git clone https://github.com/qulacs/qulacs.git
   cd qulacs
   generate_msvc_project.bat

Then, open ``Project.sln`` in ``./qulacs/visualstudio/``, and build all.

Python Interface(Qulacs)
~~~~~~~~~~~~~~~~~~~~~~~~

Install

::

   git clone https://github.com/qulacs/qulacs.git
   cd qulacs
   python setup.py install

Uninstall

::

   pip uninstall qulacs



.. _troubleshooting: http://qulacs.org/md_4__trouble_shooting.html
.. _C++ Tutorial: http://qulacs.org/md_2__tutorial__c_p_p.html
.. _Python Tutorial: http://qulacs.org/md_3__tutorial_python.html
.. _Examples: https://github.com/qulacs/quantum-circuits
.. _API document: http://qulacs.org/annotated.html
