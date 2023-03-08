# Qulacs Documentation

Qulacs is a fast quantum circuit simulator for simulating large, noisy,
or parametric quantum circuits. Implemented in C/C++ and with Python
interface, Qulacs achieved both high speed circuit simulation and high
usability.

Qulacs was developed in Prof. [Fujii\'sgroup](http://quantphys.org/wp/qinfp/). Maintained and developing new features by [QunaSys](http://www.qunasys.com/).


<style type="text/css">
.center {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 40%;
}
.column-h {
width: 100%;
}
</style>
<div style="display: flex;">
<div class="column-h">
    <div class="center">
    <a href="https://www.github.com/qulacs/qulacs">
        <img align="middle" src="_static/images/github.png" width="160">
        GitHub
    </a>
    </div>
</div>
<div class="column-h">
    <div class="center">
    <a href="https://join.slack.com/t/qulacs/shared_invite/enQtNzY1OTM5MDYxMjAxLWM1ZDc3MzdiNjZhZjdmYTQ5MTJiOTEzZjI3ZjAwZTg0OGFiNjcxY2VjZWRjMWY0YjE5ZTViOWQzZTliYzdmYzY">
        <img src="_static/images/slack.png" width="160">
        Slack Community
    </a>
    </div>
</div>
<div class="column-h">
    <div class="center">
    <a href="https://dojo.qulacs.org/">
        <img align="middle" src="_static/images/dojo.png" width="160">
        Study material (Japanese)
    </a>
    </div>
</div>
</div>

## Get Started

```
pip install qulacs
```

Detailed instructions about installation in {doc}`intro/1_install`. 
Having trouble starting? see {doc}`intro/2_faq`.
For basic usage of Qulacs in Python and C++, see {doc}`intro/3_usage`.

```{toctree}
:maxdepth: 1
:hidden:
:caption: Get Started

intro/0_about
intro/1_install
intro/2_faq
intro/3_usage
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

intro/4.1_python_tutorial
intro/4.2_cpp_tutorial
```

```{toctree}
:maxdepth: 2
:caption: User Manual

guide/2.0_python_advanced
```

```{toctree}
:maxdepth: 2
:caption: Applications

apply/0_overview
```

```{toctree}
:maxdepth: 2
:caption: API reference
:hidden:

pyRef/index
api/cpp_library_root
```

```{toctree}
:maxdepth: 1
:caption: Contributing

write/0_readme
```
