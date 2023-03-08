# Qulacs ドキュメンテーション

Qulacsは、高速な量子回路シミュレータであり、大きな量子回路やノイズがあったり、パラメトリックな量子回路にも対応しております。
Qulacsは、C/C++で実装されており、Pythonインターフェイスもあるため、高速回路シミュレーションと高い操作性の両立しました。

Qulacsは、[藤井研究室](http://quantphys.org/wp/qinfp/) で開発され、[QunaSys](http://www.qunasys.com/)による新機能の開発とメンテナンスが行われています。

<style type="text/css">
.center {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 35%;
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

## インストール方法

```
pip install qulacs
```

トラブルシューティング: {doc}`intro/2_faq`.

```{toctree}
:maxdepth: 1
:hidden:
:caption: 入門

intro/0_about
intro/1_install
intro/2_faq
intro/3_usage
```

```{toctree}
:maxdepth: 1
:caption: チュートリアル

intro/4.1_python_tutorial
guide/2.0_python_advanced
intro/4.2_cpp_tutorial
```

```{toctree}
:maxdepth: 2
:caption: API リファレンス
:hidden:

pyRef/index
api/cpp_library_root
```
