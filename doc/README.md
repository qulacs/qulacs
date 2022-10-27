## How to build

```bash
git clone https://github.com/qulacs/qulacs.git
cd qulacs
```

Open workspace using devcontainer. Rebuild if needed.

```bash
pip install .[doc]
cd doc/en
make html
```

Open `en/_build/html/index.html` in your browser.
