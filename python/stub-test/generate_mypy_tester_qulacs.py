import inspect
import os
from importlib import import_module

if __name__ == "__main__":
    check_modules = ["qulacs_osaka"]
    for module_name in check_modules:
        module = import_module(module_name)

        with open(
            "%s/names_%s.py" % (os.path.dirname(__file__), module_name), "w"
        ) as f:
            names = set()

            def append_names(obj, name):
                if name in names:
                    return

                for func in inspect.getmembers(obj, inspect.isroutine):
                    if func[0][0] == "_":
                        continue
                    names.add(name + "." + func[0])
                for cls in inspect.getmembers(obj, inspect.isclass):
                    if cls[0][0] == "_":
                        continue
                    append_names(cls[1], name + "." + cls[0])
                for mod in inspect.getmembers(obj, inspect.isclass):
                    if mod[0][0] == "_":
                        continue
                    append_names(mod[1], name + "." + mod[0])

            append_names(module, module_name)
            f.write("import " + module_name + "\n")
            names_list = sorted(names)
            for name in names_list:
                f.write(name + "\n")
