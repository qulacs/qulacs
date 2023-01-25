import inspect
import os
import sys
from importlib import import_module

if __name__ == "__main__":
    module_name = sys.argv[1]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        print("Module not found: " + module_name)

    with open("%s/names_%s.py" % (os.path.dirname(__file__), module_name), "w") as f:
        names = list()
        submodule_names = list()
        objects = list()

        def search_names(obj, name):
            # ref: https://github.com/Qulacs-Osaka/qulacs-osaka/issues/234
            if (
                name == "qulacs.Observable.get_matrix"
                or name == "qulacs.GeneralQuantumOperator.get_matrix"
            ):
                return

            if obj in objects:
                return
            names.append(name)
            if inspect.ismodule(obj):
                submodule_names.append(name)
            objects.append(obj)

            if not (inspect.isclass(obj) or inspect.ismodule(obj)):
                return
            for subobj in inspect.getmembers(obj):
                if subobj[0][0] == "_":
                    continue
                search_names(subobj[1], name + "." + subobj[0])

        search_names(module, module_name)
        for submodule_name in submodule_names:
            f.write("import " + submodule_name + "\n")
        names_list = sorted(names)
        for name in names_list:
            f.write(name + "\n")
