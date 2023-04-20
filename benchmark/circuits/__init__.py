import importlib


def get_circuit(name, **model_args):
    """Factory function for constructing a quantum circuit by name with args"""
    circ = importlib.import_module("." + name, "circuits")
    return circ.build_circuit(**model_args)
