from qulacs_core import *

import qulacs.observable._get_matrix
from qulacs._version import __version__, __version_tuple__

Observable.get_matrix = \
    lambda obs: qulacs.observable._get_matrix._get_matrix(obs)

GeneralQuantumOperator.get_matrix = \
    lambda obs: qulacs.observable._get_matrix._get_matrix(obs)
