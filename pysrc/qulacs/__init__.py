from qulacs_core import *
import qulacs.observable._get_matrix

Observable.get_matrix = \
    lambda obs: qulacs.observable._get_matrix._get_matrix(obs)

GeneralQuantumOperator.get_matrix = \
    lambda obs: qulacs.observable._get_matrix._get_matrix(obs)
