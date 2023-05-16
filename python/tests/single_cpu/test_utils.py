import warnings

from openfermion import QubitOperator

from qulacs import QuantumState, to_general_quantum_operator
from qulacs.gate import RandomUnitary
from qulacs.state import inner_product
from qulacs.utils import convert_openfermion_op

# suppress warning from openfermion/cirq
warnings.simplefilter("ignore", category=DeprecationWarning)


class TestUtils:
    def test_convert_openfermion_op(self) -> None:
        openfermion_op = QubitOperator()
        openfermion_op += 1.0 * QubitOperator("X0")
        openfermion_op += 2.0 * QubitOperator("Z0 Y1")
        openfermion_op += 1.0 * QubitOperator("Z0 Y3")

        qulacs_op = convert_openfermion_op(openfermion_op)
        str_qulacs_op = str(qulacs_op)
        # operator ordering in openfermion may differ sometimes so we have to do this
        str_qulacs_op = str_qulacs_op.replace("(1,0) X 0", "")
        str_qulacs_op = str_qulacs_op.replace("(2,0) Z 0 Y 1", "")
        str_qulacs_op = str_qulacs_op.replace("(1,0) Z 0 Y 3", "")
        assert str_qulacs_op == " +  + "

    def test_GQO(self) -> None:
        self.n = 4
        self.dim = 2**self.n
        stateA = QuantumState(self.n)
        stateB = QuantumState(self.n)
        stateC = QuantumState(self.n)
        stateB.set_Haar_random_state()
        RanGate = RandomUnitary([0, 1, 2, 3])
        GQO_ret = to_general_quantum_operator(RanGate, 4, 0.00001)
        GQO_ret.apply_to_state(stateA, stateB, stateC)
        RanGate.update_quantum_state(stateB)

        inpro = inner_product(stateB, stateC).real
        assert inpro > 0.99, "GQO_test"
