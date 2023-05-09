import unittest
import warnings

import qulacs


class TestUtils(unittest.TestCase):
    def setUp(self):
        # suppress warning from openfermion/cirq
        warnings.simplefilter("ignore", category=DeprecationWarning)

    def tearDown(self):
        pass

    def test_convert_openfermion_op(self):
        from openfermion import QubitOperator

        from qulacs.utils import convert_openfermion_op

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
        self.assertEqual(str_qulacs_op, " +  + ")

    def test_GQO(self):
        self.n = 4
        self.dim = 2**self.n
        stateA = qulacs.QuantumState(self.n)
        stateB = qulacs.QuantumState(self.n)
        stateC = qulacs.QuantumState(self.n)
        stateB.set_Haar_random_state()
        RanGate = qulacs.gate.RandomUnitary([0, 1, 2, 3])
        GQO_ret = qulacs.to_general_quantum_operator(RanGate, 4, 0.00001)
        GQO_ret.apply_to_state(stateA, stateB, stateC)
        RanGate.update_quantum_state(stateB)

        inpro = qulacs.state.inner_product(stateB, stateC).real
        self.assertTrue(inpro > 0.99, msg="GQO_test")
