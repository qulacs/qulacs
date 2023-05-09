import unittest

import numpy as np

import qulacs


class TestDensityMatrixHandling(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_density_matrix(self):
        num_qubit = 5
        sv = qulacs.StateVector(num_qubit)
        dm = qulacs.DensityMatrix(num_qubit)
        sv.set_Haar_random_state(seed=0)
        dm.load(sv)
        svv = np.atleast_2d(sv.get_vector()).T
        mat = np.dot(svv, svv.T.conj())
        self.assertTrue(
            np.allclose(dm.get_matrix(), mat), msg="check pure matrix to density matrix"
        )

    def test_tensor_product_sv(self):
        num_qubit = 4
        sv1 = qulacs.StateVector(num_qubit)
        sv2 = qulacs.StateVector(num_qubit)
        sv1.set_Haar_random_state(seed=0)
        sv2.set_Haar_random_state(seed=1)
        sv3 = qulacs.state.tensor_product(sv1, sv2)
        sv3_test = np.kron(sv1.get_vector(), sv2.get_vector())
        self.assertTrue(
            np.allclose(sv3_test, sv3.get_vector()),
            msg="check pure state tensor product",
        )
        del sv1
        del sv2
        del sv3

    def test_tensor_product_dm(self):
        num_qubit = 4
        dm1 = qulacs.DensityMatrix(num_qubit)
        dm2 = qulacs.DensityMatrix(num_qubit)
        dm1.set_Haar_random_state(seed=0)
        dm2.set_Haar_random_state(seed=1)
        dm3 = qulacs.state.tensor_product(dm1, dm2)
        dm3_test = np.kron(dm1.get_matrix(), dm2.get_matrix())
        self.assertTrue(
            np.allclose(dm3_test, dm3.get_matrix()),
            msg="check density matrix tensor product",
        )
        del dm1
        del dm2
        del dm3

    def test_tensor_product_different_size_sv(self):
        num_qubit = 4
        sv1 = qulacs.StateVector(num_qubit)
        sv2 = qulacs.StateVector(num_qubit + 1)
        sv1.set_Haar_random_state(seed=0)
        sv2.set_Haar_random_state(seed=1)
        sv3 = qulacs.state.tensor_product(sv1, sv2)
        sv3_test = np.kron(sv1.get_vector(), sv2.get_vector())
        self.assertTrue(
            np.allclose(sv3_test, sv3.get_vector()),
            msg="check pure state tensor product",
        )
        del sv1
        del sv2
        del sv3

    def test_tensor_product_different_size_dm(self):
        num_qubit = 4
        dm1 = qulacs.DensityMatrix(num_qubit)
        dm2 = qulacs.DensityMatrix(num_qubit + 1)
        dm1.set_Haar_random_state(seed=0)
        dm2.set_Haar_random_state(seed=1)
        dm3 = qulacs.state.tensor_product(dm1, dm2)
        dm3_test = np.kron(dm1.get_matrix(), dm2.get_matrix())
        self.assertTrue(
            np.allclose(dm3_test, dm3.get_matrix()),
            msg="check density matrix tensor product",
        )
        del dm1
        del dm2
        del dm3

    def test_permutate_qubit_sv(self):
        num_qubit = 8
        sv = qulacs.StateVector(num_qubit)
        sv.set_Haar_random_state(seed=0)
        order = np.arange(num_qubit)
        np.random.shuffle(order)

        arr = []
        for ind in range(2**num_qubit):
            s = format(ind, "0{}b".format(num_qubit))
            s = np.array(list(s[::-1]))
            v = np.array(["*"] * num_qubit)
            for ind in range(len(s)):
                v[order[ind]] = s[ind]
            s = ("".join(v))[::-1]
            arr.append(int(s, 2))

        sv_perm = qulacs.state.permutate_qubit(sv, order)
        self.assertTrue(
            np.allclose(sv.get_vector()[arr], sv_perm.get_vector()),
            msg="check pure state permutation",
        )
        del sv_perm
        del sv

    def test_permutate_qubit_dm(self):
        num_qubit = 3
        dm = qulacs.DensityMatrix(num_qubit)
        dm.set_Haar_random_state(seed=0)
        order = np.arange(num_qubit)
        np.random.shuffle(order)

        arr = []
        for ind in range(2**num_qubit):
            s = format(ind, "0{}b".format(num_qubit))
            s = np.array(list(s[::-1]))
            v = np.array(["*"] * num_qubit)
            for ind in range(len(s)):
                v[order[ind]] = s[ind]
            s = ("".join(v))[::-1]
            arr.append(int(s, 2))

        dm_perm = qulacs.state.permutate_qubit(dm, order)
        dm_perm_test = dm.get_matrix()
        dm_perm_test = dm_perm_test[arr, :]
        dm_perm_test = dm_perm_test[:, arr]
        self.assertTrue(
            np.allclose(dm_perm_test, dm_perm.get_matrix()),
            msg="check density matrix permutation",
        )
        del dm_perm
        del dm

    def test_partial_trace_dm(self):
        num_qubit = 5
        num_traceout = 2
        dm = qulacs.DensityMatrix(num_qubit)
        dm.set_Haar_random_state(seed=0)
        mat = dm.get_matrix()

        target = np.arange(num_qubit)
        np.random.shuffle(target)
        target = target[:num_traceout]
        target_cor = [num_qubit - 1 - i for i in target]
        target_cor.sort()

        dmt = mat.reshape([2, 2] * num_qubit)
        for cnt, val in enumerate(target_cor):
            ofs = num_qubit - cnt
            dmt = np.trace(dmt, axis1=val - cnt, axis2=ofs + val - cnt)
        dmt = dmt.reshape(
            [2 ** (num_qubit - num_traceout), 2 ** (num_qubit - num_traceout)]
        )

        pdm = qulacs.state.partial_trace(dm, target)
        self.assertTrue(
            np.allclose(pdm.get_matrix(), dmt), msg="check density matrix partial trace"
        )
        del dm, pdm

    def test_partial_trace_sv(self):
        num_qubit = 6
        num_traceout = 4
        sv = qulacs.StateVector(num_qubit)
        sv.set_Haar_random_state(seed=0)
        svv = np.atleast_2d(sv.get_vector()).T
        mat = np.dot(svv, svv.T.conj())

        target = np.arange(num_qubit)
        np.random.shuffle(target)
        target = target[:num_traceout]
        target_cor = [num_qubit - 1 - i for i in target]
        target_cor.sort()

        dmt = mat.reshape([2, 2] * num_qubit)
        for cnt, val in enumerate(target_cor):
            ofs = num_qubit - cnt
            dmt = np.trace(dmt, axis1=val - cnt, axis2=ofs + val - cnt)
        dmt = dmt.reshape(
            [2 ** (num_qubit - num_traceout), 2 ** (num_qubit - num_traceout)]
        )

        pdm = qulacs.state.partial_trace(sv, target)
        self.assertTrue(
            np.allclose(pdm.get_matrix(), dmt), msg="check pure state partial trace"
        )
