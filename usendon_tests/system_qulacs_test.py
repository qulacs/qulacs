import qulacs
from qulacs.gate import Y, CNOT, merge
from qulacs.circuit import QuantumCircuitOptimizer as QCO
import numpy as np
if qulacs.check_build_for_mpi():
    from mpi4py import MPI
else:
    print("Qulacs module was build without USE_MPI.")
    exit()

mpicomm = MPI.COMM_WORLD
mpirank = mpicomm.Get_rank()
mpisize = mpicomm.Get_size()
globalqubits = int(np.log2(mpisize))


nqubits = 4

state = qulacs.QuantumState(nqubits, use_multi_cpu = True)

if mpirank == 0:
    devicename = state.get_device_name()
    print("Device name of the state vector:", devicename)
    if devicename == "multi-cpu":
        print("- Number of qubits:", nqubits)
        print("- Number of global qubits:", globalqubits)
        print("- Number of local qubits:", nqubits - globalqubits)

# build a circuit
circuit = qulacs.QuantumCircuit(nqubits)
circuit.add_H_gate(0)
circuit.add_CNOT_gate(0, 1)
""" merged_gate = merge(CNOT(0,1), Y(1))
circuit.add_gate(merged_gate)
circuit.add_RX_gate(1, 0.5) """

# updating the state
#QCO().optimize_light(circuit)
circuit.update_quantum_state(state)

print("Sampling =", state.sampling(20))

