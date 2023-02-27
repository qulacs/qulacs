from argparse import ArgumentParser
import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.circuit import QuantumCircuitOptimizer as QCO
from circuits import get_circuit
import time
import random
from mpi4py import MPI

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-n', '--nqubits', type=int,
            default=10, help='Number of qubits (default 10)')
    argparser.add_argument('-d', '--depth', type=int,
            default=10, help='Number of Depth (default 10)')
    #argparser.add_argument('-o', '--opt', type=int,
    #        default=-1, help='Enable QuantumCircuitOptimizer: 99 is to use optmize_light(), 0-6 is to use optimize(). (default no-use)')
    #argparser.add_argument('-f', '--fused', type=int,
    #        default=-1, help='Enable QuantumCircuitOptimizer: 0 is not to use, 1-2 is to use Fused-swap opt')
    argparser.add_argument('-v', '--verbose',
            action='store_true',
            help='Verbose switch, Output circuit infomations')
    argparser.add_argument('-s', '--seed', type=int,
            default=-1, help='Random Seed')
    argparser.add_argument('-t', '--circuit', type=str,
            default="quantumvolume", help='Specify the Quantum Circuit name: "quantumvolume" or "qulacsbench" can be chosen.')
    argparser.add_argument('-c', '--check', action='store_true', help='Check each values of the state vector between with and without multi-cpu')
    return argparser.parse_args()

if __name__ == '__main__':
    mpicomm = MPI.COMM_WORLD
    mpirank = mpicomm.Get_rank()
    mpisize = mpicomm.Get_size()

    args = get_option()
    n = args.nqubits
    num_global_qubits = int(np.log2(mpisize))

    circuit_name = args.circuit + str(args.depth)

    st = QuantumState(n, use_multi_cpu=True)
    if args.circuit == "quantumvolume":
        assert n - num_global_qubits >= num_global_qubits * 2
    elif args.circuit == "qulacsbench":
        assert n - num_global_qubits >= num_global_qubits
    else:
        assert False # not supported circuit name

    if args.check:
        st_ref = QuantumState(n)

    if args.seed >= 0:
        seed = args.seed
        random.seed(seed)
        rng = np.random.default_rng(seed=seed)
    else:
        seed = random.randint(0, 9999)
        seed = mpicomm.bcast(seed, root=0)
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed=seed)

    circuit = get_circuit(args.circuit,
            nqubits=args.nqubits,
            global_nqubits=num_global_qubits,
            depth=args.depth,
            verbose=(args.verbose and (mpirank == 0)),
            random_gen=rng)
    #circuit = mpicomm.bcast(circuit, root=0)
    if args.check:
        circuit_ref = circuit.copy()
    #if args.opt == 99:
    #    if args.fused == -1:
    #        QCO().optimize_light(circuit)
    #    else:
    #        QCO().optimize_light(circuit, args.fused)
    #elif args.opt >= 0:
    #    if args.fused == -1:
    #        QCO().optimize(circuit, args.opt)
    #    else:
    #        QCO().optimize(circuit, args.opt, args.fused)

    tStart = time.perf_counter()
    circuit.update_quantum_state(st)
    mpicomm.barrier()
    if mpirank == 0: print("# The state vector has been updated! time=", time.perf_counter() - tStart)

    if args.check:
        if mpirank == 0: print("# check ", end="")
        circuit_ref.update_quantum_state(st_ref)
        vec = st.get_vector()
        vec_ref = st_ref.get_vector()
        offs = 0
        if st.get_device_name() == "multi-cpu":
            offs = ((1 << args.nqubits) // mpisize) * mpirank
        if np.allclose(vec, vec_ref[offs:offs+len(vec)]):
            if mpirank == 0: print("OK!")
        else:
            print("# check NG! rank=", mpirank)
            print("# vec    =", vec)
            print("# vec_ref=", vec_ref[offs:offs+len(vec)])
            exit()

    samples = st.sampling(10, seed)
    if mpirank == 0: print(circuit_name, samples)
    del st

#EOF
