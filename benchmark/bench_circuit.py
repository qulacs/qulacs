import random
import time
from argparse import ArgumentParser

import numpy as np
from circuits import get_circuit
from mpi4py import MPI

from qulacs import QuantumCircuit, QuantumState
from qulacs.circuit import QuantumCircuitOptimizer as QCO


def get_option():
    argparser = ArgumentParser()
    argparser.add_argument(
        "-n", "--nqubits", type=int, default=10, help="Number of qubits (default 10)"
    )
    argparser.add_argument(
        "-d", "--depth", type=int, default=10, help="Number of Depth (default 10)"
    )
    argparser.add_argument(
        "-o",
        "--opt",
        type=int,
        default=-1,
        help="Enable QuantumCircuitOptimizer: 99 is to use optmize_light(), 0-6 is to use optimize(). (default no-use)",
    )
    argparser.add_argument(
        "-f",
        "--fused",
        type=int,
        default=-1,
        help="Enable QuantumCircuitOptimizer: 0 is not to use, 1-2 is to use Fused-swap opt",
    )
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose switch, Output circuit infomations",
    )
    argparser.add_argument("-s", "--seed", type=int, default=-1, help="Random Seed")
    argparser.add_argument(
        "-t",
        "--circuit",
        type=str,
        default="quantumvolume",
        help='Specify the Quantum Circuit name: "quantumvolume" or "qulacsbench" can be chosen.',
    )
    argparser.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="Check each values of the state vector between with and without multi-cpu",
    )
    return argparser.parse_args()


tStart = time.perf_counter()


def elapsed():
    global tStart
    tNow = time.perf_counter()
    elap = tNow - tStart
    tStart = tNow
    return f"elapsed_time= {elap:.6f}"


if __name__ == "__main__":
    mpicomm = MPI.COMM_WORLD
    mpirank = mpicomm.Get_rank()
    mpisize = mpicomm.Get_size()

    args = get_option()
    n = args.nqubits
    num_global_qubits = int(np.log2(mpisize))

    st = QuantumState(n, use_multi_cpu=True)
    if args.circuit == "quantumvolume":
        assert (
            n - num_global_qubits >= num_global_qubits * 2
        ), "ERROR: number of local qubits is too small"
    elif args.circuit == "qulacsbench":
        assert (
            n - num_global_qubits >= num_global_qubits
        ), "ERROR: number of local qubits is too small"
    else:
        assert False, "ERROR: " + args.circuit + " is not supported"

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

    # make a quantum circuit
    circuit = get_circuit(
        args.circuit,
        nqubits=args.nqubits,
        global_nqubits=0 if args.fused >= 0 else num_global_qubits,
        depth=args.depth,
        verbose=(args.verbose and (mpirank == 0)),
        random_gen=rng,
    )

    if args.check:
        circuit_ref = circuit.copy()
    if args.opt == 99:
        if args.fused == -1:
            QCO().optimize_light(circuit)
        else:
            QCO().optimize_light(circuit, args.fused)
    elif args.opt >= 0:
        if args.fused == -1:
            QCO().optimize(circuit, args.opt)
        else:
            QCO().optimize(circuit, args.opt, args.fused)

    if mpirank == 0:
        print(
            "# A quantum circuit was created. ",
            args.circuit,
            f"nqubits= {args.nqubits} depth= {args.depth}",
            " optimize= {args.opt, args.fused}",
            elapsed(),
        )
        print(circuit)

    # update the state vector
    circuit.update_quantum_state(st)
    mpicomm.barrier()
    if mpirank == 0:
        print("# The state vector has been updated.", elapsed())

    # Check that the result is the same as if it were not distributed
    if args.check:
        if mpirank == 0:
            print("# check ", end="")
        circuit_ref.update_quantum_state(st_ref, seed)
        vec = st.get_vector()
        vec_ref = st_ref.get_vector()
        offs = 0
        if st.get_device_name() == "multi-cpu":
            offs = ((1 << args.nqubits) // mpisize) * mpirank
        if np.allclose(vec, vec_ref[offs : offs + len(vec)]):
            if mpirank == 0:
                print("OK!", elapsed())
        else:
            print("# check NG! rank=", mpirank)
            print("# vec    =", vec)
            print("# vec_ref=", vec_ref[offs : offs + len(vec)])
            exit()

    # sampling using the state vector
    samples = st.sampling(10, seed)
    if mpirank == 0:
        print(f"# sampling results(seed= {seed}) {samples}", elapsed())
        print()

    del st
# EOF
