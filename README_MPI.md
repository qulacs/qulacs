# Qulacs with MPI General info

## Functionality
- Quantum state generation & gate simulation with multi-process and multi-nodes using distributed state vector (Single Program Multiple Data:SPMD model).
- Qulacs distributes a state (QuantumState) when it is instantiated and flag "use_multi_cpu=true" is enabled.
  - However, in the case "(N - k) <= log2(S)", the flag is ignored.
    "S", "N" and "k" are the MPI size, the number of qubits and the min number of qubit per process (k = 2 constant), respectively.
- Please also see Limitation

<hr>

## API for MPI
  - Instantiation of QuantumState
    - QuantumState state(qubits, use_multi_cpu)
      - use_multi_cpu = false
        -  Generate state vector in a node (same as the original)
      - use_multi_cpu = true
        -  Generate a state vector in multiple nodes if possible.
        -  qubits are divided into local_qc + global_qc internally.
            - local_qc: qubits in one node
            - global_qc: qubits in multiple nodes (=log2(#rank))
    - state.get_device_name()
      - return the list of devices having the state vector.

        | ret value | explanation|
        | -------- | -------- |
        | "cpu"   | state vector generated in a cpu |
        | "multi-cpu" | state vector generated in multi cpu |
        | "gpu"   | state vector generated in a gpu |

    - state.to_string()
      Each rank outputs only state information that has itself.
        ```
        to_string() example
        -- Output from rank 0 process ------------------
         *** Quantum State ***
         * Qubit Count : 10
         * Local Qubit Count : 9
         * Local Qubit Count : 1       // MPI-size=2
         * Dimension   : 1024
          (1.0 ,0)                     // c0000000000
          (0, 0)                       // c0000000001
          ...
          (0, 0)                       // c0111111111

        -- Output from rank 1 process ------------------
         * State vector (rank 1):
          (0, 0)                       // c1000000000
          (0, 0)                       // c1000000001
          ...
          (0, 0)                       // c1111111111
        ```
  - state.set_Haar_random_state()
    - Initialize each item with random value
    - In the case state vector distributed in multi nodes
      - If the seed is not specified, random value in rank0 is broadcasted in all ranks.
      - Based on the specified or broadcasted seed, each rank uses (seed + rank) as a seed. Even if the same seed is set in a distributed state vector, the random created states are different if the number of divisions is different.

  - state.sample( number_sampling [, seed])
    - As the same as gate operation, you must call it in all ranks.
    - Even if a seed is not specified, the random value in rank0 is shared (bcast) and used as a seed.
    - If you specify a seed, use the same one in all ranks.

  - state.load(vector)  // not supported yet
    - In the case state vector distributed in multi nodes, load to the element with each rank.

  - state.get_vector()  // not supported yet
    - In the case state vector distributed in multi nodes, returns the elements that each rank has.

  - Automatic FusedSWAP gate insertion of QuantumCircuitOptimizer  // not supported yet
    - optimize(circuit, block_size, swap_level=0)
      - swap_level = 0
        - No SWAP/FusedSWAP gate insertion
      - swap_level = 1
        - Insert SWAP/FusedSWAP gates to reduce communication without changing gate order
      - swap_level = 2
        - Insert SWAP/FusedSWAP gates to reduce communication with changing gate order
    - optimize_light(circuit, swap_level=0)
      - swap_level = 0
        - No SWAP/FusedSWAP gate insertion
      - swap_level = 1
        - Insert SWAP/FusedSWAP gates to reduce communication without changing gate order
      - swap_level = 2
        - Insert SWAP/FusedSWAP gates to reduce communication with changing gate order

  - circuit.update_quantum_state(state, seed)  // not supported yet
    - Enables updating of the state vector with a random number of seeds

  - environmental variable
    QULACS_NUM_THREADS : Specifies the maximum number of threads to be used in Qulacs.
                         (Override OMP_NUM_THREADS; valid range is 1 - 1024)

## build/install
- Prerequisites (Verified version)
    - GCC 11.2
    - CMake 3.24.0
    - Open MPI 4.1

```shell
$ cd [Qulacs_Home]
$ python3 -m venv venv
$ . ./venv/bin/activate
$ pip install -U pip wheel
$ pip install pytest numpy mpi4py
$ C_COMPILER=mpicc CXX_COMPILER=mpic++ USE_MPI=Yes pip install .
```

### test
```
$ USE_TEST=Yes ./script/build_mpicc.sh
$ pushd build
$ make test
$ make pythontest
$ mpirun -n 2 pytest python/test
```

## Example
### Python sample code
```python=
import qulacs
from qulacs.gate import Y,CNOT,merge
if qulacs.check_build_for_mpi():
    from mpi4py import MPI
else:
    print("Qulacs module was build without USE_MPI.")
    exit()

mpicomm = MPI.COMM_WORLD
mpirank = mpicomm.Get_rank()

nqubits = 5

state_on_multi_cpu = qulacs.QuantumState(nqubits, use_multi_cpu=True)
state_on_multi_cpu.set_Haar_random_state()
# for check
state_on_single_cpu = qulacs.QuantumState(nqubits)
state_on_single_cpu.load(state_on_multi_cpu)

if mpirank == 0:
    print("# qulacs.check_build_for_mpi() =", qulacs.check_build_for_mpi())
print(state_on_multi_cpu)

circuit = qulacs.QuantumCircuit(nqubits)

circuit.add_X_gate(0)
merged_gate = merge(CNOT(0,1),Y(1))
circuit.add_gate(merged_gate)
circuit.add_RX_gate(1,0.5)
circuit.update_quantum_state(state_on_multi_cpu)

observable = qulacs.Observable(nqubits)
observable.add_operator(2.0, "X 2 Y 1 Z 0")
observable.add_operator(-3.0, "Z 2")
value_multi = observable.get_expectation_value(state_on_multi_cpu)
if mpirank == 0:
    print("value(multi cpu) =", value_multi)

# for check
circuit.update_quantum_state(state_on_single_cpu)
value_single = observable.get_expectation_value(state_on_single_cpu)
if mpirank == 0:
    print("value(single cpu) =", value_single)
```

### C++ sample code
```cpp=
#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nqubits = 5;
    QuantumState ref_state(nqubits); // use single cpu
    QuantumState state(nqubits, 1); // 1(ture): use_multi_cpu
    state.set_Haar_random_state();
    ref_state.load(&state);

    QuantumCircuit circuit(nqubits);
    circuit.add_X_gate(0);

    auto merged_gate = gate::merge(gate::CNOT(0, 1),gate::Y(1));
    circuit.add_gate(merged_gate);
    circuit.add_RX_gate(1, 0.5);
    circuit.update_quantum_state(&ref_state); // update SV w/o MPI
    circuit.update_quantum_state(&state); // update SV with MPI

    // sampling
    //   1st param. is number of sampling.
    //   2nd param. is random-seed.
    // You must call state.sampling on every mpi-ranks
    // with the same random seed.
    std::vector<ITYPE> ref_sample = ref_state.sampling(50, 2021); // not supported yet
    std::vector<ITYPE> sample = state.sampling(50, 2021); // not supported yet
    if (rank==0) {
        std::cout << "#result_state.sampling(      cpu): ";
        for (const auto& e : ref_sample) std::cout << e << " ";
        std::cout << std::endl;

        std::cout << "#result_state.sampling(multi-cpu): ";
        for (const auto& e : sample) std::cout << e << " ";
        std::cout << std::endl;
    }

    //
    // observable function is not available in mpi.
    //
    Observable observable(nqubits);
    observable.add_operator(2.0, "X 2 Y 1 Z 0");
    observable.add_operator(-3.0, "Z 2");
    auto ref_value = observable.get_expectation_value(&ref_state); // not supported yet
    auto value = observable.get_expectation_value(&state); // not supported yet
    std::cout << "#result observable(      cpu) " << ref_value << std::endl;
    std::cout << "#result observable(multi-cpu) " << value << std::endl;
    return 0;
}
```

## Limitation

- The number of MPI rank (WORLD_SIZE) should be 2^n
- Unsupported gates/functions may cause severe error.
- Build with USE_MPI does not support with USE_GPU

- The following items are supported. Qulacs with MPI does not support any other items.
  - QuantumCircuit
  - QuantumState
      - Constructor
      - get_device_name
      - get_vector
      - normalize
      - set_computational_basis
      - set_Haar_random_state
      - to_string
      - copy
      - load
      - get_entropy
      - sampling
  - gate
      - Identity / H / X / Y / Z
      - CNOT / CZ / SWAP
      - RX / RY / RZ
      - S / Sdag / T / Tdag
      - SqrtX / SqrtXdag / SqrtY / SqrtYdag
      - U1 / U2 / U3
      - P0 / P1
      - Pauli (single)
      - PauliRotation (single)
      - DenseMatrix gate (single-target, double-target)
      - DiagonalMatrix gate (single target)

## Additional info

- To be supported later (T.B.D.)
  - QuantumCircuitOptimizer
      - optimize
      - optimize_light
  - ParametricQuantumCircuit
  - gate
      - DenseMatrix gate (multi-target, single-control, multi-control)
      - Measurement
      - merge (number of qubits > 1)
      - Pauli (multi)
      - PauliRotation (multi)
      - CPTP
      - Instrument
      - Adaptive
      - RandomUnitary
      - to_matrix_gate
      - TOFFOLI
      - FREDKIN
      - DenseMatrix gate(multi control, multi target)
      - DiagonalMatrix(multi target)
  - GeneralQuantumOperator (w/o get_transition_amplitude)
  - Observable (w/o get_transition_amplitude)
  - PauliOperator (w/o get_transition_amplitude)

  - QuantumCircuitSimulator
  - state
      - inner_product
      - tensor_product
      - permutate_qubit
      - drop_qubit
      - partial_trace
      - get_zero_probability
      - get_marginal_probability
  - QuantumGateBase
  - QuantumGateMatrix
  - GeneralQuantumOperator.get_transition_amplitude( )
  - Observable.get_transition_amplitude( )
  - PauliOperator.get_transition_amplitude( )

- Might be supported in future (T.B.D.)
  - gate
      - SparseMatrix
      - ReversibleBoolean
      - StateReflection
      - BitFlipNoise
      - DephasingNoise
      - IndependentXZNoise
      - DepolarizingNoise
      - TwoQubitDepolarizingNoise
      - AmplitudeDampingNoise
      - add
      - Probabilistic
      - ProbabilisticInstrument
      - CP
      - merge(> 2qubit)
  - DensityMatrix simulation
  - QuantumGate_SingleParameter

