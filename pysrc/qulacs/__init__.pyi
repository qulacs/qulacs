"""cppsim python interface"""
from __future__ import annotations
import qulacs_core
import typing
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "CausalConeSimulator",
    "DensityMatrix",
    "GeneralQuantumOperator",
    "GradCalculator",
    "NoiseSimulator",
    "Observable",
    "ParametricQuantumCircuit",
    "PauliOperator",
    "QuantumCircuit",
    "QuantumCircuitSimulator",
    "QuantumGateBase",
    "QuantumGateMatrix",
    "QuantumGate_SingleParameter",
    "QuantumState",
    "QuantumStateBase",
    "StateVector",
    "circuit",
    "gate",
    "observable",
    "quantum_operator",
    "state"
]


class CausalConeSimulator():
    def __init__(self, arg0: ParametricQuantumCircuit, arg1: Observable) -> None: 
        """
        Constructor
        """
    def build(self) -> None: 
        """
        build
        """
    def get_circuit_list(self) -> typing.List[typing.List[ParametricQuantumCircuit]]: 
        """
        return circuit_list
        """
    def get_coef_list(self) -> typing.List[complex]: 
        """
        return coef_list
        """
    def get_expectation_value(self) -> complex: 
        """
        return expectation_value
        """
    def get_pauli_operator_list(self) -> typing.List[typing.List[PauliOperator]]: 
        """
        return pauli_operator_list
        """
    pass
class QuantumStateBase():
    pass
class GeneralQuantumOperator():
    def __IADD__(self, arg0: PauliOperator) -> GeneralQuantumOperator: ...
    @typing.overload
    def __IMUL__(self, arg0: PauliOperator) -> GeneralQuantumOperator: ...
    @typing.overload
    def __IMUL__(self, arg0: complex) -> GeneralQuantumOperator: ...
    def __ISUB__(self, arg0: PauliOperator) -> GeneralQuantumOperator: ...
    @typing.overload
    def __add__(self, arg0: GeneralQuantumOperator) -> GeneralQuantumOperator: ...
    @typing.overload
    def __add__(self, arg0: PauliOperator) -> GeneralQuantumOperator: ...
    def __iadd__(self, arg0: GeneralQuantumOperator) -> GeneralQuantumOperator: ...
    def __imul__(self, arg0: GeneralQuantumOperator) -> GeneralQuantumOperator: ...
    def __init__(self, qubit_count: int) -> None: 
        """
        Constructor
        """
    def __isub__(self, arg0: GeneralQuantumOperator) -> GeneralQuantumOperator: ...
    @typing.overload
    def __mul__(self, arg0: GeneralQuantumOperator) -> GeneralQuantumOperator: ...
    @typing.overload
    def __mul__(self, arg0: PauliOperator) -> GeneralQuantumOperator: ...
    @typing.overload
    def __mul__(self, arg0: complex) -> GeneralQuantumOperator: ...
    def __str__(self) -> str: 
        """
        to string
        """
    @typing.overload
    def __sub__(self, arg0: GeneralQuantumOperator) -> GeneralQuantumOperator: ...
    @typing.overload
    def __sub__(self, arg0: PauliOperator) -> GeneralQuantumOperator: ...
    @typing.overload
    def add_operator(self, coef: complex, pauli_string: str) -> None: 
        """
        Add Pauli operator

        Add Pauli operator
        """
    @typing.overload
    def add_operator(self, pauli_operator: PauliOperator) -> None: ...
    def apply_to_state(self, work_state: QuantumStateBase, state_to_be_multiplied: QuantumStateBase, dst_state: QuantumStateBase) -> None: 
        """
        Apply observable to `state_to_be_multiplied`. The result is stored into `dst_state`.
        """
    def copy(self) -> GeneralQuantumOperator: 
        """
        Create copied instance of General Quantum operator class
        """
    def get_expectation_value(self, state: QuantumStateBase) -> complex: 
        """
        Get expectation value
        """
    def get_expectation_value_single_thread(self, state: QuantumStateBase) -> complex: 
        """
        Get expectation value
        """
    def get_qubit_count(self) -> int: 
        """
        Get qubit count
        """
    def get_state_dim(self) -> int: 
        """
        Get state dimension
        """
    def get_term(self, index: int) -> PauliOperator: 
        """
        Get Pauli term
        """
    def get_term_count(self) -> int: 
        """
        Get count of Pauli terms
        """
    def get_transition_amplitude(self, state_bra: QuantumStateBase, state_ket: QuantumStateBase) -> complex: 
        """
        Get transition amplitude
        """
    def is_hermitian(self) -> bool: 
        """
        Get is Herimitian
        """
    pass
class GradCalculator():
    def __init__(self) -> None: ...
    @staticmethod
    @typing.overload
    def calculate_grad(*args, **kwargs) -> typing.Any: 
        """
        Calculate Grad

        Calculate Grad
        """
    @typing.overload
    def calculate_grad(self, parametric_circuit: ParametricQuantumCircuit, observable: Observable) -> typing.List[complex]: ...
    pass
class NoiseSimulator():
    def __init__(self, arg0: QuantumCircuit, arg1: QuantumState) -> None: 
        """
        Constructor
        """
    def execute(self, arg0: int) -> typing.List[int]: 
        """
        sampling & return result [array]
        """
    pass
class Observable(GeneralQuantumOperator):
    def __init__(self, qubit_count: int) -> None: 
        """
        Constructor
        """
    def __str__(self) -> str: 
        """
        to string
        """
    @typing.overload
    def add_operator(self, coef: complex, string: str) -> None: 
        """
        Add Pauli operator

        Add Pauli operator
        """
    @typing.overload
    def add_operator(self, pauli_operator: PauliOperator) -> None: ...
    def add_random_operator(self, operator_count: int) -> None: 
        """
        Add random pauli operator
        """
    def apply_to_state(self, work_state: QuantumStateBase, state_to_be_multiplied: QuantumStateBase, dst_state: QuantumStateBase) -> None: 
        """
        Apply observable to `state_to_be_multiplied`. The result is stored into `dst_state`.
        """
    def get_expectation_value(self, state: QuantumStateBase) -> float: 
        """
        Get expectation value
        """
    def get_expectation_value_single_thread(self, state: QuantumStateBase) -> float: 
        """
        Get expectation value
        """
    def get_qubit_count(self) -> int: 
        """
        Get qubit count
        """
    def get_state_dim(self) -> int: 
        """
        Get state dimension
        """
    def get_term(self, index: int) -> PauliOperator: 
        """
        Get Pauli term
        """
    def get_term_count(self) -> int: 
        """
        Get count of Pauli terms
        """
    def get_transition_amplitude(self, state_bra: QuantumStateBase, state_ket: QuantumStateBase) -> complex: 
        """
        Get transition amplitude
        """
    def solve_ground_state_eigenvalue_by_arnoldi_method(self, state: QuantumStateBase, iter_count: int, mu: complex = 0.0) -> complex: 
        """
        Compute ground state eigenvalue by arnoldi method
        """
    def solve_ground_state_eigenvalue_by_lanczos_method(self, state: QuantumStateBase, iter_count: int, mu: complex = 0.0) -> complex: 
        """
        Compute ground state eigenvalue by lanczos method
        """
    def solve_ground_state_eigenvalue_by_power_method(self, state: QuantumStateBase, iter_count: int, mu: complex = 0.0) -> complex: 
        """
        Compute ground state eigenvalue by power method
        """
    pass
class QuantumCircuit():
    def __init__(self, qubit_count: int) -> None: 
        """
        Constructor
        """
    def __repr__(self) -> str: ...
    def add_CNOT_gate(self, control: int, target: int) -> None: 
        """
        Add CNOT gate
        """
    def add_CZ_gate(self, control: int, target: int) -> None: 
        """
        Add CNOT gate
        """
    def add_H_gate(self, index: int) -> None: 
        """
        Add Hadamard gate
        """
    def add_P0_gate(self, index: int) -> None: 
        """
        Add projection gate to |0> subspace
        """
    def add_P1_gate(self, index: int) -> None: 
        """
        Add projection gate to |1> subspace
        """
    def add_RX_gate(self, index: int, angle: float) -> None: 
        """
        Add Pauli-X rotation gate
        """
    def add_RY_gate(self, index: int, angle: float) -> None: 
        """
        Add Pauli-Y rotation gate
        """
    def add_RZ_gate(self, index: int, angle: float) -> None: 
        """
        Add Pauli-Z rotation gate
        """
    def add_SWAP_gate(self, target1: int, target2: int) -> None: 
        """
        Add SWAP gate
        """
    def add_S_gate(self, index: int) -> None: 
        """
        Add pi/4 phase gate
        """
    def add_Sdag_gate(self, index: int) -> None: 
        """
        Add adjoint of pi/4 phsae gate
        """
    def add_T_gate(self, index: int) -> None: 
        """
        Add pi/8 phase gate
        """
    def add_Tdag_gate(self, index: int) -> None: 
        """
        Add adjoint of pi/8 phase gate
        """
    @staticmethod
    def add_U1_gate(*args, **kwargs) -> typing.Any: 
        """
        Add QASM U1 gate
        """
    @staticmethod
    def add_U2_gate(*args, **kwargs) -> typing.Any: 
        """
        Add QASM U2 gate
        """
    @staticmethod
    def add_U3_gate(*args, **kwargs) -> typing.Any: 
        """
        Add QASM U3 gate
        """
    def add_X_gate(self, index: int) -> None: 
        """
        Add Pauli-X gate
        """
    def add_Y_gate(self, index: int) -> None: 
        """
        Add Pauli-Y gate
        """
    def add_Z_gate(self, index: int) -> None: 
        """
        Add Pauli-Z gate
        """
    @typing.overload
    def add_dense_matrix_gate(self, index: int, matrix: numpy.ndarray[numpy.complex128, _Shape[m, n]]) -> None: 
        """
        Add dense matrix gate

        Add dense matrix gate
        """
    @typing.overload
    def add_dense_matrix_gate(self, index_list: typing.List[int], matrix: numpy.ndarray[numpy.complex128, _Shape[m, n]]) -> None: ...
    def add_diagonal_observable_rotation_gate(self, observable: Observable, angle: float) -> None: 
        """
        Add diagonal observable rotation gate
        """
    @typing.overload
    def add_gate(self, gate: QuantumGateBase) -> None: 
        """
        Add gate with copy

        Add gate with copy
        """
    @typing.overload
    def add_gate(self, gate: QuantumGateBase, position: int) -> None: ...
    @typing.overload
    def add_multi_Pauli_gate(self, index_list: typing.List[int], pauli_ids: typing.List[int]) -> None: 
        """
        Add multi-qubit Pauli gate

        Add multi-qubit Pauli gate
        """
    @typing.overload
    def add_multi_Pauli_gate(self, pauli: PauliOperator) -> None: ...
    @typing.overload
    def add_multi_Pauli_rotation_gate(self, index_list: typing.List[int], pauli_ids: typing.List[int], angle: float) -> None: 
        """
        Add multi-qubit Pauli rotation gate

        Add multi-qubit Pauli gate
        """
    @typing.overload
    def add_multi_Pauli_rotation_gate(self, pauli: PauliOperator) -> None: ...
    def add_noise_gate(self, gate: QuantumGateBase, NoiseType: str, NoiseProbability: float) -> None: 
        """
        Add noise gate with copy
        """
    def add_observable_rotation_gate(self, observable: Observable, angle: float, repeat: int) -> None: 
        """
        Add observable rotation gate
        """
    def add_random_unitary_gate(self, index_list: typing.List[int]) -> None: 
        """
        Add random unitary gate
        """
    def add_sqrtX_gate(self, index: int) -> None: 
        """
        Add pi/4 Pauli-X rotation gate
        """
    def add_sqrtXdag_gate(self, index: int) -> None: 
        """
        Add adjoint of pi/4 Pauli-X rotation gate
        """
    def add_sqrtY_gate(self, index: int) -> None: 
        """
        Add pi/4 Pauli-Y rotation gate
        """
    def add_sqrtYdag_gate(self, index: int) -> None: 
        """
        Add adjoint of pi/4 Pauli-Y rotation gate
        """
    def calculate_depth(self) -> int: 
        """
        Calculate depth of circuit
        """
    def copy(self) -> QuantumCircuit: 
        """
        Create copied instance
        """
    def get_gate(self, position: int) -> QuantumGateBase: 
        """
        Get gate instance
        """
    def get_gate_count(self) -> int: 
        """
        Get gate count
        """
    def get_qubit_count(self) -> int: 
        """
        Get qubit count
        """
    def merge_circuit(self, circuit: QuantumCircuit) -> None: ...
    def remove_gate(self, position: int) -> None: 
        """
        Remove gate
        """
    def to_string(self) -> str: 
        """
        Get string representation
        """
    @typing.overload
    def update_quantum_state(self, state: QuantumStateBase) -> None: 
        """
        Update quantum state

        Update quantum state
        """
    @typing.overload
    def update_quantum_state(self, state: QuantumStateBase, start: int, end: int) -> None: ...
    pass
class PauliOperator():
    def __IMUL__(self, arg0: complex) -> PauliOperator: ...
    def __imul__(self, arg0: PauliOperator) -> PauliOperator: ...
    @staticmethod
    @typing.overload
    def __init__(*args, **kwargs) -> typing.Any: 
        """
        Constructor

        Constructor

        Constructor
        """
    @typing.overload
    def __init__(self, coef: complex) -> None: ...
    @typing.overload
    def __init__(self, pauli_string: str, coef: complex) -> None: ...
    @typing.overload
    def __mul__(self, arg0: PauliOperator) -> PauliOperator: ...
    @typing.overload
    def __mul__(self, arg0: complex) -> PauliOperator: ...
    def add_single_Pauli(self, index: int, pauli_string: int) -> None: 
        """
        Add Pauli operator to this term
        """
    def change_coef(self, arg0: complex) -> None: 
        """
        change coefficient
        """
    def copy(self) -> PauliOperator: 
        """
        Create copied instance of Pauli operator class
        """
    def get_coef(self) -> complex: 
        """
        Get coefficient of Pauli term
        """
    def get_expectation_value(self, state: QuantumStateBase) -> complex: 
        """
        Get expectation value
        """
    def get_expectation_value_single_thread(self, state: QuantumStateBase) -> complex: 
        """
        Get expectation value
        """
    def get_index_list(self) -> typing.List[int]: 
        """
        Get list of target qubit indices
        """
    def get_pauli_id_list(self) -> typing.List[int]: 
        """
        Get list of Pauli IDs (I,X,Y,Z) = (0,1,2,3)
        """
    def get_pauli_string(self) -> str: 
        """
        get pauli string
        """
    def get_transition_amplitude(self, state_bra: QuantumStateBase, state_ket: QuantumStateBase) -> complex: 
        """
        Get transition amplitude
        """
    @staticmethod
    def get_x_bits(*args, **kwargs) -> typing.Any: 
        """
        get x bits
        """
    @staticmethod
    def get_z_bits(*args, **kwargs) -> typing.Any: 
        """
        get z bits
        """
    def update_quantum_state(self, arg0: QuantumStateBase) -> None: 
        """
        do update
        """
    pass
class ParametricQuantumCircuit(QuantumCircuit):
    def __init__(self, qubit_count: int) -> None: 
        """
        Constructor
        """
    def __repr__(self) -> str: ...
    @typing.overload
    def add_gate(self, gate: QuantumGateBase) -> None: 
        """
        Add gate

        Add gate
        """
    @typing.overload
    def add_gate(self, gate: QuantumGateBase, position: int) -> None: ...
    def add_parametric_RX_gate(self, index: int, angle: float) -> None: 
        """
        Add parametric Pauli-X rotation gate
        """
    def add_parametric_RY_gate(self, index: int, angle: float) -> None: 
        """
        Add parametric Pauli-Y rotation gate
        """
    def add_parametric_RZ_gate(self, index: int, angle: float) -> None: 
        """
        Add parametric Pauli-Z rotation gate
        """
    @typing.overload
    def add_parametric_gate(self, gate: QuantumGate_SingleParameter) -> None: 
        """
        Add parametric gate

        Add parametric gate
        """
    @typing.overload
    def add_parametric_gate(self, gate: QuantumGate_SingleParameter, position: int) -> None: ...
    def add_parametric_multi_Pauli_rotation_gate(self, index_list: typing.List[int], pauli_ids: typing.List[int], angle: float) -> None: 
        """
        Add parametric multi-qubit Pauli rotation gate
        """
    def backprop(self, obs: GeneralQuantumOperator) -> typing.List[float]: 
        """
        do backprop
        """
    def backprop_inner_product(self, state: QuantumState) -> typing.List[float]: 
        """
        do backprop with innder product
        """
    def copy(self) -> ParametricQuantumCircuit: 
        """
        Create copied instance
        """
    def get_parameter(self, index: int) -> float: 
        """
        Get parameter
        """
    def get_parameter_count(self) -> int: 
        """
        Get parameter count
        """
    def get_parametric_gate_position(self, index: int) -> int: 
        """
        Get parametric gate position
        """
    def remove_gate(self, position: int) -> None: 
        """
        Remove gate
        """
    def set_parameter(self, index: int, parameter: float) -> None: 
        """
        Set parameter
        """
    pass
class QuantumCircuitSimulator():
    def __init__(self, circuit: QuantumCircuit, state: QuantumStateBase) -> None: 
        """
        Constructor
        """
    def copy_state_from_buffer(self) -> None: 
        """
        Copy buffer to state
        """
    def copy_state_to_buffer(self) -> None: 
        """
        Copy state to buffer
        """
    def get_expectation_value(self, observable: Observable) -> complex: 
        """
        Get expectation value
        """
    def get_gate_count(self) -> int: 
        """
        Get gate count
        """
    def initialize_random_state(self) -> None: 
        """
        Initialize state with random pure state
        """
    def initialize_state(self, arg0: int) -> None: 
        """
        Initialize state
        """
    def simulate(self) -> None: 
        """
        Simulate circuit
        """
    def simulate_range(self, start: int, end: int) -> None: 
        """
        Simulate circuit
        """
    def swap_state_and_buffer(self) -> None: 
        """
        Swap state and buffer
        """
    pass
class QuantumGateBase():
    def __repr__(self) -> str: ...
    def copy(self) -> QuantumGateBase: 
        """
        Create copied instance
        """
    def get_control_index_list(self) -> typing.List[int]: 
        """
        Get control qubit index list
        """
    def get_matrix(self) -> numpy.ndarray[numpy.complex128, _Shape[m, n]]: 
        """
        Get gate matrix
        """
    def get_name(self) -> str: 
        """
        Get gate name
        """
    def get_target_index_list(self) -> typing.List[int]: 
        """
        Get target qubit index list
        """
    def is_Clifford(self) -> bool: 
        """
        Check this gate is element of Clifford group
        """
    def is_Gaussian(self) -> bool: 
        """
        Check this gate is element of Gaussian group
        """
    def is_Pauli(self) -> bool: 
        """
        Check this gate is element of Pauli group
        """
    def is_commute(self, gate: QuantumGateBase) -> bool: 
        """
        Check this gate commutes with a given gate
        """
    def is_diagonal(self) -> bool: 
        """
        Check the gate matrix is diagonal
        """
    def is_parametric(self) -> bool: 
        """
        Check this gate is parametric gate
        """
    def to_string(self) -> str: 
        """
        Get string representation
        """
    def update_quantum_state(self, state: QuantumStateBase) -> None: 
        """
        Update quantum state
        """
    pass
class QuantumGateMatrix(QuantumGateBase):
    def __repr__(self) -> str: ...
    def add_control_qubit(self, index: int, control_value: int) -> None: 
        """
        Add control qubit
        """
    def copy(self) -> QuantumGateBase: 
        """
        Create copied instance
        """
    def get_matrix(self) -> numpy.ndarray[numpy.complex128, _Shape[m, n]]: 
        """
        Get gate matrix
        """
    def multiply_scalar(self, value: complex) -> None: 
        """
        Multiply scalar value to gate matrix
        """
    def to_string(self) -> str: 
        """
        Get string representation
        """
    def update_quantum_state(self, state: QuantumStateBase) -> None: 
        """
        Update quantum state
        """
    pass
class QuantumGate_SingleParameter(QuantumGateBase):
    def copy(self) -> QuantumGate_SingleParameter: 
        """
        Create copied instance
        """
    def get_parameter_value(self) -> float: 
        """
        Get parameter value
        """
    def set_parameter_value(self, value: float) -> None: 
        """
        Set parameter value
        """
    pass
class QuantumState(QuantumStateBase):
    def __init__(self, qubit_count: int) -> None: 
        """
        Constructor
        """
    def __repr__(self) -> str: ...
    def add_state(self, state: QuantumStateBase) -> None: 
        """
        Add state vector to this state
        """
    def allocate_buffer(self) -> QuantumStateBase: 
        """
        Allocate buffer with the same size
        """
    def copy(self) -> QuantumStateBase: 
        """
        Create copied instance
        """
    def get_amplitude(self, index: int) -> complex: 
        """
        Get state vector
        """
    def get_classical_value(self, index: int) -> int: 
        """
        Get classical value
        """
    def get_device_name(self) -> str: 
        """
        Get allocated device name
        """
    def get_entropy(self) -> float: 
        """
        Get entropy
        """
    def get_marginal_probability(self, measured_value: typing.List[int]) -> float: 
        """
        Get merginal probability for measured values
        """
    def get_qubit_count(self) -> int: 
        """
        Get qubit count
        """
    def get_squared_norm(self) -> float: 
        """
        Get squared norm
        """
    def get_vector(self) -> numpy.ndarray[numpy.complex128, _Shape[m, 1]]: 
        """
        Get state vector
        """
    def get_zero_probability(self, index: int) -> float: 
        """
        Get probability with which we obtain 0 when we measure a qubit
        """
    @typing.overload
    def load(self, state: QuantumStateBase) -> None: 
        """
        Load quantum state vector

        Load quantum state vector
        """
    @typing.overload
    def load(self, state: typing.List[complex]) -> None: ...
    def multiply_coef(self, coef: complex) -> None: 
        """
        Multiply coefficient to this state
        """
    def multiply_elementwise_function(self, func: typing.Callable[[int], complex]) -> None: 
        """
        Multiply elementwise function
        """
    def normalize(self, squared_norm: float) -> None: 
        """
        Normalize quantum state
        """
    @typing.overload
    def sampling(self, count: int) -> typing.List[int]: 
        """
        Sampling measurement results

        Sampling measurement results
        """
    @typing.overload
    def sampling(self, count: int, seed: int) -> typing.List[int]: ...
    @typing.overload
    def set_Haar_random_state(self) -> None: 
        """
        Set Haar random state

        Set Haar random state
        """
    @typing.overload
    def set_Haar_random_state(self, seed: int) -> None: ...
    def set_classical_value(self, index: int, value: int) -> None: 
        """
        Set classical value
        """
    def set_computational_basis(self, index: int) -> None: 
        """
        Set state to computational basis
        """
    def set_zero_state(self) -> None: 
        """
        Set state to |0>
        """
    def to_string(self) -> str: 
        """
        Get string representation
        """
    pass
class DensityMatrix(QuantumStateBase):
    def __init__(self, qubit_count: int) -> None: 
        """
        Constructor
        """
    def __repr__(self) -> str: ...
    def add_state(self, state: QuantumStateBase) -> None: 
        """
        Add state vector to this state
        """
    def allocate_buffer(self) -> QuantumStateBase: 
        """
        Allocate buffer with the same size
        """
    def copy(self) -> QuantumStateBase: 
        """
        Create copied insntace
        """
    def get_classical_value(self, index: int) -> int: 
        """
        Get classical value
        """
    def get_device_name(self) -> str: 
        """
        Get allocated device name
        """
    def get_entropy(self) -> float: 
        """
        Get entropy
        """
    def get_marginal_probability(self, measured_value: typing.List[int]) -> float: 
        """
        Get merginal probability for measured values
        """
    def get_matrix(self) -> numpy.ndarray[numpy.complex128, _Shape[m, n]]: 
        """
        Get density matrix
        """
    def get_squared_norm(self) -> float: 
        """
        Get squared norm
        """
    def get_zero_probability(self, index: int) -> float: 
        """
        Get probability with which we obtain 0 when we measure a qubit
        """
    @typing.overload
    def load(self, state: QuantumStateBase) -> None: 
        """
        Load quantum state vector

        Load quantum state vector or density matrix

        Load density matrix
        """
    @typing.overload
    def load(self, state: numpy.ndarray[numpy.complex128, _Shape[m, n]]) -> None: ...
    @typing.overload
    def load(self, state: typing.List[complex]) -> None: ...
    def multiply_coef(self, coef: complex) -> None: 
        """
        Multiply coefficient to this state
        """
    def normalize(self, squared_norm: float) -> None: 
        """
        Normalize quantum state
        """
    @typing.overload
    def sampling(self, count: int) -> typing.List[int]: 
        """
        Sampling measurement results

        Sampling measurement results
        """
    @typing.overload
    def sampling(self, count: int, seed: int) -> typing.List[int]: ...
    @typing.overload
    def set_Haar_random_state(self) -> None: 
        """
        Set Haar random state

        Set Haar random state
        """
    @typing.overload
    def set_Haar_random_state(self, seed: int) -> None: ...
    def set_classical_value(self, index: int, value: int) -> None: 
        """
        Set classical value
        """
    def set_computational_basis(self, index: int) -> None: 
        """
        Set state to computational basis
        """
    def set_zero_state(self) -> None: 
        """
        Set state to |0>
        """
    def to_string(self) -> str: 
        """
        Get string representation
        """
    pass
def StateVector(arg0: int) -> QuantumState:
    """
    StateVector
    """
