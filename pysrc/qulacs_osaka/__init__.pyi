"""cppsim python interface"""
from __future__ import annotations
import qulacs_osaka_core
import typing
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "Causal",
    "DensityMatrix",
    "FermionOperator",
    "NoiseSimulator",
    "Observable",
    "PauliOperator",
    "QuantumCircuit",
    "QuantumGateBase",
    "QuantumGateBasic",
    "QuantumGateWrapped",
    "QuantumStateBase",
    "SingleFermionOperator",
    "StateVector",
    "StateVectorCpu",
    "gate",
    "state",
    "transforms"
]


class Causal():
    def CausalCone(self, arg0: QuantumCircuit, arg1: Observable) -> complex: ...
    def __init__(self) -> None: 
        """
        Constructor
        """
    pass
class QuantumStateBase():
    pass
class FermionOperator():
    def __IMUL__(self, arg0: complex) -> FermionOperator: ...
    def __add__(self, arg0: FermionOperator) -> FermionOperator: ...
    def __iadd__(self, arg0: FermionOperator) -> FermionOperator: ...
    def __imul__(self, arg0: FermionOperator) -> FermionOperator: ...
    def __init__(self) -> None: 
        """
        Constructor
        """
    def __isub__(self, arg0: FermionOperator) -> FermionOperator: ...
    @typing.overload
    def __mul__(self, arg0: FermionOperator) -> FermionOperator: ...
    @typing.overload
    def __mul__(self, arg0: complex) -> FermionOperator: ...
    def __sub__(self, arg0: FermionOperator) -> FermionOperator: ...
    @typing.overload
    def add_term(self, coef: complex, action_string: str) -> None: 
        """
        Add Fermion operator

        Add Fermion operator
        """
    @typing.overload
    def add_term(self, coef: complex, fermion_operator: SingleFermionOperator) -> None: ...
    def copy(self) -> FermionOperator: 
        """
        Make copy
        """
    def get_coef_list(self) -> typing.List[complex]: 
        """
        Get coef list
        """
    def get_fermion_list(self) -> typing.List[SingleFermionOperator]: 
        """
        Get term(SingleFermionOperator) list
        """
    def get_term(self, index: int) -> typing.Tuple[complex, SingleFermionOperator]: 
        """
        Get a Fermion term
        """
    def get_term_count(self) -> int: 
        """
        Get count of Fermion terms
        """
    pass
class NoiseSimulator():
    def __init__(self, arg0: QuantumCircuit, arg1: StateVectorCpu) -> None: 
        """
        Constructor
        """
    def execute(self, arg0: int) -> typing.List[int]: 
        """
        sampling & return result [array]
        """
    pass
class Observable():
    def __IMUL__(self, arg0: complex) -> Observable: ...
    def __add__(self, arg0: Observable) -> Observable: ...
    def __iadd__(self, arg0: Observable) -> Observable: ...
    def __imul__(self, arg0: Observable) -> Observable: ...
    def __init__(self) -> None: 
        """
        Constructor
        """
    def __isub__(self, arg0: Observable) -> Observable: ...
    @typing.overload
    def __mul__(self, arg0: Observable) -> Observable: ...
    @typing.overload
    def __mul__(self, arg0: complex) -> Observable: ...
    def __str__(self) -> str: 
        """
        to string
        """
    def __sub__(self, arg0: Observable) -> Observable: ...
    @typing.overload
    def add_term(self, coef: complex, pauli_operator: PauliOperator) -> None: 
        """
        Add Pauli operator

        Add Pauli operator
        """
    @typing.overload
    def add_term(self, coef: complex, pauli_string: str) -> None: ...
    def copy(self) -> Observable: 
        """
        Make copy
        """
    def get_expectation_value(self, state: QuantumStateBase) -> complex: 
        """
        Get expectation value
        """
    def get_term(self, index: int) -> typing.Tuple[complex, PauliOperator]: 
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
    pass
class PauliOperator():
    def __eq__(self, arg0: PauliOperator) -> bool: ...
    def __imul__(self, arg0: PauliOperator) -> PauliOperator: ...
    @typing.overload
    def __init__(self) -> None: 
        """
        Constructor

        Constructor

        Constructor
        """
    @typing.overload
    def __init__(self, pauli_string: str) -> None: ...
    @typing.overload
    def __init__(self, qubit_index: typing.List[int], pauli_id: typing.List[int]) -> None: ...
    def __mul__(self, arg0: PauliOperator) -> PauliOperator: ...
    def __str__(self) -> str: 
        """
        to string
        """
    def add_single_Pauli(self, index: int, pauli_string: int) -> None: 
        """
        Add Pauli operator to this term
        """
    def copy(self) -> PauliOperator: 
        """
        Make copy
        """
    def get_expectation_value(self, state: QuantumStateBase) -> complex: 
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
    def get_transition_amplitude(self, state_bra: QuantumStateBase, state_ket: QuantumStateBase) -> complex: 
        """
        Get transition amplitude
        """
    __hash__ = None
    pass
class QuantumCircuit():
    def __init__(self, qubit_count: int) -> None: 
        """
        Constructor
        """
    def __repr__(self) -> str: ...
    @typing.overload
    def add_gate(self, gate: QuantumGateBase) -> None: 
        """
        Add gate with copy

        Add gate with copy
        """
    @typing.overload
    def add_gate(self, gate: QuantumGateBase, position: int) -> None: ...
    def calculate_depth(self) -> int: 
        """
        Calculate depth of circuit
        """
    def copy(self) -> QuantumCircuit: 
        """
        Create copied instance
        """
    def dump_as_byte(self) -> bytes: 
        """
        Seralize object as byte
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
    def load_from_byte(self, arg0: str) -> None: 
        """
        Deseralize object as byte
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
class QuantumGateBase():
    def __repr__(self) -> str: ...
    def copy(self) -> QuantumGateBase: 
        """
        Create copied instance
        """
    def dump_as_byte(self) -> bytes: 
        """
        Seralize object as byte
        """
    def get_control_index_list(self) -> typing.List[int]: 
        """
        Get control qubit index list
        """
    def get_matrix(self) -> numpy.ndarray[numpy.complex128, _Shape[m, n]]: 
        """
        Get gate matrix
        """
    def get_target_index_list(self) -> typing.List[int]: 
        """
        Get target qubit index list
        """
    def load_from_byte(self, arg0: str) -> None: 
        """
        Deseralize object as byte
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
class QuantumGateBasic(QuantumGateBase):
    def __repr__(self) -> str: ...
    def add_control_qubit(self, index: int, control_value: int) -> None: 
        """
        Add control qubit
        """
    def copy(self) -> QuantumGateBase: 
        """
        Create copied instance
        """
    def dump_as_byte(self) -> bytes: 
        """
        Seralize object as byte
        """
    def get_matrix(self) -> numpy.ndarray[numpy.complex128, _Shape[m, n]]: 
        """
        Get gate matrix
        """
    def load_from_byte(self, arg0: str) -> None: 
        """
        Deseralize object as byte
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
class QuantumGateWrapped(QuantumGateBase):
    def __repr__(self) -> str: ...
    def copy(self) -> QuantumGateBase: 
        """
        Create copied instance
        """
    def dump_as_byte(self) -> bytes: 
        """
        Seralize object as byte
        """
    def get_gate(self, index: int) -> QuantumGateBase: 
        """
        Get Kraus operator
        """
    def get_gate_count(self) -> int: 
        """
        Get the number of Kraus operators
        """
    def load_from_byte(self, arg0: str) -> None: 
        """
        Deseralize object as byte
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
    def get_classical_value(self, index: str) -> int: 
        """
        Get classical value
        """
    def get_device_type(self) -> DeviceType: 
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
    def set_classical_value(self, index: str, value: int) -> None: 
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
class SingleFermionOperator():
    def __imul__(self, arg0: SingleFermionOperator) -> SingleFermionOperator: ...
    @typing.overload
    def __init__(self) -> None: 
        """
        Constructor

        Constructor

        Constructor
        """
    @typing.overload
    def __init__(self, action_string: str) -> None: ...
    @typing.overload
    def __init__(self, target_index_list: typing.List[int], action_id_list: typing.List[int]) -> None: ...
    def __mul__(self, arg0: SingleFermionOperator) -> SingleFermionOperator: ...
    def __str__(self) -> str: 
        """
        to string
        """
    def get_action_id_list(self) -> typing.List[int]: 
        """
        Get list of action IDs (Create action: 1, Destroy action: 0)
        """
    def get_target_index_list(self) -> typing.List[int]: 
        """
        Get list of target indices
        """
    pass
class StateVectorCpu(QuantumStateBase):
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
    def get_classical_value(self, index: str) -> int: 
        """
        Get classical value
        """
    def get_device_type(self) -> DeviceType: 
        """
        Get allocated device type
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
    def set_classical_value(self, index: str, value: int) -> None: 
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
def StateVector(arg0: int) -> StateVectorCpu:
    """
    StateVector
    """
