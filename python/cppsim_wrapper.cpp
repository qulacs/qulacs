
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#ifndef _MSC_VER
extern "C" {
#include <csim/update_ops.h>
#include <csim/memory_ops.h>
#include <csim/stat_ops.h>
}
#else
#include <csim/update_ops.h>
#include <csim/memory_ops.h>
#include <csim/stat_ops.h>
#endif

#include <cppsim/observable.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/general_quantum_operator.hpp>
#include <cppsim/state.hpp>
#include <cppsim/state_dm.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/circuit_optimizer.hpp>
#include <cppsim/simulator.hpp>

#ifdef _USE_GPU
#include <cppsim/state_gpu.hpp>
#endif

#include <vqcsim/parametric_gate.hpp>
#include <vqcsim/parametric_gate_factory.hpp>
#include <vqcsim/parametric_circuit.hpp>

namespace py = pybind11;
PYBIND11_MODULE(qulacs, m) {
    m.doc() = "cppsim python interface";

    py::class_<PauliOperator>(m, "PauliOperator")
        .def(py::init<std::complex<double>>(), "Constructor", py::arg("coef"))
        .def(py::init<std::string, std::complex<double>>(), "Constructor", py::arg("pauli_string"), py::arg("coef"))
        //.def(py::init<std::vector<unsigned int>&, std::string, std::complex<double>>(), "Constructor")
        //.def(py::init<std::vector<unsigned int>&, std::vector<unsigned int>&, std::complex<double>>(), "Constructor")
        //.def(py::init<std::vector<unsigned int>&, std::complex<double>>(), "Constructor")
        .def("get_index_list", &PauliOperator::get_index_list,"Get list of target qubit indices")
        .def("get_pauli_id_list", &PauliOperator::get_pauli_id_list, "Get list of Pauli IDs (I,X,Y,Z) = (0,1,2,3)")
        .def("get_coef", &PauliOperator::get_coef, "Get coefficient of Pauli term")
        .def("add_single_Pauli", &PauliOperator::add_single_Pauli, "Add Pauli operator to this term", py::arg("index"), py::arg("pauli_string"))
        .def("get_expectation_value", &PauliOperator::get_expectation_value, "Get expectation value", py::arg("state"))
        .def("get_transition_amplitude", &PauliOperator::get_transition_amplitude, "Get transition amplitude", py::arg("state_bra"), py::arg("state_ket"))
        .def("copy", &PauliOperator::copy, pybind11::return_value_policy::take_ownership, "Create copied instance of Pauli operator class")
        ;

    py::class_<GeneralQuantumOperator>(m, "GeneralQuantumOperator")
        .def(py::init<unsigned int>(), "Constructor", py::arg("qubit_count"))
        // .def(py::init<std::string>())
        .def("add_operator", (void (GeneralQuantumOperator::*)(const PauliOperator*)) &GeneralQuantumOperator::add_operator, "Add Pauli operator", py::arg("pauli_operator"))
        .def("add_operator", (void (GeneralQuantumOperator::*)(std::complex<double> coef, std::string))&GeneralQuantumOperator::add_operator, "Add Pauli operator", py::arg("coef"), py::arg("pauli_string"))
        .def("is_hermitian", &GeneralQuantumOperator::is_hermitian, "Get is Herimitian")
        .def("get_qubit_count", &GeneralQuantumOperator::get_qubit_count, "Get qubit count")
        .def("get_state_dim", &GeneralQuantumOperator::get_state_dim, "Get state dimension")
        .def("get_term_count", &GeneralQuantumOperator::get_term_count, "Get count of Pauli terms")
        //.def("get_term", &GeneralQuantumOperator::get_term, pybind11::return_value_policy::take_ownership)
        .def("get_term",[](const GeneralQuantumOperator& quantum_operator, const unsigned int index) {
            return quantum_operator.get_term(index)->copy();
        }, pybind11::return_value_policy::take_ownership, "Get Pauli term", py::arg("index"))
        .def("get_expectation_value", &GeneralQuantumOperator::get_expectation_value, "Get expectation value", py::arg("state"))
        .def("get_transition_amplitude", &GeneralQuantumOperator::get_transition_amplitude, "Get transition amplitude", py::arg("state_bra"), py::arg("state_ket"))
        //.def_static("get_split_GeneralQuantumOperator", &(GeneralQuantumOperator::get_split_observable));
        ;
    auto mquantum_operator = m.def_submodule("quantum_operator");
    mquantum_operator.def("create_quantum_operator_from_openfermion_file", &quantum_operator::create_general_quantum_operator_from_openfermion_file, pybind11::return_value_policy::take_ownership);
    mquantum_operator.def("create_quantum_operator_from_openfermion_text", &quantum_operator::create_general_quantum_operator_from_openfermion_text, pybind11::return_value_policy::take_ownership);
    mquantum_operator.def("create_split_quantum_operator", &quantum_operator::create_split_general_quantum_operator, pybind11::return_value_policy::take_ownership);

    py::class_<HermitianQuantumOperator, GeneralQuantumOperator>(m, "Observable")
        .def(py::init<unsigned int>(), "Constructor", py::arg("qubit_count"))
        // .def(py::init<std::string>())
        .def("add_operator", (void (HermitianQuantumOperator::*)(const PauliOperator*)) &HermitianQuantumOperator::add_operator, "Add Pauli operator", py::arg("pauli_operator"))
        .def("add_operator", (void (HermitianQuantumOperator::*)(std::complex<double> coef, std::string))&HermitianQuantumOperator::add_operator, "Add Pauli operator", py::arg("coef"), py::arg("string"))
        .def("get_qubit_count", &HermitianQuantumOperator::get_qubit_count, "Get qubit count")
        .def("get_state_dim", &HermitianQuantumOperator::get_state_dim, "Get state dimension")
        .def("get_term_count", &HermitianQuantumOperator::get_term_count, "Get count of Pauli terms")
        //.def("get_term", &HermitianQuantumOperator::get_term, pybind11::return_value_policy::take_ownership)
        .def("get_term",[](const HermitianQuantumOperator& quantum_operator, const unsigned int index) {
            return quantum_operator.get_term(index)->copy();
        }, pybind11::return_value_policy::take_ownership, "Get Pauli term", py::arg("index"))
        // .def("get_expectation_value", &HermitianQuantumOperator::get_expectation_value)
        .def("get_expectation_value", [](const HermitianQuantumOperator& observable, const QuantumStateBase* state) {
                                          double res = observable.get_expectation_value(state).real();
                                          return res;}, "Get expectation value", py::arg("state"))
        .def("get_transition_amplitude", &HermitianQuantumOperator::get_transition_amplitude, "Get transition amplitude", py::arg("state_bra"), py::arg("state_ket"))
        //.def_static("get_split_Observable", &(HermitianQuantumOperator::get_split_observable));
        ;
    auto mobservable = m.def_submodule("observable");
    mobservable.def("create_observable_from_openfermion_file", &observable::create_observable_from_openfermion_file, pybind11::return_value_policy::take_ownership);
    mobservable.def("create_observable_from_openfermion_text", &observable::create_observable_from_openfermion_text, pybind11::return_value_policy::take_ownership);
    mobservable.def("create_split_observable", &observable::create_split_observable, pybind11::return_value_policy::take_ownership);


    py::class_<QuantumStateBase>(m, "QuantumStateBase");
    py::class_<QuantumState, QuantumStateBase>(m, "QuantumState")
        .def(py::init<unsigned int>(), "Constructor", py::arg("qubit_count"))
        .def("set_zero_state", &QuantumState::set_zero_state, "Set state to |0>")
        .def("set_computational_basis", &QuantumState::set_computational_basis, "Set state to computational basis", py::arg("index"))
        .def("set_Haar_random_state", (void (QuantumState::*)(void))&QuantumState::set_Haar_random_state, "Set Haar random state")
        .def("set_Haar_random_state", (void (QuantumState::*)(UINT))&QuantumState::set_Haar_random_state, "Set Haar random state", py::arg("seed"))
        .def("get_zero_probability", &QuantumState::get_zero_probability, "Get probability with which we obtain 0 when we measure a qubit", py::arg("index"))
		.def("get_marginal_probability", &QuantumState::get_marginal_probability, "Get merginal probability for measured values", py::arg("measured_value"))
		.def("get_entropy", &QuantumState::get_entropy, "Get entropy")
        .def("get_squared_norm", &QuantumState::get_squared_norm, "Get squared norm")
        .def("normalize", &QuantumState::normalize, "Normalize quantum state", py::arg("squared_norm"))
        .def("allocate_buffer", &QuantumState::allocate_buffer, pybind11::return_value_policy::automatic_reference, "Allocate buffer with the same size")
		//.def("copy", &QuantumState::copy, pybind11::return_value_policy::automatic_reference)
        .def("copy", &QuantumState::copy, "Create copied instance")
        .def("load", (void (QuantumState::*)(const QuantumStateBase*))&QuantumState::load, "Load quantum state vector", py::arg("state"))
        .def("load", (void (QuantumState::*)(const std::vector<CPPCTYPE>&))&QuantumState::load, "Load quantum state vector", py::arg("state"))
		.def("get_device_name", &QuantumState::get_device_name, "Get allocated device name")
        //.def("data_cpp", &QuantumState::data_cpp)
        //.def("data_c", &QuantumState::data_c)
        .def("add_state", &QuantumState::add_state, "Add state vector to this state", py::arg("state"))
        .def("multiply_coef", &QuantumState::multiply_coef, "Multiply coefficient to this state", py::arg("coef"))
        .def("multiply_elementwise_function", &QuantumState::multiply_elementwise_function, "Multiply elementwise function", py::arg("func"))
        .def("get_classical_value", &QuantumState::get_classical_value, "Get classical value", py::arg("index"))
        .def("set_classical_value", &QuantumState::set_classical_value, "Set classical value", py::arg("index"), py::arg("value"))
        .def("to_string",&QuantumState::to_string, "Get string representation")
        .def("sampling", (std::vector<ITYPE> (QuantumState::*)(UINT))&QuantumState::sampling, "Sampling measurement results", py::arg("count"))
		.def("sampling", (std::vector<ITYPE>(QuantumState::*)(UINT, UINT))&QuantumState::sampling, "Sampling measurement results", py::arg("count"), py::arg("seed"))

        .def("get_vector", [](const QuantumState& state) {
        Eigen::VectorXcd vec = Eigen::Map<Eigen::VectorXcd>(state.data_cpp(), state.dim);
        return vec;
        }, "Get state vector")
        .def("get_qubit_count", [](const QuantumState& state) -> unsigned int {return (unsigned int) state.qubit_count; }, "Get qubit count")
        .def("__repr__", [](const QuantumState &p) {return p.to_string();});
        ;

		m.def("StateVector", [](const unsigned int qubit_count) {
			auto ptr = new QuantumState(qubit_count);
			return ptr;
		}, "StateVector");

	py::class_<DensityMatrix, QuantumStateBase>(m, "DensityMatrix")
		.def(py::init<unsigned int>(), "Constructor", py::arg("qubit_count"))
		.def("set_zero_state", &DensityMatrix::set_zero_state, "Set state to |0>")
		.def("set_computational_basis", &DensityMatrix::set_computational_basis, "Set state to computational basis", py::arg("index"))
		.def("set_Haar_random_state", (void (DensityMatrix::*)(void))&DensityMatrix::set_Haar_random_state, "Set Haar random state")
		.def("set_Haar_random_state", (void (DensityMatrix::*)(UINT))&DensityMatrix::set_Haar_random_state, "Set Haar random state", py::arg("seed"))
		.def("get_zero_probability", &DensityMatrix::get_zero_probability, "Get probability with which we obtain 0 when we measure a qubit", py::arg("index"))
		.def("get_marginal_probability", &DensityMatrix::get_marginal_probability, "Get merginal probability for measured values", py::arg("measured_value"))
		.def("get_entropy", &DensityMatrix::get_entropy, "Get entropy")
		.def("get_squared_norm", &DensityMatrix::get_squared_norm, "Get squared norm")
		.def("normalize", &DensityMatrix::normalize, "Normalize quantum state", py::arg("squared_norm"))
		.def("allocate_buffer", &DensityMatrix::allocate_buffer, pybind11::return_value_policy::automatic_reference, "Allocate buffer with the same size")
		.def("copy", &DensityMatrix::copy, "Create copied insntace")
		.def("load", (void (DensityMatrix::*)(const QuantumStateBase*))&DensityMatrix::load, "Load quantum state vector", py::arg("state"))
		.def("load", (void (DensityMatrix::*)(const std::vector<CPPCTYPE>&))&DensityMatrix::load, "Load quantum state vector or density matrix", py::arg("state"))
		.def("load", (void (DensityMatrix::*)(const ComplexMatrix&))&DensityMatrix::load, "Load density matrix", py::arg("state"))
		.def("get_device_name", &DensityMatrix::get_device_name, "Get allocated device name")
		//.def("data_cpp", &DensityMatrix::data_cpp)
		//.def("data_c", &DensityMatrix::data_c)
		.def("add_state", &DensityMatrix::add_state, "Add state vector to this state", py::arg("state"))
		.def("multiply_coef", &DensityMatrix::multiply_coef, "Multiply coefficient to this state", py::arg("coef"))
		.def("get_classical_value", &DensityMatrix::get_classical_value, "Get classical value", py::arg("index"))
		.def("set_classical_value", &DensityMatrix::set_classical_value, "Set classical value", py::arg("index"), py::arg("value"))
		.def("to_string", &QuantumState::to_string, "Get string representation")
		.def("sampling", (std::vector<ITYPE>(DensityMatrix::*)(UINT))&DensityMatrix::sampling, "Sampling measurement results", py::arg("count"))
		.def("sampling", (std::vector<ITYPE>(DensityMatrix::*)(UINT, UINT))&DensityMatrix::sampling, "Sampling measurement results", py::arg("count"), py::arg("seed"))

		.def("get_matrix", [](const DensityMatrix& state) {
			Eigen::MatrixXcd mat(state.dim, state.dim);
			CTYPE* ptr = state.data_c();
			for (ITYPE y = 0; y < state.dim; ++y) {
				for (ITYPE x = 0; x < state.dim; ++x) {
					mat(y, x) = ptr[y*state.dim + x];
				}
			}
			return mat;
		}, "Get density matrix")
		.def("__repr__", [](const DensityMatrix &p) {return p.to_string(); });
		;

#ifdef _USE_GPU
    py::class_<QuantumStateGpu, QuantumStateBase>(m, "QuantumStateGpu")
        .def(py::init<unsigned int>(), "Constructor", py::arg("qubit_count"))
        .def(py::init<unsigned int, unsigned int>(), "Constructor", py::arg("qubit_count"), py::arg("gpu_id"))
        .def("set_zero_state", &QuantumStateGpu::set_zero_state, "Set state to |0>")
        .def("set_computational_basis", &QuantumStateGpu::set_computational_basis, "Set state to computational basis", py::arg("index"))
        .def("set_Haar_random_state", (void (QuantumStateGpu::*)(void))&QuantumStateGpu::set_Haar_random_state, "Set Haar random state")
        .def("set_Haar_random_state", (void (QuantumStateGpu::*)(UINT))&QuantumStateGpu::set_Haar_random_state, "Set Haar random state", py::arg("seed"))
        .def("get_zero_probability", &QuantumStateGpu::get_zero_probability, "Get probability with which we obtain 0 when we measure a qubit", py::arg("index"))
        .def("get_marginal_probability", &QuantumStateGpu::get_marginal_probability, "Get merginal probability for measured values", py::arg("measured_value"))
        .def("get_entropy", &QuantumStateGpu::get_entropy, "Get entropy")
        .def("get_squared_norm", &QuantumStateGpu::get_squared_norm, "Get squared norm")
        .def("normalize", &QuantumStateGpu::normalize, "Normalize quantum state", py::arg("squared_norm"))
        .def("allocate_buffer", &QuantumStateGpu::allocate_buffer, pybind11::return_value_policy::automatic_reference, "Allocate buffer with the same size")
        //.def("copy", &QuantumStateGpu::copy, pybind11::return_value_policy::automatic_reference)
        .def("copy", &QuantumStateGpu::copy, "Create copied insntace")
        .def("load", (void (QuantumStateGpu::*)(const QuantumStateBase*))&QuantumStateGpu::load, "Load quantum state vector", py::arg("state"))
        .def("load", (void (QuantumStateGpu::*)(const std::vector<CPPCTYPE>&))&QuantumStateGpu::load, "Load quantum state vector", py::arg("state"))
        .def("get_device_name", &QuantumStateGpu::get_device_name, "Get allocated device name")
        //.def("data_cpp", &QuantumStateGpu::data_cpp)
        //.def("data_c", &QuantumStateGpu::data_c)
        .def("add_state", &QuantumStateGpu::add_state, "Add state vector to this state", py::arg("state"))
        .def("multiply_coef", &QuantumStateGpu::multiply_coef, "Multiply coefficient to this state", py::arg("coef"))
        .def("multiply_elementwise_function", &QuantumStateGpu::multiply_elementwise_function, "Multiply elementwise function", py::arg("func"))
        .def("get_classical_value", &QuantumStateGpu::get_classical_value, "Get classical value", py::arg("index"))
        .def("set_classical_value", &QuantumStateGpu::set_classical_value, "Set classical value", py::arg("index"), py::arg("value"))
        .def("to_string", &QuantumStateGpu::to_string, "Get string representation")
		.def("sampling", (std::vector<ITYPE>(QuantumStateGpu::*)(UINT))&QuantumStateGpu::sampling, "Sampling measurement results", py::arg("count"))
		.def("sampling", (std::vector<ITYPE>(QuantumStateGpu::*)(UINT, UINT))&QuantumStateGpu::sampling, "Sampling measurement results", py::arg("count"), py::arg("seed"))
		.def("get_vector", [](const QuantumStateGpu& state) {
            Eigen::VectorXcd vec = Eigen::Map<Eigen::VectorXcd>(state.duplicate_data_cpp(), state.dim);
            return vec;
        }, pybind11::return_value_policy::take_ownership, "Get state vector")
        .def("get_qubit_count", [](const QuantumStateGpu& state) -> unsigned int {return (unsigned int) state.qubit_count; }, "Get qubit count")
        .def("__repr__", [](const QuantumStateGpu &p) {return p.to_string(); });
        ;
		m.def("StateVectorGpu", [](const unsigned int qubit_count) {
			auto ptr = new QuantumStateGpu(qubit_count);
			return ptr;
		}, "StateVectorGpu");

#endif

    auto mstate = m.def_submodule("state");
    //using namespace state;
#ifdef _USE_GPU
    mstate.def("inner_product", py::overload_cast<const QuantumStateGpu*, const QuantumStateGpu*>(&state::inner_product), "Get inner product", py::arg("state_bra"), py::arg("state_ket"));
#endif
    mstate.def("inner_product", py::overload_cast<const QuantumState*, const QuantumState*>(&state::inner_product), "Get inner product", py::arg("state_bra"), py::arg("state_ket"));
    //mstate.def("inner_product", &state::inner_product);
    //mstate.def("tensor_product", &state::tensor_product, pybind11::return_value_policy::take_ownership, "Get tensor product of states", py::arg("state_left"), py::arg("state_right"));
    //mstate.def("permutate_qubit", &state::permutate_qubit, pybind11::return_value_policy::take_ownership, "Permutate qubits from state", py::arg("state"), py::arg("order"));
    mstate.def("tensor_product", py::overload_cast<const QuantumState*, const QuantumState*>(&state::tensor_product), pybind11::return_value_policy::take_ownership, "Get tensor product of states", py::arg("state_left"), py::arg("state_right"));
    mstate.def("tensor_product", py::overload_cast<const DensityMatrix*, const DensityMatrix*>(&state::tensor_product), pybind11::return_value_policy::take_ownership, "Get tensor product of states", py::arg("state_left"), py::arg("state_right"));
    mstate.def("permutate_qubit", py::overload_cast<const QuantumState*, std::vector<UINT>>(&state::permutate_qubit), pybind11::return_value_policy::take_ownership, "Permutate qubits from state", py::arg("state"), py::arg("order"));
    mstate.def("permutate_qubit", py::overload_cast<const DensityMatrix*, std::vector<UINT>>(&state::permutate_qubit), pybind11::return_value_policy::take_ownership, "Permutate qubits from state", py::arg("state"), py::arg("order"));

    mstate.def("drop_qubit", &state::drop_qubit, pybind11::return_value_policy::take_ownership, "Drop qubits from state", py::arg("state"), py::arg("target"), py::arg("projection"));
    mstate.def("partial_trace", py::overload_cast<const QuantumState*, std::vector<UINT>>(&state::partial_trace), pybind11::return_value_policy::take_ownership, "Take partial trace", py::arg("state"), py::arg("target_traceout"));
    mstate.def("partial_trace", py::overload_cast<const DensityMatrix*, std::vector<UINT>>(&state::partial_trace), pybind11::return_value_policy::take_ownership, "Take partial trace", py::arg("state"), py::arg("target_traceout"));

    py::class_<QuantumGateBase>(m, "QuantumGateBase")
        .def("update_quantum_state", &QuantumGateBase::update_quantum_state, "Update quantum state", py::arg("state"))
        .def("copy",&QuantumGateBase::copy, pybind11::return_value_policy::take_ownership, "Create copied instance")
        .def("to_string", &QuantumGateBase::to_string, "Get string representation")
        .def("get_matrix", [](const QuantumGateBase& gate) {
            ComplexMatrix mat;
            gate.set_matrix(mat);
            return mat;
        }, "Get gate matrix")
        .def("__repr__", [](const QuantumGateBase &p) {return p.to_string(); })
        .def("get_target_index_list", &QuantumGateBase::get_target_index_list, "Get target qubit index list")
        .def("get_control_index_list", &QuantumGateBase::get_control_index_list, "Get control qubit index list")
        .def("get_name", &QuantumGateBase::get_name, "Get gate name")
        .def("get_angle", &QuantumGateBase::get_angle, "Get gate angle")
        .def("is_commute", &QuantumGateBase::is_commute, "Check this gate commutes with a given gate", py::arg("gate"))
        .def("is_Pauli", &QuantumGateBase::is_Pauli, "Check this gate is element of Pauli group")
        .def("is_Clifford", &QuantumGateBase::is_Clifford, "Check this gate is element of Clifford group")
        .def("is_Gaussian", &QuantumGateBase::is_Gaussian, "Check this gate is element of Gaussian group")
        .def("is_parametric", &QuantumGateBase::is_parametric, "Check this gate is parametric gate")
        .def("is_diagonal", &QuantumGateBase::is_diagonal, "Check the gate matrix is diagonal")
        ;

    py::class_<QuantumGateMatrix,QuantumGateBase>(m, "QuantumGateMatrix")
        .def("update_quantum_state", &QuantumGateMatrix::update_quantum_state, "Update quantum state", py::arg("state"))
        .def("add_control_qubit", &QuantumGateMatrix::add_control_qubit, "Add control qubit", py::arg("index"), py::arg("control_value"))
        .def("multiply_scalar", &QuantumGateMatrix::multiply_scalar, "Multiply scalar value to gate matrix", py::arg("value"))
        .def("copy", &QuantumGateMatrix::copy, pybind11::return_value_policy::take_ownership, "Create copied instance")
        .def("to_string", &QuantumGateMatrix::to_string, "Get string representation")

        .def("get_matrix", [](const QuantumGateMatrix& gate) {
        ComplexMatrix mat;
        gate.set_matrix(mat);
        return mat;
        }, "Get gate matrix")
        .def("__repr__", [](const QuantumGateMatrix &p) {return p.to_string(); });
        ;

    auto mgate = m.def_submodule("gate");
    mgate.def("Identity", &gate::Identity, pybind11::return_value_policy::take_ownership, "Create identity gate", py::arg("index"));
    mgate.def("X", &gate::X, pybind11::return_value_policy::take_ownership, "Create Pauli-X gate", py::arg("index"));
    mgate.def("Y", &gate::Y, pybind11::return_value_policy::take_ownership, "Create Pauli-Y gate", py::arg("index"));
    mgate.def("Z", &gate::Z, pybind11::return_value_policy::take_ownership, "Create Pauli-Z gate", py::arg("index"));
    mgate.def("H", &gate::H, pybind11::return_value_policy::take_ownership, "Create Hadamard gate", py::arg("index"));
    mgate.def("S", &gate::S, pybind11::return_value_policy::take_ownership, "Create pi/4-phase gate", py::arg("index"));
    mgate.def("Sdag", &gate::Sdag, pybind11::return_value_policy::take_ownership, "Create adjoint of pi/4-phase gate", py::arg("index"));
    mgate.def("T", &gate::T, pybind11::return_value_policy::take_ownership, "Create pi/8-phase gate", py::arg("index"));
    mgate.def("Tdag", &gate::Tdag, pybind11::return_value_policy::take_ownership, "Create adjoint of pi/8-phase gate", py::arg("index"));
    mgate.def("sqrtX", &gate::sqrtX, pybind11::return_value_policy::take_ownership, "Create pi/4 Pauli-X rotation gate", py::arg("index"));
    mgate.def("sqrtXdag", &gate::sqrtXdag, pybind11::return_value_policy::take_ownership, "Create adjoint of pi/4 Pauli-X rotation gate", py::arg("index"));
    mgate.def("sqrtY", &gate::sqrtY, pybind11::return_value_policy::take_ownership, "Create pi/4 Pauli-Y rotation gate", py::arg("index"));
    mgate.def("sqrtYdag", &gate::sqrtYdag, pybind11::return_value_policy::take_ownership, "Create adjoint of pi/4 Pauli-Y rotation gate", py::arg("index"));
    mgate.def("P0", &gate::P0, pybind11::return_value_policy::take_ownership, "Create projection gate to |0> subspace", py::arg("index"));
    mgate.def("P1", &gate::P1, pybind11::return_value_policy::take_ownership, "Create projection gate to |1> subspace", py::arg("index"));

    mgate.def("U1", &gate::U1, pybind11::return_value_policy::take_ownership, "Create QASM U1 gate", py::arg("index"), py::arg("lambda"));
    mgate.def("U2", &gate::U2, pybind11::return_value_policy::take_ownership, "Create QASM U2 gate", py::arg("index"), py::arg("phi"), py::arg("lambda"));
    mgate.def("U3", &gate::U3, pybind11::return_value_policy::take_ownership, "Create QASM U3 gate", py::arg("index"), py::arg("theta"), py::arg("phi"), py::arg("lambda"));

    mgate.def("RX", &gate::RX, pybind11::return_value_policy::take_ownership, "Create Pauli-X rotation gate", py::arg("index"), py::arg("angle"));
    mgate.def("RY", &gate::RY, pybind11::return_value_policy::take_ownership, "Create Pauli-Y rotation gate", py::arg("index"), py::arg("angle"));
    mgate.def("RZ", &gate::RZ, pybind11::return_value_policy::take_ownership, "Create Pauli-Z rotation gate", py::arg("index"), py::arg("angle"));

	mgate.def("CNOT", [](UINT control_qubit_index, UINT target_qubit_index) {
		auto ptr = gate::CNOT(control_qubit_index, target_qubit_index);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to CNOT.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership, "Create CNOT gate", py::arg("control"), py::arg("target"));
    mgate.def("CZ", [](UINT control_qubit_index, UINT target_qubit_index) {
		auto ptr = gate::CZ(control_qubit_index, target_qubit_index);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to CZ.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership, "Create CZ gate", py::arg("control"), py::arg("target"));
	mgate.def("SWAP", [](UINT target_index1, UINT target_index2) {
		auto ptr = gate::SWAP(target_index1, target_index2);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to SWAP.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership, "Create SWAP gate", py::arg("target1"), py::arg("target2")); 

	mgate.def("TOFFOLI", [](UINT control_index1, UINT control_index2, UINT target_index) {
		auto ptr = gate::X(target_index);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to TOFFOLI.");
		auto toffoli = gate::to_matrix_gate(ptr);
		toffoli->add_control_qubit(control_index1, 1);
		toffoli->add_control_qubit(control_index2, 1);
		delete ptr;
		return toffoli;
	}, pybind11::return_value_policy::take_ownership, "Create TOFFOLI gate", py::arg("control1"), py::arg("control2"), py::arg("target"));
	mgate.def("FREDKIN", [](UINT control_index, UINT target_index1, UINT target_index2) {
		auto ptr = gate::SWAP(target_index1, target_index2);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to FREDKIN.");
		auto fredkin = gate::to_matrix_gate(ptr);
		fredkin->add_control_qubit(control_index, 1);
		delete ptr;
		return fredkin;
	}, pybind11::return_value_policy::take_ownership, "Create FREDKIN gate", py::arg("control"), py::arg("target1"), py::arg("target2"));

    mgate.def("Pauli", [](std::vector<unsigned int> target_qubit_index_list, std::vector<unsigned int> pauli_ids) {
		if (target_qubit_index_list.size() != pauli_ids.size()) throw std::invalid_argument("Size of qubit list and pauli list must be equal.");
		auto ptr = gate::Pauli(target_qubit_index_list, pauli_ids);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to Pauli.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership, "Create multi-qubit Pauli gate", py::arg("index_list"), py::arg("pauli_ids"));
    mgate.def("PauliRotation", [](std::vector<unsigned int> target_qubit_index_list, std::vector<unsigned int> pauli_ids, double angle) {
		if (target_qubit_index_list.size() != pauli_ids.size()) throw std::invalid_argument("Size of qubit list and pauli list must be equal.");
		auto ptr = gate::PauliRotation(target_qubit_index_list, pauli_ids, angle);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to PauliRotation.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership, "Create multi-qubit Pauli rotation", py::arg("index_list"), py::arg("pauli_ids"), py::arg("angle"));

    //QuantumGateMatrix*(*ptr1)(unsigned int, ComplexMatrix) = &gate::DenseMatrix;
    //QuantumGateMatrix*(*ptr2)(std::vector<unsigned int>, ComplexMatrix) = &gate::DenseMatrix;
    mgate.def("DenseMatrix", [](unsigned int target_qubit_index, ComplexMatrix matrix) {
		if (matrix.rows() != 2 || matrix.cols() != 2) throw std::invalid_argument("matrix dims is not 2x2.");
		auto ptr = gate::DenseMatrix(target_qubit_index, matrix);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to DenseMatrix.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership, "Create dense matrix gate", py::arg("index"), py::arg("matrix")); 
    mgate.def("DenseMatrix", [](std::vector<unsigned int> target_qubit_index_list, ComplexMatrix matrix) {
		const ITYPE dim = 1ULL << target_qubit_index_list.size();
		if (matrix.rows() != dim || matrix.cols() != dim) throw std::invalid_argument("matrix dims is not consistent.");
		auto ptr = gate::DenseMatrix(target_qubit_index_list, matrix);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to DenseMatrix.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership, "Create dense matrix gate", py::arg("index_list"), py::arg("matrix")); 
	mgate.def("SparseMatrix", [](std::vector<unsigned int> target_qubit_index_list, SparseComplexMatrix matrix) {
		const ITYPE dim = 1ULL << target_qubit_index_list.size();
		if (matrix.rows() != dim || matrix.cols() != dim) throw std::invalid_argument("matrix dims is not consistent.");
		auto ptr = gate::SparseMatrix(target_qubit_index_list, matrix);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to SparseMatrix.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership, "Create sparse matrix gate", py::arg("index_list"), py::arg("matrix")); 
	mgate.def("DiagonalMatrix", [](std::vector<unsigned int> target_qubit_index_list, ComplexVector diagonal_element) {
		const ITYPE dim = 1ULL << target_qubit_index_list.size();
		if (diagonal_element.size() != dim) throw std::invalid_argument("dim of diagonal elemet is not consistent.");
		auto ptr = gate::DiagonalMatrix(target_qubit_index_list, diagonal_element);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to SparseMatrix.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership, "Create diagonal matrix gate", py::arg("index_list"), py::arg("diagonal_element"));

    mgate.def("RandomUnitary", [](std::vector<unsigned int> target_qubit_index_list) {
		auto ptr = gate::RandomUnitary(target_qubit_index_list);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to RandomUnitary.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership, "Create random unitary gate", py::arg("index_list")); 
    mgate.def("ReversibleBoolean", [](std::vector<UINT> target_qubit_list, std::function<ITYPE(ITYPE,ITYPE)> function_py) {
		auto ptr = gate::ReversibleBoolean(target_qubit_list, function_py);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to ReversibleBoolean.");
		return ptr;
    }, pybind11::return_value_policy::take_ownership, "Create reversible boolean gate", py::arg("index_list"), py::arg("func"));
	mgate.def("StateReflection", [](const QuantumStateBase* reflection_state) {
		auto ptr = gate::StateReflection(reflection_state);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to StateReflection.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership, "Create state reflection gate", py::arg("state"));

    mgate.def("BitFlipNoise", &gate::BitFlipNoise, pybind11::return_value_policy::take_ownership, "Create bit-flip noise", py::arg("index"), py::arg("prob"));
    mgate.def("DephasingNoise", &gate::DephasingNoise, pybind11::return_value_policy::take_ownership, "Create dephasing noise", py::arg("index"), py::arg("prob"));
    mgate.def("IndependentXZNoise", &gate::IndependentXZNoise, pybind11::return_value_policy::take_ownership, "Create independent XZ noise", py::arg("index"), py::arg("prob"));
    mgate.def("DepolarizingNoise", &gate::DepolarizingNoise, pybind11::return_value_policy::take_ownership, "Create depolarizing noise", py::arg("index"),py::arg("prob"));
	mgate.def("TwoQubitDepolarizingNoise", [](UINT target_index1, UINT target_index2, double probability) {
		auto ptr = gate::TwoQubitDepolarizingNoise(target_index1, target_index2, probability);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to TwoQubitDepolarizingNoise.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership, "Create two-qubit depolarizing noise", py::arg("index1"), py::arg("index2"), py::arg("prob"));
	mgate.def("AmplitudeDampingNoise", &gate::AmplitudeDampingNoise, pybind11::return_value_policy::take_ownership, "Create amplitude damping noise", py::arg("index"), py::arg("prob"));
    mgate.def("Measurement", &gate::Measurement, pybind11::return_value_policy::take_ownership, "Create measurement gate", py::arg("index"), py::arg("register"));

    QuantumGateMatrix*(*ptr3)(const QuantumGateBase*, const QuantumGateBase*) = &gate::merge;
    mgate.def("merge", ptr3, pybind11::return_value_policy::take_ownership, "Merge two quantum gate", py::arg("gate1"), py::arg("gate2"));

    QuantumGateMatrix*(*ptr4)(std::vector<const QuantumGateBase*>) = &gate::merge;
    mgate.def("merge", ptr4, pybind11::return_value_policy::take_ownership, "Merge quantum gate list", py::arg("gate_list"));

    QuantumGateMatrix*(*ptr5)(const QuantumGateBase*, const QuantumGateBase*) = &gate::add;
    mgate.def("add", ptr5, pybind11::return_value_policy::take_ownership, "Add quantum gate matrices", py::arg("gate1"), py::arg("gate2"));

    QuantumGateMatrix*(*ptr6)(std::vector<const QuantumGateBase*>) = &gate::add;
    mgate.def("add", ptr6, pybind11::return_value_policy::take_ownership, "Add quantum gate matrices", py::arg("gate_list"));

    mgate.def("to_matrix_gate", &gate::to_matrix_gate, pybind11::return_value_policy::take_ownership, "Convert named gate to matrix gate", py::arg("gate"));
    mgate.def("Probabilistic", &gate::Probabilistic, pybind11::return_value_policy::take_ownership, "Create probabilistic gate", py::arg("prob_list"), py::arg("gate_list"));
	mgate.def("ProbabilisticInstrument", &gate::ProbabilisticInstrument, pybind11::return_value_policy::take_ownership, "Create probabilistic instrument gate", py::arg("prob_list"), py::arg("gate_list"), py::arg("register"));
	mgate.def("CPTP", &gate::CPTP, pybind11::return_value_policy::take_ownership, "Create completely-positive trace preserving map", py::arg("kraus_list"));
	mgate.def("CP", &gate::CP, pybind11::return_value_policy::take_ownership, "Create completely-positive map", py::arg("kraus_list"), py::arg("state_normalize"), py::arg("probability_normalize"), py::arg("assign_zero_if_not_matched"));
	mgate.def("Instrument", &gate::Instrument, pybind11::return_value_policy::take_ownership, "Create instruments", py::arg("kraus_list"), py::arg("register"));
    mgate.def("Adaptive", &gate::Adaptive, pybind11::return_value_policy::take_ownership, "Create adaptive gate", py::arg("gate"), py::arg("condition"));

	py::class_<QuantumGate_SingleParameter, QuantumGateBase>(m, "QuantumGate_SingleParameter")
		.def("get_parameter_value", &QuantumGate_SingleParameter::get_parameter_value, "Get parameter value")
		.def("set_parameter_value", &QuantumGate_SingleParameter::set_parameter_value, "Set parameter value", py::arg("value"))
		.def("copy", &QuantumGate_SingleParameter::copy, pybind11::return_value_policy::take_ownership, "Create copied instance")
		;
	mgate.def("ParametricRX", &gate::ParametricRX, pybind11::return_value_policy::take_ownership, "Create parametric Pauli-X rotation gate", py::arg("index"), py::arg("angle"));
    mgate.def("ParametricRY", &gate::ParametricRY, pybind11::return_value_policy::take_ownership, "Create parametric Pauli-Y rotation gate", py::arg("index"), py::arg("angle"));
    mgate.def("ParametricRZ", &gate::ParametricRZ, pybind11::return_value_policy::take_ownership, "Create parametric Pauli-Z rotation gate", py::arg("index"), py::arg("angle"));
    mgate.def("ParametricPauliRotation", [](std::vector<unsigned int> target_qubit_index_list, std::vector<unsigned int> pauli_ids, double angle) {
		auto ptr = gate::ParametricPauliRotation(target_qubit_index_list, pauli_ids, angle);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to ParametricPauliRotation.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership, "Create parametric multi-qubit Pauli rotation gate", py::arg("index_list"), py::arg("pauli_ids"), py::arg("angle"));

    py::class_<QuantumCircuit>(m, "QuantumCircuit")
        .def(py::init<unsigned int>(), "Constructor", py::arg("qubit_count"))
        .def("copy", &QuantumCircuit::copy, pybind11::return_value_policy::take_ownership, "Create copied instance")
        // In order to avoid double release, we force using add_gate_copy in python
        //.def("add_gate_consume", (void (QuantumCircuit::*)(QuantumGateBase*))&QuantumCircuit::add_gate, "Add gate and take ownership", py::arg("gate"))
        //.def("add_gate_consume", (void (QuantumCircuit::*)(QuantumGateBase*, unsigned int))&QuantumCircuit::add_gate, "Add gate and take ownership", py::arg("gate"), py::arg("position"))
        .def("add_gate", (void (QuantumCircuit::*)(const QuantumGateBase*))&QuantumCircuit::add_gate_copy, "Add gate with copy", py::arg("gate"))
        .def("add_gate", (void (QuantumCircuit::*)(const QuantumGateBase*, unsigned int))&QuantumCircuit::add_gate_copy, "Add gate with copy", py::arg("gate"), py::arg("position"))
        .def("remove_gate", &QuantumCircuit::remove_gate, "Remove gate", py::arg("position"))

        .def("get_gate", [](const QuantumCircuit& circuit, unsigned int index) -> QuantumGateBase* { 
            if (index >= circuit.gate_list.size()) {
                std::cerr << "Error: QuantumCircuit::get_gate(const QuantumCircuit&, unsigned int): gate index is out of range" << std::endl;
                return NULL;
            }
            return circuit.gate_list[index]->copy(); 
        }, pybind11::return_value_policy::take_ownership, "Get gate instance", py::arg("position"))
        .def("get_gate_count", [](const QuantumCircuit& circuit) -> unsigned int {return (unsigned int)circuit.gate_list.size(); }, "Get gate count")
		.def("get_qubit_count", [](const QuantumCircuit& circuit) -> unsigned int {return circuit.qubit_count;}, "Get qubit count")

        .def("update_quantum_state", (void (QuantumCircuit::*)(QuantumStateBase*))&QuantumCircuit::update_quantum_state, "Update quantum state", py::arg("state"))
        .def("update_quantum_state", (void (QuantumCircuit::*)(QuantumStateBase*, unsigned int, unsigned int))&QuantumCircuit::update_quantum_state, "Update quantum state", py::arg("state"), py::arg("start"), py::arg("end"))
        .def("calculate_depth", &QuantumCircuit::calculate_depth, "Calculate depth of circuit")
        .def("to_string", &QuantumCircuit::to_string, "Get string representation")

        .def("add_X_gate", &QuantumCircuit::add_X_gate, "Add Pauli-X gate", py::arg("index"))
        .def("add_Y_gate", &QuantumCircuit::add_Y_gate, "Add Pauli-Y gate", py::arg("index"))
        .def("add_Z_gate", &QuantumCircuit::add_Z_gate, "Add Pauli-Z gate", py::arg("index"))
        .def("add_H_gate", &QuantumCircuit::add_H_gate, "Add Hadamard gate", py::arg("index"))
        .def("add_S_gate", &QuantumCircuit::add_S_gate, "Add pi/4 phase gate", py::arg("index"))
        .def("add_Sdag_gate", &QuantumCircuit::add_Sdag_gate, "Add adjoint of pi/4 phsae gate", py::arg("index"))
        .def("add_T_gate", &QuantumCircuit::add_T_gate, "Add pi/8 phase gate", py::arg("index"))
        .def("add_Tdag_gate", &QuantumCircuit::add_Tdag_gate, "Add adjoint of pi/8 phase gate", py::arg("index"))
        .def("add_sqrtX_gate", &QuantumCircuit::add_sqrtX_gate, "Add pi/4 Pauli-X rotation gate", py::arg("index"))
        .def("add_sqrtXdag_gate", &QuantumCircuit::add_sqrtXdag_gate, "Add adjoint of pi/4 Pauli-X rotation gate", py::arg("index"))
        .def("add_sqrtY_gate", &QuantumCircuit::add_sqrtY_gate, "Add pi/4 Pauli-Y rotation gate", py::arg("index"))
        .def("add_sqrtYdag_gate", &QuantumCircuit::add_sqrtYdag_gate, "Add adjoint of pi/4 Pauli-Y rotation gate", py::arg("index"))
        .def("add_P0_gate", &QuantumCircuit::add_P0_gate, "Add projection gate to |0> subspace", py::arg("index"))
        .def("add_P1_gate", &QuantumCircuit::add_P1_gate, "Add projection gate to |1> subspace", py::arg("index"))

        .def("add_CNOT_gate", &QuantumCircuit::add_CNOT_gate, "Add CNOT gate", py::arg("control"), py::arg("target"))
        .def("add_CZ_gate", &QuantumCircuit::add_CZ_gate, "Add CNOT gate", py::arg("control"), py::arg("target"))
        .def("add_SWAP_gate", &QuantumCircuit::add_SWAP_gate, "Add SWAP gate", py::arg("target1"), py::arg("target2"))

        .def("add_RX_gate", &QuantumCircuit::add_RX_gate, "Add Pauli-X rotation gate", py::arg("index"), py::arg("angle"))
        .def("add_RY_gate", &QuantumCircuit::add_RY_gate, "Add Pauli-Y rotation gate", py::arg("index"), py::arg("angle"))
        .def("add_RZ_gate", &QuantumCircuit::add_RZ_gate, "Add Pauli-Z rotation gate", py::arg("index"), py::arg("angle"))
        .def("add_U1_gate", &QuantumCircuit::add_U1_gate, "Add QASM U1 gate", py::arg("index"), py::arg("lambda"))
        .def("add_U2_gate", &QuantumCircuit::add_U2_gate, "Add QASM U2 gate", py::arg("index"), py::arg("phi"), py::arg("lambda"))
        .def("add_U3_gate", &QuantumCircuit::add_U3_gate, "Add QASM U3 gate", py::arg("index"), py::arg("theta"), py::arg("phi"), py::arg("lambda"))

        .def("add_multi_Pauli_gate", (void (QuantumCircuit::*)(std::vector<unsigned int>, std::vector<unsigned int>))&QuantumCircuit::add_multi_Pauli_gate, "Add multi-qubit Pauli gate", py::arg("index_list"), py::arg("pauli_ids"))
        .def("add_multi_Pauli_gate", (void (QuantumCircuit::*)(const PauliOperator&)) &QuantumCircuit::add_multi_Pauli_gate, "Add multi-qubit Pauli gate", py::arg("pauli"))
        .def("add_multi_Pauli_rotation_gate", (void (QuantumCircuit::*)(std::vector<unsigned int>, std::vector<unsigned int>, double))&QuantumCircuit::add_multi_Pauli_rotation_gate, "Add multi-qubit Pauli rotation gate", py::arg("index_list"), py::arg("pauli_ids"), py::arg("angle"))
        .def("add_multi_Pauli_rotation_gate", (void (QuantumCircuit::*)(const PauliOperator&))&QuantumCircuit::add_multi_Pauli_rotation_gate, "Add multi-qubit Pauli gate", py::arg("pauli"))
        .def("add_dense_matrix_gate", (void (QuantumCircuit::*)(unsigned int, const ComplexMatrix&))&QuantumCircuit::add_dense_matrix_gate, "Add dense matrix gate", py::arg("index"), py::arg("matrix"))
        .def("add_dense_matrix_gate", (void (QuantumCircuit::*)(std::vector<unsigned int>, const ComplexMatrix&))&QuantumCircuit::add_dense_matrix_gate, "Add dense matrix gate", py::arg("index_list"), py::arg("matrix"))
        .def("add_random_unitary_gate", &QuantumCircuit::add_random_unitary_gate, "Add random unitary gate", py::arg("index_list"))
        .def("add_diagonal_observable_rotation_gate", &QuantumCircuit::add_diagonal_observable_rotation_gate, "Add diagonal observable rotation gate", py::arg("observable"), py::arg("angle"))
        .def("add_observable_rotation_gate", &QuantumCircuit::add_observable_rotation_gate, "Add observable rotation gate", py::arg("observable"), py::arg("angle"), py::arg("repeat"))
			
		.def("__repr__", [](const QuantumCircuit &p) {return p.to_string(); });
    ;

    py::class_<ParametricQuantumCircuit, QuantumCircuit>(m, "ParametricQuantumCircuit")
        .def(py::init<unsigned int>(), "Constructor", py::arg("qubit_count"))
        .def("copy", &ParametricQuantumCircuit::copy, pybind11::return_value_policy::take_ownership, "Create copied instance")		
		.def("add_parametric_gate", (void (ParametricQuantumCircuit::*)(QuantumGate_SingleParameter* gate))  &ParametricQuantumCircuit::add_parametric_gate_copy, "Add parametric gate", py::arg("gate"))
        .def("add_parametric_gate", (void (ParametricQuantumCircuit::*)(QuantumGate_SingleParameter* gate, UINT))  &ParametricQuantumCircuit::add_parametric_gate_copy, "Add parametric gate", py::arg("gate"), py::arg("position"))
        .def("add_gate", (void (ParametricQuantumCircuit::*)(const QuantumGateBase* gate))  &ParametricQuantumCircuit::add_gate_copy, "Add gate", py::arg("gate"))
        .def("add_gate", (void (ParametricQuantumCircuit::*)(const QuantumGateBase* gate, unsigned int))  &ParametricQuantumCircuit::add_gate_copy, "Add gate", py::arg("gate"), py::arg("position"))
        .def("get_parameter_count", &ParametricQuantumCircuit::get_parameter_count, "Get parameter count")
        .def("get_parameter", &ParametricQuantumCircuit::get_parameter, "Get parameter", py::arg("index"))
        .def("set_parameter", &ParametricQuantumCircuit::set_parameter, "Set parameter", py::arg("index"), py::arg("parameter"))
        .def("get_parametric_gate_position", &ParametricQuantumCircuit::get_parametric_gate_position, "Get parametric gate position", py::arg("index"))
        .def("remove_gate", &ParametricQuantumCircuit::remove_gate, "Remove gate", py::arg("position"))

        .def("add_parametric_RX_gate", &ParametricQuantumCircuit::add_parametric_RX_gate, "Add parametric Pauli-X rotation gate", py::arg("index"), py::arg("angle"))
        .def("add_parametric_RY_gate", &ParametricQuantumCircuit::add_parametric_RY_gate, "Add parametric Pauli-Y rotation gate", py::arg("index"), py::arg("angle"))
        .def("add_parametric_RZ_gate", &ParametricQuantumCircuit::add_parametric_RZ_gate, "Add parametric Pauli-Z rotation gate", py::arg("index"), py::arg("angle"))
        .def("add_parametric_multi_Pauli_rotation_gate", &ParametricQuantumCircuit::add_parametric_multi_Pauli_rotation_gate, "Add parametric multi-qubit Pauli rotation gate", py::arg("index_list"), py::arg("pauli_ids"), py::arg("angle"))

        .def("__repr__", [](const ParametricQuantumCircuit &p) {return p.to_string(); });
    ;



    auto mcircuit = m.def_submodule("circuit");
    py::class_<QuantumCircuitOptimizer>(mcircuit, "QuantumCircuitOptimizer")
        .def(py::init<>(), "Constructor")
        .def("optimize", &QuantumCircuitOptimizer::optimize, "Optimize quantum circuit", py::arg("circuit"), py::arg("block_size"))
		.def("optimize_light", &QuantumCircuitOptimizer::optimize_light, "Optimize quantum circuit with light method", py::arg("circuit"))
		.def("merge_all", &QuantumCircuitOptimizer::merge_all, pybind11::return_value_policy::take_ownership, py::arg("circuit"))
        ;

    py::class_<QuantumCircuitSimulator>(m, "QuantumCircuitSimulator")
        .def(py::init<QuantumCircuit*, QuantumStateBase*>(), "Constructor", py::arg("circuit"), py::arg("state"))
        .def("initialize_state", &QuantumCircuitSimulator::initialize_state, "Initialize state")
        .def("initialize_random_state", &QuantumCircuitSimulator::initialize_random_state, "Initialize state with random pure state")
        .def("simulate", &QuantumCircuitSimulator::simulate, "Simulate circuit")
        .def("simulate_range", &QuantumCircuitSimulator::simulate_range, "Simulate circuit", py::arg("start"), py::arg("end"))
        .def("get_expectation_value", &QuantumCircuitSimulator::get_expectation_value, "Get expectation value", py::arg("observable"))
        .def("get_gate_count", &QuantumCircuitSimulator::get_gate_count, "Get gate count")
        .def("copy_state_to_buffer", &QuantumCircuitSimulator::copy_state_to_buffer, "Copy state to buffer")
        .def("copy_state_from_buffer", &QuantumCircuitSimulator::copy_state_from_buffer, "Copy buffer to state")
        .def("swap_state_and_buffer", &QuantumCircuitSimulator::swap_state_and_buffer, "Swap state and buffer")
        //.def("get_state_ptr", &QuantumCircuitSimulator::get_state_ptr, pybind11::return_value_policy::automatic_reference, "Get state ptr")
        ;
}



