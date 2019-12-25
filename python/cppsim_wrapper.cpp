
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
        .def(py::init<std::complex<double>>())
        .def(py::init<std::string, std::complex<double>>())
        .def(py::init<std::vector<unsigned int>&, std::string, std::complex<double>>())
        .def(py::init<std::vector<unsigned int>&, std::vector<unsigned int>&, std::complex<double>>())
        .def(py::init<std::vector<unsigned int>&, std::complex<double>>())
        .def("get_index_list", &PauliOperator::get_index_list)
        .def("get_pauli_id_list", &PauliOperator::get_pauli_id_list)
        .def("get_coef", &PauliOperator::get_coef)
        .def("add_single_Pauli", &PauliOperator::add_single_Pauli)
        .def("get_expectation_value", &PauliOperator::get_expectation_value)
        .def("get_transition_amplitude", &PauliOperator::get_transition_amplitude)
        .def("copy", &PauliOperator::copy, pybind11::return_value_policy::take_ownership)
        ;

    py::class_<GeneralQuantumOperator>(m, "GeneralQuantumOperator")
        .def(py::init<unsigned int>())
        // .def(py::init<std::string>())
        .def("add_operator", (void (GeneralQuantumOperator::*)(const PauliOperator*)) &GeneralQuantumOperator::add_operator)
        .def("add_operator", (void (GeneralQuantumOperator::*)(std::complex<double> coef, std::string))&GeneralQuantumOperator::add_operator)
        .def("is_hermitian", &GeneralQuantumOperator::is_hermitian)
        .def("get_qubit_count", &GeneralQuantumOperator::get_qubit_count)
        .def("get_state_dim", &GeneralQuantumOperator::get_state_dim)
        .def("get_term_count", &GeneralQuantumOperator::get_term_count)
        //.def("get_term", &GeneralQuantumOperator::get_term, pybind11::return_value_policy::take_ownership)
        .def("get_term",[](const GeneralQuantumOperator& quantum_operator, const unsigned int index) {
            return quantum_operator.get_term(index)->copy();
        }, pybind11::return_value_policy::take_ownership)
        .def("get_expectation_value", &GeneralQuantumOperator::get_expectation_value)
        .def("get_transition_amplitude", &GeneralQuantumOperator::get_transition_amplitude)
        //.def_static("get_split_GeneralQuantumOperator", &(GeneralQuantumOperator::get_split_observable));
        ;
    auto mquantum_operator = m.def_submodule("quantum_operator");
    mquantum_operator.def("create_quantum_operator_from_openfermion_file", &quantum_operator::create_general_quantum_operator_from_openfermion_file, pybind11::return_value_policy::take_ownership);
    mquantum_operator.def("create_quantum_operator_from_openfermion_text", &quantum_operator::create_general_quantum_operator_from_openfermion_text, pybind11::return_value_policy::take_ownership);
    mquantum_operator.def("create_split_quantum_operator", &quantum_operator::create_split_general_quantum_operator, pybind11::return_value_policy::take_ownership);

    py::class_<HermitianQuantumOperator, GeneralQuantumOperator>(m, "Observable")
        .def(py::init<unsigned int>())
        // .def(py::init<std::string>())
        .def("add_operator", (void (HermitianQuantumOperator::*)(const PauliOperator*)) &HermitianQuantumOperator::add_operator)
        .def("add_operator", (void (HermitianQuantumOperator::*)(std::complex<double> coef, std::string))&HermitianQuantumOperator::add_operator)
        .def("get_qubit_count", &HermitianQuantumOperator::get_qubit_count)
        .def("get_state_dim", &HermitianQuantumOperator::get_state_dim)
        .def("get_term_count", &HermitianQuantumOperator::get_term_count)
        //.def("get_term", &HermitianQuantumOperator::get_term, pybind11::return_value_policy::take_ownership)
        .def("get_term",[](const HermitianQuantumOperator& quantum_operator, const unsigned int index) {
            return quantum_operator.get_term(index)->copy();
        }, pybind11::return_value_policy::take_ownership)
        // .def("get_expectation_value", &HermitianQuantumOperator::get_expectation_value)
        .def("get_expectation_value", [](const HermitianQuantumOperator& observable, const QuantumStateBase* state) {
                                          double res = observable.get_expectation_value(state).real();
                                          return res;})
        .def("get_transition_amplitude", &HermitianQuantumOperator::get_transition_amplitude)
        //.def_static("get_split_Observable", &(HermitianQuantumOperator::get_split_observable));
        ;
    auto mobservable = m.def_submodule("observable");
    mobservable.def("create_observable_from_openfermion_file", &observable::create_observable_from_openfermion_file, pybind11::return_value_policy::take_ownership);
    mobservable.def("create_observable_from_openfermion_text", &observable::create_observable_from_openfermion_text, pybind11::return_value_policy::take_ownership);
    mobservable.def("create_split_observable", &observable::create_split_observable, pybind11::return_value_policy::take_ownership);


    py::class_<QuantumStateBase>(m, "QuantumStateBase");
    py::class_<QuantumState, QuantumStateBase>(m, "QuantumState")
        .def(py::init<unsigned int>())
        .def("set_zero_state", &QuantumState::set_zero_state)
        .def("set_computational_basis", &QuantumState::set_computational_basis)
        .def("set_Haar_random_state", (void (QuantumState::*)(void))&QuantumState::set_Haar_random_state)
        .def("set_Haar_random_state", (void (QuantumState::*)(UINT))&QuantumState::set_Haar_random_state)
        .def("get_zero_probability", &QuantumState::get_zero_probability)
        .def("get_marginal_probability", &QuantumState::get_marginal_probability)
        .def("get_entropy", &QuantumState::get_entropy)
        .def("get_squared_norm", &QuantumState::get_squared_norm)
        .def("normalize", &QuantumState::normalize)
        .def("allocate_buffer", &QuantumState::allocate_buffer, pybind11::return_value_policy::automatic_reference)
        //.def("copy", &QuantumState::copy, pybind11::return_value_policy::automatic_reference)
        .def("copy", &QuantumState::copy)
        .def("load", (void (QuantumState::*)(const QuantumStateBase*))&QuantumState::load)
        .def("load", (void (QuantumState::*)(const std::vector<CPPCTYPE>&))&QuantumState::load)
        .def("get_device_name", &QuantumState::get_device_name)
        .def("data_cpp", &QuantumState::data_cpp)
        .def("data_c", &QuantumState::data_c)
        .def("add_state", &QuantumState::add_state)
        .def("multiply_coef", &QuantumState::multiply_coef)
        .def("multiply_elementwise_function", &QuantumState::multiply_elementwise_function)
        .def("get_classical_value", &QuantumState::get_classical_value)
        .def("set_classical_value", &QuantumState::set_classical_value)
        .def("to_string",&QuantumState::to_string)
        .def("sampling",&QuantumState::sampling)

        .def("get_vector", [](const QuantumState& state) {
        Eigen::VectorXcd vec = Eigen::Map<Eigen::VectorXcd>(state.data_cpp(), state.dim);
        return vec;
        })
        .def("get_qubit_count", [](const QuantumState& state) -> unsigned int {return (unsigned int) state.qubit_count; })
        .def("__repr__", [](const QuantumState &p) {return p.to_string();});
        ;

	py::class_<DensityMatrix, QuantumStateBase>(m, "DensityMatrix")
		.def(py::init<unsigned int>())
		.def("set_zero_state", &DensityMatrix::set_zero_state)
		.def("set_computational_basis", &DensityMatrix::set_computational_basis)
		.def("set_Haar_random_state", (void (DensityMatrix::*)(void))&DensityMatrix::set_Haar_random_state)
		.def("set_Haar_random_state", (void (DensityMatrix::*)(UINT))&DensityMatrix::set_Haar_random_state)
		.def("get_zero_probability", &DensityMatrix::get_zero_probability)
		.def("get_marginal_probability", &DensityMatrix::get_marginal_probability)
		.def("get_entropy", &DensityMatrix::get_entropy)
		.def("get_squared_norm", &DensityMatrix::get_squared_norm)
		.def("normalize", &DensityMatrix::normalize)
		.def("allocate_buffer", &DensityMatrix::allocate_buffer, pybind11::return_value_policy::automatic_reference)
		.def("copy", &DensityMatrix::copy)
		.def("load", (void (DensityMatrix::*)(const QuantumStateBase*))&DensityMatrix::load)
		.def("load", (void (DensityMatrix::*)(const std::vector<CPPCTYPE>&))&DensityMatrix::load)
		.def("load", (void (DensityMatrix::*)(const ComplexMatrix&))&DensityMatrix::load)
		.def("get_device_name", &DensityMatrix::get_device_name)
		.def("data_cpp", &DensityMatrix::data_cpp)
		.def("data_c", &DensityMatrix::data_c)
		.def("add_state", &DensityMatrix::add_state)
		.def("multiply_coef", &DensityMatrix::multiply_coef)
		.def("get_classical_value", &DensityMatrix::get_classical_value)
		.def("set_classical_value", &DensityMatrix::set_classical_value)
		.def("to_string", &DensityMatrix::to_string)
		.def("sampling", &DensityMatrix::sampling)

		.def("get_matrix", [](const DensityMatrix& state) {
			Eigen::MatrixXcd mat(state.dim, state.dim);
			CTYPE* ptr = state.data_c();
			for (ITYPE y = 0; y < state.dim; ++y) {
				for (ITYPE x = 0; x < state.dim; ++x) {
					mat(y, x) = ptr[y*state.dim + x];
				}
			}
			return mat;
		})
		.def("__repr__", [](const DensityMatrix &p) {return p.to_string(); });
		;

#ifdef _USE_GPU
    py::class_<QuantumStateGpu, QuantumStateBase>(m, "QuantumStateGpu")
        .def(py::init<unsigned int>())
        .def(py::init<unsigned int, unsigned int>())
        .def("set_zero_state", &QuantumStateGpu::set_zero_state)
        .def("set_computational_basis", &QuantumStateGpu::set_computational_basis)
        .def("set_Haar_random_state", (void (QuantumStateGpu::*)(void))&QuantumStateGpu::set_Haar_random_state)
        .def("set_Haar_random_state", (void (QuantumStateGpu::*)(UINT))&QuantumStateGpu::set_Haar_random_state)
        .def("get_zero_probability", &QuantumStateGpu::get_zero_probability)
        .def("get_marginal_probability", &QuantumStateGpu::get_marginal_probability)
        .def("get_entropy", &QuantumStateGpu::get_entropy)
        .def("get_squared_norm", &QuantumStateGpu::get_squared_norm)
        .def("normalize", &QuantumStateGpu::normalize)
        .def("allocate_buffer", &QuantumStateGpu::allocate_buffer, pybind11::return_value_policy::automatic_reference)
        //.def("copy", &QuantumStateGpu::copy, pybind11::return_value_policy::automatic_reference)
        .def("copy", &QuantumStateGpu::copy)
        .def("load", (void (QuantumStateGpu::*)(const QuantumStateBase*))&QuantumStateGpu::load)
        .def("load", (void (QuantumStateGpu::*)(const std::vector<CPPCTYPE>&))&QuantumStateGpu::load)
        .def("get_device_name", &QuantumStateGpu::get_device_name)
        //.def("data_cpp", &QuantumStateGpu::data_cpp)
        //.def("data_c", &QuantumStateGpu::data_c)
        .def("add_state", &QuantumStateGpu::add_state)
        .def("multiply_coef", &QuantumStateGpu::multiply_coef)
        .def("multiply_elementwise_function", &QuantumStateGpu::multiply_elementwise_function)
        .def("get_classical_value", &QuantumStateGpu::get_classical_value)
        .def("set_classical_value", &QuantumStateGpu::set_classical_value)
        .def("to_string", &QuantumStateGpu::to_string)
        .def("sampling", &QuantumStateGpu::sampling)
        .def("get_vector", [](const QuantumStateGpu& state) {
            Eigen::VectorXcd vec = Eigen::Map<Eigen::VectorXcd>(state.duplicate_data_cpp(), state.dim);
            return vec;
        }, pybind11::return_value_policy::take_ownership)
        .def("get_qubit_count", [](const QuantumStateGpu& state) -> unsigned int {return (unsigned int) state.qubit_count; })
        .def("__repr__", [](const QuantumStateGpu &p) {return p.to_string(); });
        ;

#endif

    auto mstate = m.def_submodule("state");
    //using namespace state;
#ifdef _USE_GPU
    mstate.def("inner_product", py::overload_cast<const QuantumStateGpu*, const QuantumStateGpu*>(&state::inner_product));
#endif
    mstate.def("inner_product", py::overload_cast<const QuantumState*, const QuantumState*>(&state::inner_product));
    //mstate.def("inner_product", &state::inner_product);

    py::class_<QuantumGateBase>(m, "QuantumGateBase")
        .def("update_quantum_state", &QuantumGateBase::update_quantum_state)
        .def("copy",&QuantumGateBase::copy, pybind11::return_value_policy::take_ownership)
        .def("to_string", &QuantumGateBase::to_string)
        .def("get_matrix", [](const QuantumGateBase& gate) {
            ComplexMatrix mat;
            gate.set_matrix(mat);
            return mat;
        })
        .def("__repr__", [](const QuantumGateBase &p) {return p.to_string(); })
        .def("get_target_index_list", &QuantumGateBase::get_target_index_list)
        .def("get_control_index_list", &QuantumGateBase::get_control_index_list)
        .def("get_name", &QuantumGateBase::get_name)
        .def("is_commute", &QuantumGateBase::is_commute)
        .def("is_Pauli", &QuantumGateBase::is_Pauli)
        .def("is_Clifford", &QuantumGateBase::is_Clifford)
        .def("is_Gaussian", &QuantumGateBase::is_Gaussian)
        .def("is_parametric", &QuantumGateBase::is_parametric)
        .def("is_diagonal", &QuantumGateBase::is_diagonal)
        ;

    py::class_<QuantumGateMatrix,QuantumGateBase>(m, "QuantumGateMatrix")
        .def("update_quantum_state", &QuantumGateMatrix::update_quantum_state)
        .def("add_control_qubit", &QuantumGateMatrix::add_control_qubit)
        .def("multiply_scalar", &QuantumGateMatrix::multiply_scalar)
        .def("copy", &QuantumGateMatrix::copy, pybind11::return_value_policy::take_ownership)
        .def("to_string", &QuantumGateMatrix::to_string)

        .def("get_matrix", [](const QuantumGateMatrix& gate) {
        ComplexMatrix mat;
        gate.set_matrix(mat);
        return mat;
        })
        .def("__repr__", [](const QuantumGateMatrix &p) {return p.to_string(); });
        ;

    auto mgate = m.def_submodule("gate");
    mgate.def("Identity", &gate::Identity, pybind11::return_value_policy::take_ownership);
    mgate.def("X", &gate::X, pybind11::return_value_policy::take_ownership);
    mgate.def("Y", &gate::Y, pybind11::return_value_policy::take_ownership);
    mgate.def("Z", &gate::Z, pybind11::return_value_policy::take_ownership);
    mgate.def("H", &gate::H, pybind11::return_value_policy::take_ownership);
    mgate.def("S", &gate::S, pybind11::return_value_policy::take_ownership);
    mgate.def("Sdag", &gate::Sdag, pybind11::return_value_policy::take_ownership);
    mgate.def("T", &gate::T, pybind11::return_value_policy::take_ownership);
    mgate.def("Tdag", &gate::Tdag, pybind11::return_value_policy::take_ownership);
    mgate.def("sqrtX", &gate::sqrtX, pybind11::return_value_policy::take_ownership);
    mgate.def("sqrtXdag", &gate::sqrtXdag, pybind11::return_value_policy::take_ownership);
    mgate.def("sqrtY", &gate::sqrtY, pybind11::return_value_policy::take_ownership);
    mgate.def("sqrtYdag", &gate::sqrtYdag, pybind11::return_value_policy::take_ownership);
    mgate.def("P0", &gate::P0, pybind11::return_value_policy::take_ownership);
    mgate.def("P1", &gate::P1, pybind11::return_value_policy::take_ownership);

    mgate.def("U1", &gate::U1, pybind11::return_value_policy::take_ownership);
    mgate.def("U2", &gate::U2, pybind11::return_value_policy::take_ownership);
    mgate.def("U3", &gate::U3, pybind11::return_value_policy::take_ownership);

    mgate.def("RX", &gate::RX, pybind11::return_value_policy::take_ownership);
    mgate.def("RY", &gate::RY, pybind11::return_value_policy::take_ownership);
    mgate.def("RZ", &gate::RZ, pybind11::return_value_policy::take_ownership);

	mgate.def("CNOT", [](UINT control_qubit_index, UINT target_qubit_index) {
		auto ptr = gate::CNOT(control_qubit_index, target_qubit_index);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to CNOT.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership);
    mgate.def("CZ", [](UINT control_qubit_index, UINT target_qubit_index) {
		auto ptr = gate::CZ(control_qubit_index, target_qubit_index);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to CZ.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership);
    mgate.def("SWAP", [](UINT target_index1, UINT target_index2) {
		auto ptr = gate::SWAP(target_index1, target_index2);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to SWAP.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership); 

    mgate.def("Pauli", [](std::vector<unsigned int> target_qubit_index_list, std::vector<unsigned int> pauli_ids) {
		if (target_qubit_index_list.size() != pauli_ids.size()) throw std::invalid_argument("Size of qubit list and pauli list must be equal.");
		auto ptr = gate::Pauli(target_qubit_index_list, pauli_ids);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to Pauli.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership);
    mgate.def("PauliRotation", [](std::vector<unsigned int> target_qubit_index_list, std::vector<unsigned int> pauli_ids, double angle) {
		if (target_qubit_index_list.size() != pauli_ids.size()) throw std::invalid_argument("Size of qubit list and pauli list must be equal.");
		auto ptr = gate::PauliRotation(target_qubit_index_list, pauli_ids, angle);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to PauliRotation.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership);

    //QuantumGateMatrix*(*ptr1)(unsigned int, ComplexMatrix) = &gate::DenseMatrix;
    //QuantumGateMatrix*(*ptr2)(std::vector<unsigned int>, ComplexMatrix) = &gate::DenseMatrix;
    mgate.def("DenseMatrix", [](unsigned int target_qubit_index, ComplexMatrix matrix) {
		if (matrix.rows() != 2 || matrix.cols() != 2) throw std::invalid_argument("matrix dims is not 2x2.");
		auto ptr = gate::DenseMatrix(target_qubit_index, matrix);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to DenseMatrix.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership); 
    mgate.def("DenseMatrix", [](std::vector<unsigned int> target_qubit_index_list, ComplexMatrix matrix) {
		const ITYPE dim = 1ULL << target_qubit_index_list.size();
		if (matrix.rows() != dim || matrix.cols() != dim) throw std::invalid_argument("matrix dims is not consistent.");
		auto ptr = gate::DenseMatrix(target_qubit_index_list, matrix);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to DenseMatrix.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership); 
	mgate.def("SparseMatrix", [](std::vector<unsigned int> target_qubit_index_list, SparseComplexMatrix matrix) {
		const ITYPE dim = 1ULL << target_qubit_index_list.size();
		if (matrix.rows() != dim || matrix.cols() != dim) throw std::invalid_argument("matrix dims is not consistent.");
		auto ptr = gate::SparseMatrix(target_qubit_index_list, matrix);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to SparseMatrix.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership); 

    mgate.def("RandomUnitary", [](std::vector<unsigned int> target_qubit_index_list) {
		auto ptr = gate::RandomUnitary(target_qubit_index_list);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to RandomUnitary.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership); 
    mgate.def("ReversibleBoolean", [](std::vector<UINT> target_qubit_list, std::function<ITYPE(ITYPE,ITYPE)> function_py) {
		auto ptr = gate::ReversibleBoolean(target_qubit_list, function_py);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to ReversibleBoolean.");
		return ptr;
    }, pybind11::return_value_policy::take_ownership);
	mgate.def("StateReflection", [](const QuantumStateBase* reflection_state) {
		auto ptr = gate::StateReflection(reflection_state);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to StateReflection.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership);

    mgate.def("BitFlipNoise", &gate::BitFlipNoise);
    mgate.def("DephasingNoise", &gate::DephasingNoise);
    mgate.def("IndependentXZNoise", &gate::IndependentXZNoise);
    mgate.def("DepolarizingNoise", &gate::DepolarizingNoise);
	mgate.def("TwoQubitDepolarizingNoise", [](UINT target_index1, UINT target_index2, double probability) {
		auto ptr = gate::TwoQubitDepolarizingNoise(target_index1, target_index2, probability);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to TwoQubitDepolarizingNoise.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership);
	mgate.def("AmplitudeDampingNoise", &gate::AmplitudeDampingNoise);
    mgate.def("Measurement", &gate::Measurement);

    QuantumGateMatrix*(*ptr3)(const QuantumGateBase*, const QuantumGateBase*) = &gate::merge;
    mgate.def("merge", ptr3, pybind11::return_value_policy::take_ownership);

    QuantumGateMatrix*(*ptr4)(std::vector<const QuantumGateBase*>) = &gate::merge;
    mgate.def("merge", ptr4, pybind11::return_value_policy::take_ownership);

    QuantumGateMatrix*(*ptr5)(const QuantumGateBase*, const QuantumGateBase*) = &gate::add;
    mgate.def("add", ptr5, pybind11::return_value_policy::take_ownership);

    QuantumGateMatrix*(*ptr6)(std::vector<const QuantumGateBase*>) = &gate::add;
    mgate.def("add", ptr6, pybind11::return_value_policy::take_ownership);

    mgate.def("to_matrix_gate", &gate::to_matrix_gate, pybind11::return_value_policy::take_ownership);
    mgate.def("Probabilistic", &gate::Probabilistic, pybind11::return_value_policy::take_ownership);
    mgate.def("CPTP", &gate::CPTP, pybind11::return_value_policy::take_ownership);
    mgate.def("Instrument", &gate::Instrument, pybind11::return_value_policy::take_ownership);
    mgate.def("Adaptive", &gate::Adaptive, pybind11::return_value_policy::take_ownership);

	py::class_<QuantumGate_SingleParameter, QuantumGateBase>(m, "QuantumGate_SingleParameter")
		.def("get_parameter_value", &QuantumGate_SingleParameter::get_parameter_value)
		.def("set_parameter_value", &QuantumGate_SingleParameter::set_parameter_value)
		;
	mgate.def("ParametricRX", &gate::ParametricRX);
    mgate.def("ParametricRY", &gate::ParametricRY);
    mgate.def("ParametricRZ", &gate::ParametricRZ);
    mgate.def("ParametricPauliRotation", [](std::vector<unsigned int> target_qubit_index_list, std::vector<unsigned int> pauli_ids, double angle) {
		auto ptr = gate::ParametricPauliRotation(target_qubit_index_list, pauli_ids, angle);
		if (ptr == NULL) throw std::invalid_argument("Invalid argument passed to ParametricPauliRotation.");
		return ptr;
	}, pybind11::return_value_policy::take_ownership);


    py::class_<QuantumCircuit>(m, "QuantumCircuit")
        .def(py::init<unsigned int>())
        .def("copy", &QuantumCircuit::copy, pybind11::return_value_policy::take_ownership)
        // In order to avoid double release, we force using add_gate_copy in python
        .def("add_gate_consume", (void (QuantumCircuit::*)(QuantumGateBase*))&QuantumCircuit::add_gate)
        .def("add_gate_consume", (void (QuantumCircuit::*)(QuantumGateBase*, unsigned int))&QuantumCircuit::add_gate)
        .def("add_gate", (void (QuantumCircuit::*)(const QuantumGateBase*))&QuantumCircuit::add_gate_copy)
        .def("add_gate", (void (QuantumCircuit::*)(const QuantumGateBase*, unsigned int))&QuantumCircuit::add_gate_copy)
        .def("remove_gate", &QuantumCircuit::remove_gate)

        .def("get_gate", [](const QuantumCircuit& circuit, unsigned int index) -> QuantumGateBase* { 
            if (index >= circuit.gate_list.size()) {
                std::cerr << "Error: QuantumCircuit::get_gate(const QuantumCircuit&, unsigned int): gate index is out of range" << std::endl;
                return NULL;
            }
            return circuit.gate_list[index]->copy(); 
        }, pybind11::return_value_policy::take_ownership)
        .def("get_gate_count", [](const QuantumCircuit& circuit) -> unsigned int {return (unsigned int)circuit.gate_list.size(); })
		.def("get_qubit_count", [](const QuantumCircuit& circuit) -> unsigned int {return circuit.qubit_count;})

        .def("update_quantum_state", (void (QuantumCircuit::*)(QuantumStateBase*))&QuantumCircuit::update_quantum_state)
        .def("update_quantum_state", (void (QuantumCircuit::*)(QuantumStateBase*, unsigned int, unsigned int))&QuantumCircuit::update_quantum_state)
        .def("calculate_depth", &QuantumCircuit::calculate_depth)
        .def("to_string", &QuantumCircuit::to_string)

        .def("add_X_gate", &QuantumCircuit::add_X_gate)
        .def("add_Y_gate", &QuantumCircuit::add_Y_gate)
        .def("add_Z_gate", &QuantumCircuit::add_Z_gate)
        .def("add_H_gate", &QuantumCircuit::add_H_gate)
        .def("add_S_gate", &QuantumCircuit::add_S_gate)
        .def("add_Sdag_gate", &QuantumCircuit::add_Sdag_gate)
        .def("add_T_gate", &QuantumCircuit::add_T_gate)
        .def("add_Tdag_gate", &QuantumCircuit::add_Tdag_gate)
        .def("add_sqrtX_gate", &QuantumCircuit::add_sqrtX_gate)
        .def("add_sqrtXdag_gate", &QuantumCircuit::add_sqrtXdag_gate)
        .def("add_sqrtY_gate", &QuantumCircuit::add_sqrtY_gate)
        .def("add_sqrtYdag_gate", &QuantumCircuit::add_sqrtYdag_gate)
        .def("add_P0_gate", &QuantumCircuit::add_P0_gate)
        .def("add_P1_gate", &QuantumCircuit::add_P1_gate)

        .def("add_CNOT_gate", &QuantumCircuit::add_CNOT_gate)
        .def("add_CZ_gate", &QuantumCircuit::add_CZ_gate)
        .def("add_SWAP_gate", &QuantumCircuit::add_SWAP_gate)

        .def("add_RX_gate", &QuantumCircuit::add_RX_gate)
        .def("add_RY_gate", &QuantumCircuit::add_RY_gate)
        .def("add_RZ_gate", &QuantumCircuit::add_RZ_gate)
        .def("add_U1_gate", &QuantumCircuit::add_U1_gate)
        .def("add_U2_gate", &QuantumCircuit::add_U2_gate)
        .def("add_U3_gate", &QuantumCircuit::add_U3_gate)

        .def("add_multi_Pauli_gate", (void (QuantumCircuit::*)(std::vector<unsigned int>, std::vector<unsigned int>))&QuantumCircuit::add_multi_Pauli_gate)
        .def("add_multi_Pauli_gate", (void (QuantumCircuit::*)(const PauliOperator&)) &QuantumCircuit::add_multi_Pauli_gate)
        .def("add_multi_Pauli_rotation_gate", (void (QuantumCircuit::*)(std::vector<unsigned int>, std::vector<unsigned int>, double))&QuantumCircuit::add_multi_Pauli_rotation_gate)
        .def("add_multi_Pauli_rotation_gate", (void (QuantumCircuit::*)(const PauliOperator&))&QuantumCircuit::add_multi_Pauli_rotation_gate)
        .def("add_dense_matrix_gate", (void (QuantumCircuit::*)(unsigned int, const ComplexMatrix&))&QuantumCircuit::add_dense_matrix_gate)
        .def("add_dense_matrix_gate", (void (QuantumCircuit::*)(std::vector<unsigned int>, const ComplexMatrix&))&QuantumCircuit::add_dense_matrix_gate)
        .def("add_random_unitary_gate", &QuantumCircuit::add_random_unitary_gate)
        .def("add_diagonal_observable_rotation_gate", &QuantumCircuit::add_diagonal_observable_rotation_gate)
        .def("add_observable_rotation_gate", &QuantumCircuit::add_observable_rotation_gate)
        .def("__repr__", [](const QuantumCircuit &p) {return p.to_string(); });
    ;

    py::class_<ParametricQuantumCircuit, QuantumCircuit>(m, "ParametricQuantumCircuit")
        .def(py::init<unsigned int>())
        .def("copy", &ParametricQuantumCircuit::copy, pybind11::return_value_policy::take_ownership)
        .def("add_parametric_gate", (void (ParametricQuantumCircuit::*)(QuantumGate_SingleParameter* gate))  &ParametricQuantumCircuit::add_parametric_gate)
        .def("add_parametric_gate", (void (ParametricQuantumCircuit::*)(QuantumGate_SingleParameter* gate, UINT))  &ParametricQuantumCircuit::add_parametric_gate)
        .def("add_gate", (void (ParametricQuantumCircuit::*)(const QuantumGateBase* gate))  &ParametricQuantumCircuit::add_gate_copy )
        .def("add_gate", (void (ParametricQuantumCircuit::*)(const QuantumGateBase* gate, unsigned int))  &ParametricQuantumCircuit::add_gate_copy)
        .def("get_parameter_count", &ParametricQuantumCircuit::get_parameter_count)
        .def("get_parameter", &ParametricQuantumCircuit::get_parameter)
        .def("set_parameter", &ParametricQuantumCircuit::set_parameter)
        .def("get_parametric_gate_position", &ParametricQuantumCircuit::get_parametric_gate_position)
        .def("remove_gate", &ParametricQuantumCircuit::remove_gate)

        .def("add_parametric_RX_gate", &ParametricQuantumCircuit::add_parametric_RX_gate)
        .def("add_parametric_RY_gate", &ParametricQuantumCircuit::add_parametric_RY_gate)
        .def("add_parametric_RZ_gate", &ParametricQuantumCircuit::add_parametric_RZ_gate)
        .def("add_parametric_multi_Pauli_rotation_gate", &ParametricQuantumCircuit::add_parametric_multi_Pauli_rotation_gate)

        .def("__repr__", [](const ParametricQuantumCircuit &p) {return p.to_string(); });
    ;



    auto mcircuit = m.def_submodule("circuit");
    py::class_<QuantumCircuitOptimizer>(mcircuit, "QuantumCircuitOptimizer")
        .def(py::init<>())
        .def("optimize", &QuantumCircuitOptimizer::optimize)
        .def("merge_all", &QuantumCircuitOptimizer::merge_all, pybind11::return_value_policy::take_ownership)
        ;

    py::class_<QuantumCircuitSimulator>(m, "QuantumCircuitSimulator")
        .def(py::init<QuantumCircuit*, QuantumStateBase*>())
        .def("initialize_state", &QuantumCircuitSimulator::initialize_state)
        .def("initialize_random_state", &QuantumCircuitSimulator::initialize_random_state)
        .def("simulate", &QuantumCircuitSimulator::simulate)
        .def("simulate_range", &QuantumCircuitSimulator::simulate_range)
        .def("get_expectation_value", &QuantumCircuitSimulator::get_expectation_value)
        .def("get_gate_count", &QuantumCircuitSimulator::get_gate_count)
        .def("copy_state_to_buffer", &QuantumCircuitSimulator::copy_state_to_buffer)
        .def("copy_state_from_buffer", &QuantumCircuitSimulator::copy_state_from_buffer)
        .def("swap_state_and_buffer", &QuantumCircuitSimulator::swap_state_and_buffer)
        .def("get_state_ptr", &QuantumCircuitSimulator::get_state_ptr, pybind11::return_value_policy::automatic_reference)
        ;
}



