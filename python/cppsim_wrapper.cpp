#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cppsim/circuit.hpp>
#include <cppsim/circuit_optimizer.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_matrix_diagonal.hpp>
#include <cppsim/gate_matrix_sparse.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_to_gqo.hpp>
#include <cppsim/general_quantum_operator.hpp>
#include <cppsim/noisesimulator.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/simulator.hpp>
#include <cppsim/state.hpp>
#include <cppsim/state_dm.hpp>
#include <cppsim/utility.hpp>
#include <csim/memory_ops.hpp>
#include <csim/stat_ops.hpp>
#include <csim/update_ops.hpp>
#include <vqcsim/causalcone_simulator.hpp>

#ifdef _USE_GPU
#include <cppsim/state_gpu.hpp>
#endif

#include <vqcsim/GradCalculator.hpp>
#include <vqcsim/parametric_circuit.hpp>
#include <vqcsim/parametric_gate.hpp>
#include <vqcsim/parametric_gate_factory.hpp>

namespace py = pybind11;
PYBIND11_MODULE(qulacs_core, m) {
    m.doc() = "cppsim python interface";

    py::class_<PauliOperator>(m, "PauliOperator")
        .def(py::init<std::complex<double>>(), "Constructor",
            py::arg("coef") = 1. + 0.i)
        .def(py::init<std::string, std::complex<double>>(), "Constructor",
            py::arg("pauli_string"),
            py::arg("coef") = std::complex<double>(1., 0.))
        .def(py::init<std::vector<UINT>, std::string, std::complex<double>>(),
            "Constructor", py::arg("target_qubit_index_list"),
            py::arg("pauli_operator_type_list"), py::arg("coef") = 1. + 0.i)
        .def("get_index_list", &PauliOperator::get_index_list,
            "Get list of target qubit indices")
        .def("get_pauli_id_list", &PauliOperator::get_pauli_id_list,
            "Get list of Pauli IDs (I,X,Y,Z) = (0,1,2,3)")
        .def("get_coef", &PauliOperator::get_coef,
            "Get coefficient of Pauli term")
        .def("add_single_Pauli", &PauliOperator::add_single_Pauli,
            "Add Pauli operator to this term", py::arg("index"),
            py::arg("pauli_type"))
        .def("get_expectation_value", &PauliOperator::get_expectation_value,
            "Get expectation value", py::arg("state"))
        .def("get_expectation_value_single_thread",
            &PauliOperator::get_expectation_value_single_thread,
            "Get expectation value", py::arg("state"))
        .def("get_transition_amplitude",
            &PauliOperator::get_transition_amplitude,
            "Get transition amplitude", py::arg("state_bra"),
            py::arg("state_ket"))
        .def("copy", &PauliOperator::copy,
            py::return_value_policy::take_ownership,
            "Create copied instance of Pauli operator class")
        .def("get_pauli_string", &PauliOperator::get_pauli_string,
            "Get pauli string")
        .def("change_coef", &PauliOperator::change_coef, "Change coefficient",
            py::arg("new_coef"))
        .def(py::self * py::self)
        .def(
            "__mul__",
            [](const PauliOperator& a, std::complex<double>& b) {
                return a * b;
            },
            py::is_operator())
        .def(py::self *= py::self)
        .def(
            "__IMUL__",
            [](PauliOperator& a, std::complex<double>& b) { return a *= b; },
            py::is_operator());

    py::class_<GeneralQuantumOperator>(m, "GeneralQuantumOperator")
        .def(py::init<UINT>(), "Constructor", py::arg("qubit_count"))
        .def("add_operator",
            py::overload_cast<const PauliOperator*>(
                &GeneralQuantumOperator::add_operator),
            "Add Pauli operator", py::arg("pauli_operator"))
        .def("add_operator",
            py::overload_cast<const CPPCTYPE, std::string>(
                &GeneralQuantumOperator::add_operator),
            "Add Pauli operator", py::arg("coef"), py::arg("pauli_string"))
        .def("add_operator_move", &GeneralQuantumOperator::add_operator_move,
            "Add Pauli operator", py::arg("pauli_operator"))
        .def("add_operator_copy", &GeneralQuantumOperator::add_operator_copy,
            "Add Pauli operator", py::arg("pauli_operator"))
        .def("is_hermitian", &GeneralQuantumOperator::is_hermitian,
            "Get is Herimitian")
        .def("get_qubit_count", &GeneralQuantumOperator::get_qubit_count,
            "Get qubit count")
        .def("get_state_dim", &GeneralQuantumOperator::get_state_dim,
            "Get state dimension")
        .def("get_term_count", &GeneralQuantumOperator::get_term_count,
            "Get count of Pauli terms")
        .def("get_matrix", &GeneralQuantumOperator::get_matrix,
            "Get the Hermitian matrix representation of the observable")
        .def("apply_to_state",
            py::overload_cast<QuantumStateBase*, const QuantumStateBase&,
                QuantumStateBase*>(
                &GeneralQuantumOperator::apply_to_state, py::const_),
            "Apply observable to `state_to_be_multiplied`. The result is "
            "stored into `dst_state`.",
            py::arg("work_state"), py::arg("state_to_be_multiplied"),
            py::arg("dst_state"))
        .def(
            "apply_to_state",
            [](const GeneralQuantumOperator& self,
                const QuantumStateBase& state, QuantumStateBase* dst_state) {
                QuantumStateBase* work_state;
                if (state.is_state_vector())
                    work_state = new QuantumState(state.qubit_count);
                else
                    work_state = new DensityMatrix(state.qubit_count);
                self.apply_to_state(work_state, state, dst_state);
                delete work_state;
            },
            "Apply observable to `state_to_be_multiplied`. The result is "
            "stored into `dst_state`.",
            py::arg("state_to_be_multiplied"), py::arg("dst_state"))
        .def(
            "get_term",
            [](const GeneralQuantumOperator& quantum_operator,
                const UINT index) {
                return quantum_operator.get_term(index)->copy();
            },
            py::return_value_policy::take_ownership, "Get Pauli term",
            py::arg("index"))
        .def("get_expectation_value",
            &GeneralQuantumOperator::get_expectation_value,
            "Get expectation value", py::arg("state"))
        .def("get_expectation_value_single_thread",
            &GeneralQuantumOperator::get_expectation_value_single_thread,
            "Get expectation value", py::arg("state"))
        .def("get_transition_amplitude",
            &GeneralQuantumOperator::get_transition_amplitude,
            "Get transition amplitude", py::arg("state_bra"),
            py::arg("state_ket"))
        .def("__str__", &GeneralQuantumOperator::to_string, "to string")
        .def("copy", &GeneralQuantumOperator::copy,
            py::return_value_policy::take_ownership,
            "Create copied instance of General Quantum operator class")
        .def(
            "to_json",
            [](const GeneralQuantumOperator& gqo) -> std::string {
                return ptree::to_json(gqo.to_ptree());
            },
            "to json string")
        .def(py::self + py::self)
        .def(
            "__add__",
            [](const GeneralQuantumOperator& a, const PauliOperator& b) {
                return a + b;
            },
            py::is_operator())
        .def(py::self += py::self)
        .def(
            "__IADD__",
            [](GeneralQuantumOperator& a, const PauliOperator& b) {
                return a += b;
            },
            py::is_operator())
        .def(py::self - py::self)
        .def(
            "__sub__",
            [](const GeneralQuantumOperator& a, const PauliOperator& b) {
                return a - b;
            },
            py::is_operator())
        .def(py::self -= py::self)
        .def(
            "__ISUB__",
            [](GeneralQuantumOperator& a, const PauliOperator& b) {
                return a -= b;
            },
            py::is_operator())
        .def(py::self * py::self)
        .def(
            "__mul__",
            [](const GeneralQuantumOperator& a, const PauliOperator& b) {
                return a * b;
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const GeneralQuantumOperator& a, std::complex<double>& b) {
                return a * b;
            },
            py::is_operator())
        .def(py::self *= py::self)
        .def(
            "__IMUL__",
            [](GeneralQuantumOperator& a, const PauliOperator& b) {
                return a *= b;
            },
            py::is_operator())
        .def(
            "__IMUL__",
            [](GeneralQuantumOperator& a, std::complex<double>& b) {
                return a *= b;
            },
            py::is_operator());
    auto mquantum_operator = m.def_submodule("quantum_operator");
    mquantum_operator.def("create_quantum_operator_from_openfermion_file",
        &quantum_operator::
            create_general_quantum_operator_from_openfermion_file,
        py::return_value_policy::take_ownership);
    mquantum_operator.def("create_quantum_operator_from_openfermion_text",
        &quantum_operator::
            create_general_quantum_operator_from_openfermion_text,
        py::return_value_policy::take_ownership);
    mquantum_operator.def("create_split_quantum_operator",
        &quantum_operator::create_split_general_quantum_operator,
        py::return_value_policy::take_ownership);
    mquantum_operator.def(
        "from_json",
        [](const std::string& json) -> GeneralQuantumOperator* {
            return quantum_operator::from_ptree(ptree::from_json(json));
        },
        py::return_value_policy::take_ownership, "from json string",
        py::arg("json"));

    py::class_<HermitianQuantumOperator, GeneralQuantumOperator>(
        m, "Observable")
        .def(py::init<UINT>(), "Constructor", py::arg("qubit_count"))
        .def("add_operator",
            py::overload_cast<const PauliOperator*>(
                &HermitianQuantumOperator::add_operator),
            "Add Pauli operator", py::arg("pauli_operator"))
        .def("add_operator_move", &HermitianQuantumOperator::add_operator_move,
            "Add Pauli operator", py::arg("pauli_operator"))
        .def("add_operator_copy", &HermitianQuantumOperator::add_operator_copy,
            "Add Pauli operator", py::arg("pauli_operator"))

        .def("add_operator",
            py::overload_cast<CPPCTYPE, std::string>(
                &HermitianQuantumOperator::add_operator),
            "Add Pauli operator", py::arg("coef"), py::arg("string"))
        .def("get_qubit_count", &HermitianQuantumOperator::get_qubit_count,
            "Get qubit count")
        .def("get_state_dim", &HermitianQuantumOperator::get_state_dim,
            "Get state dimension")
        .def("get_term_count", &HermitianQuantumOperator::get_term_count,
            "Get count of Pauli terms")
        .def(
            "get_term",
            [](const HermitianQuantumOperator& quantum_operator,
                const UINT index) -> PauliOperator* {
                return quantum_operator.get_term(index)->copy();
            },
            py::return_value_policy::take_ownership, "Get Pauli term",
            py::arg("index"))
        .def(
            "get_expectation_value",
            [](const HermitianQuantumOperator& observable,
                const QuantumStateBase* state) -> double {
                double res = observable.get_expectation_value(state).real();
                return res;
            },
            "Get expectation value", py::arg("state"))
        .def(
            "get_expectation_value_single_thread",
            [](const HermitianQuantumOperator& observable,
                const QuantumStateBase* state) -> double {
                double res =
                    observable.get_expectation_value_single_thread(state)
                        .real();
                return res;
            },
            "Get expectation value", py::arg("state"))
        .def("get_transition_amplitude",
            &HermitianQuantumOperator::get_transition_amplitude,
            "Get transition amplitude", py::arg("state_bra"),
            py::arg("state_ket"))
        .def("add_random_operator",
            py::overload_cast<UINT>(
                &HermitianQuantumOperator::add_random_operator),
            "Add random pauli operator", py::arg("operator_count"))
        .def("add_random_operator",
            py::overload_cast<UINT, UINT>(
                &HermitianQuantumOperator::add_random_operator),
            "Add random pauli operator", py::arg("operator_count"),
            py::arg("seed"))
        .def("solve_ground_state_eigenvalue_by_arnoldi_method",
            &HermitianQuantumOperator::
                solve_ground_state_eigenvalue_by_arnoldi_method,
            "Compute ground state eigenvalue by arnoldi method",
            py::arg("state"), py::arg("iter_count"), py::arg("mu") = 0.0)
        .def("solve_ground_state_eigenvalue_by_power_method",
            &HermitianQuantumOperator::
                solve_ground_state_eigenvalue_by_power_method,
            "Compute ground state eigenvalue by power method", py::arg("state"),
            py::arg("iter_count"), py::arg("mu") = 0.0)
        .def("solve_ground_state_eigenvalue_by_lanczos_method",
            &HermitianQuantumOperator::
                solve_ground_state_eigenvalue_by_lanczos_method,
            "Compute ground state eigenvalue by lanczos method",
            py::arg("state"), py::arg("iter_count"), py::arg("mu") = 0.0)
        .def("apply_to_state",
            py::overload_cast<QuantumStateBase*, const QuantumStateBase&,
                QuantumStateBase*>(
                &HermitianQuantumOperator::apply_to_state, py::const_),
            "Apply observable to `state_to_be_multiplied`. The result is "
            "stored into `dst_state`.",
            py::arg("work_state"), py::arg("state_to_be_multiplied"),
            py::arg("dst_state"))
        .def("__str__", &HermitianQuantumOperator::to_string, "to string");
    auto mobservable = m.def_submodule("observable");
    mobservable.def("create_observable_from_openfermion_file",
        &observable::create_observable_from_openfermion_file,
        py::return_value_policy::take_ownership,
        "Create GeneralQuantumOperator from openfermion file",
        py::arg("file_path"));
    mobservable.def("create_observable_from_openfermion_text",
        &observable::create_observable_from_openfermion_text,
        py::return_value_policy::take_ownership,
        "Create GeneralQuantumOperator from openfermion text", py::arg("text"));
    mobservable.def("create_split_observable",
        &observable::create_split_observable,
        py::return_value_policy::take_ownership);
    mobservable.def(
        "from_json",
        [](const std::string& json) -> HermitianQuantumOperator* {
            return observable::from_ptree(ptree::from_json(json));
        },
        py::return_value_policy::take_ownership, "from json string",
        py::arg("json"));

    py::class_<QuantumStateBase>(m, "QuantumStateBase");
    py::class_<QuantumState, QuantumStateBase>(m, "QuantumState")
        .def(py::init<UINT>(), "Constructor", py::arg("qubit_count"))
        .def(py::init<UINT, bool>(), "Constructor", py::arg("qubit_count"),
            py::arg("use_multi_cpu"))
        .def(
            "set_zero_state", &QuantumState::set_zero_state, "Set state to |0>")
        .def("set_computational_basis", &QuantumState::set_computational_basis,
            "Set state to computational basis", py::arg("comp_basis"))
        .def("set_Haar_random_state",
            py::overload_cast<>(&QuantumState::set_Haar_random_state),
            "Set Haar random state")
        .def("set_Haar_random_state",
            py::overload_cast<UINT>(&QuantumState::set_Haar_random_state),
            "Set Haar random state", py::arg("seed"))
        .def("get_zero_probability", &QuantumState::get_zero_probability,
            "Get probability with which we obtain 0 when we measure a qubit",
            py::arg("index"))
        .def("get_marginal_probability",
            &QuantumState::get_marginal_probability,
            "Get merginal probability for measured values",
            py::arg("measured_values"))
        .def("get_entropy", &QuantumState::get_entropy, "Get entropy")
        .def("get_squared_norm", &QuantumState::get_squared_norm,
            "Get squared norm")
        .def("normalize", &QuantumState::normalize, "Normalize quantum state",
            py::arg("squared_norm"))
        .def("allocate_buffer", &QuantumState::allocate_buffer,
            py::return_value_policy::take_ownership,
            "Allocate buffer with the same size")
        .def("copy", &QuantumState::copy,
            py::return_value_policy::take_ownership, "Create copied instance")
        .def("load",
            py::overload_cast<const QuantumStateBase*>(&QuantumState::load),
            "Load quantum state vector", py::arg("state"))
        .def("load",
            py::overload_cast<const std::vector<CPPCTYPE>&>(
                &QuantumState::load),
            "Load quantum state vector", py::arg("state"))
        .def("get_device_name", &QuantumState::get_device_name,
            "Get allocated device name")
        .def("add_state", &QuantumState::add_state,
            "Add state vector to this state", py::arg("state"))
        .def("multiply_coef", &QuantumState::multiply_coef,
            "Multiply coefficient to this state", py::arg("coef"))
        .def("multiply_elementwise_function",
            &QuantumState::multiply_elementwise_function,
            "Multiply elementwise function", py::arg("func"))
        .def("get_classical_value", &QuantumState::get_classical_value,
            "Get classical value", py::arg("index"))
        .def("set_classical_value", &QuantumState::set_classical_value,
            "Set classical value", py::arg("index"), py::arg("value"))
        .def("to_string", &QuantumState::to_string, "to string")
        .def("sampling", py::overload_cast<UINT>(&QuantumState::sampling),
            "Sampling measurement results", py::arg("sampling_count"))
        .def("sampling", py::overload_cast<UINT, UINT>(&QuantumState::sampling),
            "Sampling measurement results", py::arg("sampling_count"),
            py::arg("random_seed"))
        .def(
            "get_vector",
            [](const QuantumState& state) -> Eigen::VectorXcd {
                Eigen::VectorXcd vec =
                    Eigen::Map<Eigen::VectorXcd>(state.data_cpp(), state.dim);
                return vec;
            },
            "Get state vector")
        .def(
            "get_amplitude",
            [](const QuantumState& state, const UINT index) -> CPPCTYPE {
                return state.data_cpp()[index];
            },
            "Get Amplitude of a specified computational basis",
            py::arg("comp_basis"))
        .def(
            "get_qubit_count",
            [](const QuantumState& state) -> UINT { return state.qubit_count; },
            "Get qubit count")
        .def(
            "__str__", [](const QuantumState& p) { return p.to_string(); },
            "to string")
        .def(
            "to_json",
            [](const QuantumState& state) -> std::string {
                return ptree::to_json(state.to_ptree());
            },
            "to json string")
        .def(py::pickle(
            [](const QuantumState& state) -> std::string {
                return ptree::to_json(state.to_ptree());
            },
            [](std::string json) -> QuantumState* {
                return static_cast<QuantumState*>(
                    state::from_ptree(ptree::from_json(json)));
            }));

    ;

    m.def(
        "StateVector",
        [](const UINT qubit_count) -> QuantumState* {
            auto ptr = new QuantumState(qubit_count);
            return ptr;
        },
        py::return_value_policy::take_ownership, "StateVector");

    py::class_<DensityMatrix, QuantumStateBase>(m, "DensityMatrix")
        .def(py::init<UINT>(), "Constructor", py::arg("qubit_count"))
        .def("set_zero_state", &DensityMatrix::set_zero_state,
            "Set state to |0>")
        .def("set_computational_basis", &DensityMatrix::set_computational_basis,
            "Set state to computational basis", py::arg("comp_basis"))
        .def("set_Haar_random_state",
            py::overload_cast<>(&DensityMatrix::set_Haar_random_state),
            "Set Haar random state")
        .def("set_Haar_random_state",
            py::overload_cast<UINT>(&DensityMatrix::set_Haar_random_state),
            "Set Haar random state", py::arg("seed"))
        .def("get_zero_probability", &DensityMatrix::get_zero_probability,
            "Get probability with which we obtain 0 when we measure a qubit",
            py::arg("index"))
        .def("get_marginal_probability",
            &DensityMatrix::get_marginal_probability,
            "Get merginal probability for measured values",
            py::arg("measured_values"))
        .def("get_entropy", &DensityMatrix::get_entropy, "Get entropy")
        .def("get_squared_norm", &DensityMatrix::get_squared_norm,
            "Get squared norm")
        .def("normalize", &DensityMatrix::normalize, "Normalize quantum state",
            py::arg("squared_norm"))
        .def("allocate_buffer", &DensityMatrix::allocate_buffer,
            py::return_value_policy::take_ownership,
            "Allocate buffer with the same size")
        .def("copy", &DensityMatrix::copy,
            py::return_value_policy::take_ownership, "Create copied insntace")
        .def("load",
            py::overload_cast<const QuantumStateBase*>(&DensityMatrix::load),
            "Load quantum state vector", py::arg("state"))
        .def("load",
            py::overload_cast<const std::vector<CPPCTYPE>&>(
                &DensityMatrix::load),
            "Load quantum state represented as a list", py::arg("state"))
        .def("load",
            py::overload_cast<const ComplexMatrix&>(&DensityMatrix::load),
            "Load quantum state represented as a two-dimensional list",
            py::arg("state"))
        .def("get_device_name", &DensityMatrix::get_device_name,
            "Get allocated device name")
        .def("add_state", &DensityMatrix::add_state,
            "Add state vector to this state", py::arg("state"))
        .def("multiply_coef", &DensityMatrix::multiply_coef,
            "Multiply coefficient to this state", py::arg("coef"))
        .def("get_classical_value", &DensityMatrix::get_classical_value,
            "Get classical value", py::arg("index"))
        .def("set_classical_value", &DensityMatrix::set_classical_value,
            "Set classical value", py::arg("index"), py::arg("value"))
        .def("to_string", &QuantumState::to_string, "to string")
        .def("sampling", py::overload_cast<UINT>(&DensityMatrix::sampling),
            "Sampling measurement results", py::arg("sampling_count"))
        .def("sampling",
            py::overload_cast<UINT, UINT>(&DensityMatrix::sampling),
            "Sampling measurement results", py::arg("sampling_count"),
            py::arg("random_seed"))
        .def(
            "get_matrix",
            [](const DensityMatrix& state) -> Eigen::MatrixXcd {
                Eigen::MatrixXcd mat(state.dim, state.dim);
                CTYPE* ptr = state.data_c();
                for (ITYPE y = 0; y < state.dim; ++y) {
                    for (ITYPE x = 0; x < state.dim; ++x) {
                        mat(y, x) = ptr[y * state.dim + x];
                    }
                }
                return mat;
            },
            "Get density matrix")
        .def(
            "get_qubit_count",
            [](const DensityMatrix& state) -> UINT {
                return state.qubit_count;
            },
            "Get qubit count")
        .def(
            "__str__", [](const DensityMatrix& p) { return p.to_string(); },
            "to string")
        .def(
            "to_json",
            [](const DensityMatrix& state) -> std::string {
                return ptree::to_json(state.to_ptree());
            },
            "to json string")
        .def(py::pickle(
            [](const DensityMatrix& state) -> std::string {
                return ptree::to_json(state.to_ptree());
            },
            [](std::string json) -> DensityMatrix* {
                return static_cast<DensityMatrix*>(
                    state::from_ptree(ptree::from_json(json)));
            }));
    ;

#ifdef _USE_MPI
    m.def("check_build_for_mpi", []() { return true; });
#else
    m.def("check_build_for_mpi", []() { return false; });
#endif

#ifdef _USE_GPU
    py::class_<QuantumStateGpu, QuantumStateBase>(m, "QuantumStateGpu")
        .def(py::init<UINT>(), "Constructor", py::arg("qubit_count"))
        .def(py::init<UINT, UINT>(), "Constructor", py::arg("qubit_count"),
            py::arg("device_number"))
        .def("set_zero_state", &QuantumStateGpu::set_zero_state,
            "Set state to |0>")
        .def("set_computational_basis",
            &QuantumStateGpu::set_computational_basis,
            "Set state to computational basis", py::arg("comp_basis"))
        .def("set_Haar_random_state",
            py::overload_cast<>(&QuantumStateGpu::set_Haar_random_state),
            "Set Haar random state")
        .def("set_Haar_random_state",
            py::overload_cast<UINT>(&QuantumStateGpu::set_Haar_random_state),
            "Set Haar random state", py::arg("seed"))
        .def("get_zero_probability", &QuantumStateGpu::get_zero_probability,
            "Get probability with which we obtain 0 when we measure a qubit",
            py::arg("index"))
        .def("get_marginal_probability",
            &QuantumStateGpu::get_marginal_probability,
            "Get merginal probability for measured values",
            py::arg("measured_values"))
        .def("get_entropy", &QuantumStateGpu::get_entropy, "Get entropy")
        .def("get_squared_norm", &QuantumStateGpu::get_squared_norm,
            "Get squared norm")
        .def("normalize", &QuantumStateGpu::normalize,
            "Normalize quantum state", py::arg("squared_norm"))
        .def("allocate_buffer", &QuantumStateGpu::allocate_buffer,
            py::return_value_policy::take_ownership,
            "Allocate buffer with the same size")
        .def("copy", &QuantumStateGpu::copy,
            py::return_value_policy::take_ownership, "Create copied insntace")
        .def("load",
            py::overload_cast<const QuantumStateBase*>(&QuantumStateGpu::load),
            "Load quantum state vector", py::arg("state"))
        .def("load",
            py::overload_cast<const std::vector<CPPCTYPE>&>(
                &QuantumStateGpu::load),
            "Load quantum state vector represented as a list", py::arg("state"))
        .def("get_device_name", &QuantumStateGpu::get_device_name,
            "Get allocated device name")
        .def("add_state", &QuantumStateGpu::add_state,
            "Add state vector to this state", py::arg("state"))
        .def("multiply_coef", &QuantumStateGpu::multiply_coef,
            "Multiply coefficient to this state", py::arg("coef"))
        .def("multiply_elementwise_function",
            &QuantumStateGpu::multiply_elementwise_function,
            "Multiply elementwise function", py::arg("func"))
        .def("get_classical_value", &QuantumStateGpu::get_classical_value,
            "Get classical value", py::arg("index"))
        .def("set_classical_value", &QuantumStateGpu::set_classical_value,
            "Set classical value", py::arg("index"), py::arg("value"))
        .def("to_string", &QuantumStateGpu::to_string, "to string")
        .def("sampling", py::overload_cast<UINT>(&QuantumStateGpu::sampling),
            "Sampling measurement results", py::arg("sampling_count"))
        .def("sampling",
            py::overload_cast<UINT, UINT>(&QuantumStateGpu::sampling),
            "Sampling measurement results", py::arg("sampling_count"),
            py::arg("random_seed"))
        .def(
            "get_vector",
            [](const QuantumStateGpu& state) -> Eigen::VectorXcd {
                Eigen::VectorXcd vec = Eigen::Map<Eigen::VectorXcd>(
                    state.duplicate_data_cpp(), state.dim);
                return vec;
            },
            "Get state vector")
        .def(
            "get_qubit_count",
            [](const QuantumStateGpu& state) -> UINT {
                return (UINT)state.qubit_count;
            },
            "Get qubit count")
        .def(
            "__str__", [](const QuantumStateGpu& p) { return p.to_string(); },
            "to string");
    ;

    m.def(
        "StateVectorGpu",
        [](const UINT qubit_count) -> QuantumStateGpu* {
            auto ptr = new QuantumStateGpu(qubit_count);
            return ptr;
        },
        py::return_value_policy::take_ownership, "StateVectorGpu");
#endif

    auto mstate = m.def_submodule("state");
    mstate.def("inner_product",
        py::overload_cast<const QuantumState*, const QuantumState*>(
            &state::inner_product),
        "Get inner product", py::arg("state_bra"), py::arg("state_ket"));
#ifdef _USE_GPU
    mstate.def("inner_product",
        py::overload_cast<const QuantumStateGpu*, const QuantumStateGpu*>(
            &state::inner_product),
        "Get inner product", py::arg("state_bra"), py::arg("state_ket"));
#endif
    mstate.def("tensor_product",
        py::overload_cast<const QuantumState*, const QuantumState*>(
            &state::tensor_product),
        py::return_value_policy::take_ownership, "Get tensor product of states",
        py::arg("state_left"), py::arg("state_right"));
    mstate.def("tensor_product",
        py::overload_cast<const DensityMatrix*, const DensityMatrix*>(
            &state::tensor_product),
        py::return_value_policy::take_ownership, "Get tensor product of states",
        py::arg("state_left"), py::arg("state_right"));
    mstate.def("permutate_qubit",
        py::overload_cast<const QuantumState*, std::vector<UINT>>(
            &state::permutate_qubit),
        py::return_value_policy::take_ownership, "Permutate qubits from state",
        py::arg("state"), py::arg("qubit_order"));
    mstate.def("permutate_qubit",
        py::overload_cast<const DensityMatrix*, std::vector<UINT>>(
            &state::permutate_qubit),
        py::return_value_policy::take_ownership, "Permutate qubits from state",
        py::arg("state"), py::arg("qubit_order"));
    mstate.def("drop_qubit", &state::drop_qubit,
        py::return_value_policy::take_ownership, "Drop qubits from state",
        py::arg("state"), py::arg("target"), py::arg("projection"));
    mstate.def("partial_trace",
        py::overload_cast<const QuantumState*, std::vector<UINT>>(
            &state::partial_trace),
        py::return_value_policy::take_ownership, "Take partial trace",
        py::arg("state"), py::arg("target_traceout"));
    mstate.def("partial_trace",
        py::overload_cast<const DensityMatrix*, std::vector<UINT>>(
            &state::partial_trace),
        py::return_value_policy::take_ownership, "Take partial trace",
        py::arg("state"), py::arg("target_traceout"));
    mstate.def("make_superposition", &state::make_superposition,
        py::return_value_policy::take_ownership,
        "Create superposition of states", py::arg("coef1"), py::arg("state1"),
        py::arg("coef2"), py::arg("state2"));
    mstate.def("make_mixture", &state::make_mixture,
        py::return_value_policy::take_ownership, "Create a mixed state",
        py::arg("prob1"), py::arg("state1"), py::arg("prob2"),
        py::arg("state2"));
    mstate.def(
        "from_json",
        [](const std::string& json) -> QuantumStateBase* {
            return state::from_ptree(ptree::from_json(json));
        },
        py::return_value_policy::take_ownership, "from json string",
        py::arg("json"));

    py::class_<QuantumGateBase>(m, "QuantumGateBase")
        .def("update_quantum_state", &QuantumGateBase::update_quantum_state,
            "Update quantum state", py::arg("state"))
        .def("copy", &QuantumGateBase::copy,
            py::return_value_policy::take_ownership, "Create copied instance")
        .def("to_string", &QuantumGateBase::to_string, "to string")
        .def(
            "get_matrix",
            [](const QuantumGateBase& gate) {
                ComplexMatrix mat;
                gate.set_matrix(mat);
                return mat;
            },
            "Get gate matrix")
        .def("__str__", [](const QuantumGateBase& p) { return p.to_string(); })
        .def("get_target_index_list", &QuantumGateBase::get_target_index_list,
            "Get target qubit index list")
        .def("get_control_index_list", &QuantumGateBase::get_control_index_list,
            "Get control qubit index list")
        .def("get_control_value_list", &QuantumGateBase::get_control_value_list,
            "Get control qubit value list")
        .def("get_control_index_value_list",
            &QuantumGateBase::get_control_index_value_list,
            "Get control qubit pair index value list")
        .def("get_name", &QuantumGateBase::get_name, "Get gate name")
        .def("is_commute", &QuantumGateBase::is_commute,
            "Check this gate commutes with a given gate", py::arg("gate"))
        .def("is_Pauli", &QuantumGateBase::is_Pauli,
            "Check this gate is element of Pauli group")
        .def("is_Clifford", &QuantumGateBase::is_Clifford,
            "Check this gate is element of Clifford group")
        .def("is_Gaussian", &QuantumGateBase::is_Gaussian,
            "Check this gate is element of Gaussian group")
        .def("is_parametric", &QuantumGateBase::is_parametric,
            "Check this gate is parametric gate")
        .def("is_diagonal", &QuantumGateBase::is_diagonal,
            "Check the gate matrix is diagonal")
        .def(
            "to_json",
            [](const QuantumGateBase& gate) -> std::string {
                return ptree::to_json(gate.to_ptree());
            },
            "to json string")
        .def("get_inverse", &QuantumGateBase::get_inverse, "get inverse gate");

    py::class_<QuantumGateMatrix, QuantumGateBase>(m, "QuantumGateMatrix")
        .def("add_control_qubit", &QuantumGateMatrix::add_control_qubit,
            "Add control qubit", py::arg("index"), py::arg("control_value"))
        .def("multiply_scalar", &QuantumGateMatrix::multiply_scalar,
            "Multiply scalar value to gate matrix", py::arg("value"));

    py::class_<ClsOneQubitGate, QuantumGateBase>(m, "ClsOneQubitGate");
    py::class_<ClsOneQubitRotationGate, QuantumGateBase>(
        m, "ClsOneQubitRotationGate");
    py::class_<ClsOneControlOneTargetGate, QuantumGateBase>(
        m, "ClsOneControlOneTargetGate");
    py::class_<ClsTwoQubitGate, QuantumGateBase>(m, "ClsTwoQubitGate");
    py::class_<ClsPauliGate, QuantumGateBase>(m, "ClsPauliGate");
    py::class_<ClsPauliRotationGate, QuantumGateBase>(
        m, "ClsPauliRotationGate");
    py::class_<ClsNpairQubitGate, QuantumGateBase>(m, "ClsNpairQubitGate");
    py::class_<ClsNoisyEvolution, QuantumGateBase>(m, "ClsNoisyEvolution");
    py::class_<ClsStateReflectionGate, QuantumGateBase>(
        m, "ClsStateReflectionGate");
    py::class_<ClsReversibleBooleanGate, QuantumGateBase>(
        m, "ClsReversibleBooleanGate");
    py::class_<ClsNoisyEvolution_fast, QuantumGateBase>(
        m, "ClsNoisyEvolution_fast");
    py::class_<QuantumGate_LinearCombination, QuantumGateBase>(
        m, "QuantumGate_LinearCombination")
        .def("get_coef_list", &QuantumGate_LinearCombination::get_coef_list,
            "get coef_list")
        .def("get_gate_list", &QuantumGate_LinearCombination::get_gate_list,
            "get gate_list");
    py::class_<QuantumGate_Probabilistic, QuantumGateBase>(
        m, "QuantumGate_Probabilistic", "QuantumGate_ProbabilisticInstrument")
        .def("get_gate_list", &QuantumGate_Probabilistic::get_gate_list,
            py::return_value_policy::reference, "get_gate_list")
        .def("optimize_ProbablisticGate",
            &QuantumGate_Probabilistic::optimize_ProbablisticGate,
            "optimize_ProbablisticGate")
        .def("get_distribution", &QuantumGate_Probabilistic::get_distribution,
            "get_distribution")
        .def("get_cumulative_distribution",
            &QuantumGate_Probabilistic::get_cumulative_distribution,
            "get_cumulative_distribution");
    py::class_<QuantumGate_CPTP, QuantumGateBase>(
        m, "QuantumGate_CPTP", "QuantumGate_Instrument")
        .def("get_gate_list", &QuantumGate_CPTP::get_gate_list,
            py::return_value_policy::reference, "get_gate_list");
    py::class_<QuantumGate_CP, QuantumGateBase>(m, "QuantumGate_CP")
        .def("get_gate_list", &QuantumGate_CP::get_gate_list,
            py::return_value_policy::reference, "get_gate_list");
    py::class_<QuantumGate_Adaptive, QuantumGateBase>(
        m, "QuantumGate_Adaptive");
    py::class_<QuantumGateDiagonalMatrix, QuantumGateBase>(
        m, "QuantumGateDiagonalMatrix");
    py::class_<QuantumGateSparseMatrix, QuantumGateBase>(
        m, "QuantumGateSparseMatrix");
    auto mgate = m.def_submodule("gate");
    mgate.def("Identity", &gate::Identity,
        py::return_value_policy::take_ownership, "Create identity gate",
        py::arg("index"));
    mgate.def("X", &gate::X, py::return_value_policy::take_ownership,
        "Create Pauli-X gate", py::arg("index"));
    mgate.def("Y", &gate::Y, py::return_value_policy::take_ownership,
        "Create Pauli-Y gate", py::arg("index"));
    mgate.def("Z", &gate::Z, py::return_value_policy::take_ownership,
        "Create Pauli-Z gate", py::arg("index"));
    mgate.def("H", &gate::H, py::return_value_policy::take_ownership,
        "Create Hadamard gate", py::arg("index"));
    mgate.def("S", &gate::S, py::return_value_policy::take_ownership,
        "Create pi/4-phase gate", py::arg("index"));
    mgate.def("Sdag", &gate::Sdag, py::return_value_policy::take_ownership,
        "Create adjoint of pi/4-phase gate", py::arg("index"));
    mgate.def("T", &gate::T, py::return_value_policy::take_ownership,
        "Create pi/8-phase gate", py::arg("index"));
    mgate.def("Tdag", &gate::Tdag, py::return_value_policy::take_ownership,
        "Create adjoint of pi/8-phase gate", py::arg("index"));
    mgate.def("sqrtX", &gate::sqrtX, py::return_value_policy::take_ownership,
        "Create pi/4 Pauli-X rotation gate", py::arg("index"));
    mgate.def("sqrtXdag", &gate::sqrtXdag,
        py::return_value_policy::take_ownership,
        "Create adjoint of pi/4 Pauli-X rotation gate", py::arg("index"));
    mgate.def("sqrtY", &gate::sqrtY, py::return_value_policy::take_ownership,
        "Create pi/4 Pauli-Y rotation gate", py::arg("index"));
    mgate.def("sqrtYdag", &gate::sqrtYdag,
        py::return_value_policy::take_ownership,
        "Create adjoint of pi/4 Pauli-Y rotation gate", py::arg("index"));
    mgate.def("P0", &gate::P0, py::return_value_policy::take_ownership,
        "Create projection gate to |0> subspace", py::arg("index"));
    mgate.def("P1", &gate::P1, py::return_value_policy::take_ownership,
        "Create projection gate to |1> subspace", py::arg("index"));

    mgate.def("U1", &gate::U1, py::return_value_policy::take_ownership,
        "Create QASM U1 gate", py::arg("index"), py::arg("lambda_"));
    mgate.def("U2", &gate::U2, py::return_value_policy::take_ownership,
        "Create QASM U2 gate", py::arg("index"), py::arg("phi"),
        py::arg("lambda_"));
    mgate.def("U3", &gate::U3, py::return_value_policy::take_ownership,
        "Create QASM U3 gate", py::arg("index"), py::arg("theta"),
        py::arg("phi"), py::arg("lambda_"));

    mgate.def("RX", &gate::RX, py::return_value_policy::take_ownership,
        "Create Pauli-X rotation gate", py::arg("index"), py::arg("angle"));
    mgate.def("RY", &gate::RY, py::return_value_policy::take_ownership,
        "Create Pauli-Y rotation gate", py::arg("index"), py::arg("angle"));
    mgate.def("RZ", &gate::RZ, py::return_value_policy::take_ownership,
        "Create Pauli-Z rotation gate", py::arg("index"), py::arg("angle"));
    mgate.def("RotInvX", &gate::RotInvX,
        py::return_value_policy::take_ownership, "Create Pauli-X rotation gate",
        py::arg("index"), py::arg("angle"));
    mgate.def("RotInvY", &gate::RotInvY,
        py::return_value_policy::take_ownership, "Create Pauli-Y rotation gate",
        py::arg("index"), py::arg("angle"));
    mgate.def("RotInvZ", &gate::RotInvZ,
        py::return_value_policy::take_ownership, "Create Pauli-Z rotation gate",
        py::arg("index"), py::arg("angle"));
    mgate.def("RotX", &gate::RotX, py::return_value_policy::take_ownership,
        "Create Pauli-X rotation gate", py::arg("index"), py::arg("angle"));
    mgate.def("RotY", &gate::RotY, py::return_value_policy::take_ownership,
        "Create Pauli-Y rotation gate", py::arg("index"), py::arg("angle"));
    mgate.def("RotZ", &gate::RotZ, py::return_value_policy::take_ownership,
        "Create Pauli-Z rotation gate", py::arg("index"), py::arg("angle"));

    mgate.def("CNOT", &gate::CNOT, py::return_value_policy::take_ownership,
        "Create CNOT gate", py::arg("control"), py::arg("target"));
    mgate.def("CZ", &gate::CZ, py::return_value_policy::take_ownership,
        "Create CZ gate", py::arg("control"), py::arg("target"));
    mgate.def("SWAP", &gate::SWAP, py::return_value_policy::take_ownership,
        "Create SWAP gate", py::arg("target1"), py::arg("target2"));
    mgate.def("FusedSWAP", &gate::FusedSWAP,
        py::return_value_policy::take_ownership, "Create FusedSWAP gate",
        py::arg("target1"), py::arg("target2"), py::arg("block_size"));

    mgate.def(
        "TOFFOLI",
        [](UINT control_index1, UINT control_index2,
            UINT target_index) -> QuantumGateMatrix* {
            auto ptr = gate::X(target_index);
            auto toffoli = gate::to_matrix_gate(ptr);
            toffoli->add_control_qubit(control_index1, 1);
            toffoli->add_control_qubit(control_index2, 1);
            delete ptr;
            return toffoli;
        },
        py::return_value_policy::take_ownership, "Create TOFFOLI gate",
        py::arg("control1"), py::arg("control2"), py::arg("target"));
    mgate.def(
        "FREDKIN",
        [](UINT control_index, UINT target_index1,
            UINT target_index2) -> QuantumGateMatrix* {
            auto ptr = gate::SWAP(target_index1, target_index2);
            auto fredkin = gate::to_matrix_gate(ptr);
            fredkin->add_control_qubit(control_index, 1);
            delete ptr;
            return fredkin;
        },
        py::return_value_policy::take_ownership, "Create FREDKIN gate",
        py::arg("control"), py::arg("target1"), py::arg("target2"));

    mgate.def(
        "Pauli",
        [](std::vector<UINT> target_qubit_index_list,
            std::vector<UINT> pauli_ids) -> ClsPauliGate* {
            if (target_qubit_index_list.size() != pauli_ids.size())
                throw std::invalid_argument(
                    "Size of qubit list and pauli list must be equal.");
            auto ptr = gate::Pauli(target_qubit_index_list, pauli_ids);
            return ptr;
        },
        py::return_value_policy::take_ownership,
        "Create multi-qubit Pauli gate", py::arg("index_list"),
        py::arg("pauli_ids"));
    mgate.def(
        "PauliRotation",
        [](std::vector<UINT> target_qubit_index_list,
            std::vector<UINT> pauli_ids,
            double angle) -> ClsPauliRotationGate* {
            if (target_qubit_index_list.size() != pauli_ids.size())
                throw std::invalid_argument(
                    "Size of qubit list and pauli list must be equal.");
            auto ptr =
                gate::PauliRotation(target_qubit_index_list, pauli_ids, angle);
            return ptr;
        },
        py::return_value_policy::take_ownership,
        "Create multi-qubit Pauli rotation", py::arg("index_list"),
        py::arg("pauli_ids"), py::arg("angle"));

    mgate.def(
        "DenseMatrix",
        [](UINT target_qubit_index,
            ComplexMatrix matrix) -> QuantumGateMatrix* {
            if (matrix.rows() != 2 || matrix.cols() != 2)
                throw std::invalid_argument("matrix dims is not 2x2.");
            auto ptr = gate::DenseMatrix(target_qubit_index, matrix);
            return ptr;
        },
        py::return_value_policy::take_ownership, "Create dense matrix gate",
        py::arg("index"), py::arg("matrix"));
    mgate.def(
        "DenseMatrix",
        [](std::vector<UINT> target_qubit_index_list,
            ComplexMatrix matrix) -> QuantumGateMatrix* {
            const ITYPE dim = 1ULL << target_qubit_index_list.size();
            if (matrix.rows() != dim || matrix.cols() != dim)
                throw std::invalid_argument("matrix dims is not consistent.");
            auto ptr = gate::DenseMatrix(target_qubit_index_list, matrix);
            return ptr;
        },
        py::return_value_policy::take_ownership, "Create dense matrix gate",
        py::arg("index_list"), py::arg("matrix"));
    mgate.def(
        "SparseMatrix",
        [](std::vector<UINT> target_qubit_index_list,
            SparseComplexMatrix matrix) -> QuantumGateSparseMatrix* {
            const ITYPE dim = 1ULL << target_qubit_index_list.size();
            if (matrix.rows() != dim || matrix.cols() != dim)
                throw std::invalid_argument("matrix dims is not consistent.");
            auto ptr = gate::SparseMatrix(target_qubit_index_list, matrix);
            return ptr;
        },
        py::return_value_policy::take_ownership, "Create sparse matrix gate",
        py::arg("index_list"), py::arg("matrix"));
    mgate.def(
        "DiagonalMatrix",
        [](std::vector<UINT> target_qubit_index_list,
            ComplexVector diagonal_element) -> QuantumGateDiagonalMatrix* {
            const ITYPE dim = 1ULL << target_qubit_index_list.size();
            if (diagonal_element.size() != dim)
                throw std::invalid_argument(
                    "dim of diagonal element is not consistent.");
            auto ptr =
                gate::DiagonalMatrix(target_qubit_index_list, diagonal_element);
            return ptr;
        },
        py::return_value_policy::take_ownership, "Create diagonal matrix gate",
        py::arg("index_list"), py::arg("diagonal_element"));

    mgate.def("RandomUnitary",
        py::overload_cast<std::vector<UINT>>(&gate::RandomUnitary),
        py::return_value_policy::take_ownership, "Create random unitary gate",
        py::arg("index_list"));
    mgate.def("RandomUnitary",
        py::overload_cast<std::vector<UINT>, UINT>(&gate::RandomUnitary),
        py::return_value_policy::take_ownership, "Create random unitary gate",
        py::arg("index_list"), py::arg("seed"));
    mgate.def("ReversibleBoolean", &gate::ReversibleBoolean,
        py::return_value_policy::take_ownership,
        "Create reversible boolean gate", py::arg("index_list"),
        py::arg("func"));
    mgate.def("StateReflection", &gate::StateReflection,
        py::return_value_policy::take_ownership, "Create state reflection gate",
        py::arg("state"));
    mgate.def("LinearCombination", &gate::LinearCombination,
        py::return_value_policy::take_ownership,
        "Create linear combination gate", py::arg("coefs"),
        py::arg("gate_list"));

    mgate.def("BitFlipNoise",
        py::overload_cast<UINT, double>(&gate::BitFlipNoise),
        py::return_value_policy::take_ownership, "Create bit-flip noise",
        py::arg("index"), py::arg("prob"));
    mgate.def("BitFlipNoise",
        py::overload_cast<UINT, double, UINT>(&gate::BitFlipNoise),
        py::return_value_policy::take_ownership, "Create bit-flip noise",
        py::arg("index"), py::arg("prob"), py::arg("seed"));
    mgate.def("DephasingNoise",
        py::overload_cast<UINT, double>(&gate::DephasingNoise),
        py::return_value_policy::take_ownership, "Create dephasing noise",
        py::arg("index"), py::arg("prob"));
    mgate.def("DephasingNoise",
        py::overload_cast<UINT, double, UINT>(&gate::DephasingNoise),
        py::return_value_policy::take_ownership, "Create dephasing noise",
        py::arg("index"), py::arg("prob"), py::arg("seed"));
    mgate.def("IndependentXZNoise",
        py::overload_cast<UINT, double>(&gate::IndependentXZNoise),
        py::return_value_policy::take_ownership, "Create independent XZ noise",
        py::arg("index"), py::arg("prob"));
    mgate.def("IndependentXZNoise",
        py::overload_cast<UINT, double, UINT>(&gate::IndependentXZNoise),
        py::return_value_policy::take_ownership, "Create independent XZ noise",
        py::arg("index"), py::arg("prob"), py::arg("seed"));
    mgate.def("DepolarizingNoise",
        py::overload_cast<UINT, double>(&gate::DepolarizingNoise),
        py::return_value_policy::take_ownership, "Create depolarizing noise",
        py::arg("index"), py::arg("prob"));
    mgate.def("DepolarizingNoise",
        py::overload_cast<UINT, double, UINT>(&gate::DepolarizingNoise),
        py::return_value_policy::take_ownership, "Create depolarizing noise",
        py::arg("index"), py::arg("prob"), py::arg("seed"));
    mgate.def("TwoQubitDepolarizingNoise",
        py::overload_cast<UINT, UINT, double>(&gate::TwoQubitDepolarizingNoise),
        py::return_value_policy::take_ownership,
        "Create two-qubit depolarizing noise", py::arg("index1"),
        py::arg("index2"), py::arg("prob"));
    mgate.def("TwoQubitDepolarizingNoise",
        py::overload_cast<UINT, UINT, double, UINT>(
            &gate::TwoQubitDepolarizingNoise),
        py::return_value_policy::take_ownership,
        "Create two-qubit depolarizing noise", py::arg("index1"),
        py::arg("index2"), py::arg("prob"), py::arg("seed"));
    mgate.def("AmplitudeDampingNoise",
        py::overload_cast<UINT, double>(&gate::AmplitudeDampingNoise),
        py::return_value_policy::take_ownership,
        "Create amplitude damping noise", py::arg("index"), py::arg("prob"));
    mgate.def("AmplitudeDampingNoise",
        py::overload_cast<UINT, double, UINT>(&gate::AmplitudeDampingNoise),
        py::return_value_policy::take_ownership,
        "Create amplitude damping noise", py::arg("index"), py::arg("prob"),
        py::arg("seed"));
    mgate.def("Measurement", py::overload_cast<UINT, UINT>(&gate::Measurement),
        py::return_value_policy::take_ownership, "Create measurement gate",
        py::arg("index"), py::arg("register"));
    mgate.def("Measurement",
        py::overload_cast<UINT, UINT, UINT>(&gate::Measurement),
        py::return_value_policy::take_ownership, "Create measurement gate",
        py::arg("index"), py::arg("register"), py::arg("seed"));
    mgate.def("MultiQubitPauliMeasurement",
        py::overload_cast<const std::vector<UINT>&, const std::vector<UINT>&,
            UINT>(&gate::MultiQubitPauliMeasurement),
        py::return_value_policy::take_ownership,
        "Create multi qubit pauli measurement gate",
        py::arg("target_index_list"), py::arg("pauli_id_list"),
        py::arg("classical_register_address"));
    mgate.def("MultiQubitPauliMeasurement",
        py::overload_cast<const std::vector<UINT>&, const std::vector<UINT>&,
            UINT, UINT>(&gate::MultiQubitPauliMeasurement),
        py::return_value_policy::take_ownership,
        "Create multi qubit pauli measurement gate",
        py::arg("target_index_list"), py::arg("pauli_id_list"),
        py::arg("classical_register_address"), py::arg("seed"));

    mgate.def("merge",
        py::overload_cast<const QuantumGateBase*, const QuantumGateBase*>(
            &gate::merge),
        py::return_value_policy::take_ownership, "Merge two quantum gates",
        py::arg("gate1"), py::arg("gate2"));
    mgate.def("merge",
        py::overload_cast<std::vector<QuantumGateBase*>>(&gate::merge),
        py::return_value_policy::take_ownership, "Merge quantum gate list",
        py::arg("gate_list"));
    mgate.def("add",
        py::overload_cast<const QuantumGateBase*, const QuantumGateBase*>(
            &gate::add),
        py::return_value_policy::take_ownership, "Add quantum gate matrices",
        py::arg("gate1"), py::arg("gate2"));
    mgate.def("add",
        py::overload_cast<std::vector<QuantumGateBase*>>(&gate::add),
        py::return_value_policy::take_ownership, "Add quantum gate matrices",
        py::arg("gate_list"));
    mgate.def("to_matrix_gate", &gate::to_matrix_gate,
        py::return_value_policy::take_ownership,
        "Convert named gate to matrix gate", py::arg("gate"));

    mgate.def("Probabilistic", &gate::Probabilistic,
        py::return_value_policy::take_ownership, "Create probabilistic gate",
        py::arg("prob_list"), py::arg("gate_list"));
    mgate.def("ProbabilisticInstrument", &gate::ProbabilisticInstrument,
        py::return_value_policy::take_ownership,
        "Create probabilistic instrument gate", py::arg("prob_list"),
        py::arg("gate_list"), py::arg("register"));
    mgate.def("CPTP", &gate::CPTP, py::return_value_policy::take_ownership,
        "Create completely-positive trace preserving map",
        py::arg("kraus_list"));
    mgate.def("CP", &gate::CP, py::return_value_policy::take_ownership,
        "Create completely-positive map", py::arg("kraus_list"),
        py::arg("state_normalize"), py::arg("probability_normalize"),
        py::arg("assign_zero_if_not_matched"));
    mgate.def("Instrument", &gate::Instrument,
        py::return_value_policy::take_ownership, "Create instruments",
        py::arg("kraus_list"), py::arg("register"));
    mgate.def("Adaptive",
        py::overload_cast<QuantumGateBase*,
            std::function<bool(const std::vector<UINT>&)>>(&gate::Adaptive),
        py::return_value_policy::take_ownership, "Create adaptive gate",
        py::arg("gate"), py::arg("condition"));
    mgate.def("Adaptive",
        py::overload_cast<QuantumGateBase*,
            std::function<bool(const std::vector<UINT>&, UINT)>, UINT>(
            &gate::Adaptive),
        py::return_value_policy::take_ownership, "Create adaptive gate",
        py::arg("gate"), py::arg("condition"), py::arg("id"));
    mgate.def("NoisyEvolution", &gate::NoisyEvolution,
        py::return_value_policy::take_ownership, "Create noisy evolution",
        py::arg("hamiltonian"), py::arg("c_ops"), py::arg("time"),
        py::arg("dt"));
    mgate.def("NoisyEvolution_fast", &gate::NoisyEvolution_fast,
        py::return_value_policy::take_ownership,
        "Create noisy evolution fast version", py::arg("hamiltonian"),
        py::arg("c_ops"), py::arg("time"));

    py::class_<QuantumGate_SingleParameter, QuantumGateBase>(
        m, "QuantumGate_SingleParameter")
        .def("get_parameter_value",
            &QuantumGate_SingleParameter::get_parameter_value,
            "Get parameter value")
        .def("set_parameter_value",
            &QuantumGate_SingleParameter::set_parameter_value,
            "Set parameter value", py::arg("value"))
        .def("copy", &QuantumGate_SingleParameter::copy,
            py::return_value_policy::take_ownership, "Create copied instance");
    mgate.def("ParametricRX", &gate::ParametricRX,
        py::return_value_policy::take_ownership,
        "Create parametric Pauli-X rotation gate", py::arg("index"),
        py::arg("angle"));
    mgate.def("ParametricRY", &gate::ParametricRY,
        py::return_value_policy::take_ownership,
        "Create parametric Pauli-Y rotation gate", py::arg("index"),
        py::arg("angle"));
    mgate.def("ParametricRZ", &gate::ParametricRZ,
        py::return_value_policy::take_ownership,
        "Create parametric Pauli-Z rotation gate", py::arg("index"),
        py::arg("angle"));
    mgate.def("ParametricPauliRotation", &gate::ParametricPauliRotation,
        py::return_value_policy::take_ownership,
        "Create parametric multi-qubit Pauli rotation gate",
        py::arg("index_list"), py::arg("pauli_ids"), py::arg("angle"));
    mgate.def(
        "from_json",
        [](std::string json) -> QuantumGateBase* {
            boost::property_tree::ptree pt = ptree::from_json(json);
            if (pt.get<std::string>("name").substr(0, 10) == "Parametric") {
                return gate::parametric_gate_from_ptree(pt);
            }
            return gate::from_ptree(ptree::from_json(json));
        },
        py::return_value_policy::take_ownership, "from json string");

    m.def("to_general_quantum_operator", &to_general_quantum_operator,
        py::arg("gate"), py::arg("qubits"), py::arg("tol"));

    py::class_<QuantumCircuit>(m, "QuantumCircuit")
        .def(py::init<UINT>(), "Constructor", py::arg("qubit_count"))
        .def("copy", &QuantumCircuit::copy,
            py::return_value_policy::take_ownership, "Create copied instance")
        // In order to avoid double release, we force using add_gate_copy in
        // python
        //.def("add_gate_consume", (void
        //(QuantumCircuit::*)(QuantumGateBase*))&QuantumCircuit::add_gate,
        //"Add
        // gate and take ownership", py::arg("gate"))
        // .def("add_gate_consume",
        //(void (QuantumCircuit::*)(QuantumGateBase*,
        // UINT))&QuantumCircuit::add_gate, "Add gate and take ownership",
        // py::arg("gate"), py::arg("position"))
        .def("add_gate",
            py::overload_cast<const QuantumGateBase*>(
                &QuantumCircuit::add_gate_copy),
            "Add gate with copy", py::arg("gate"))
        .def("add_gate",
            py::overload_cast<const QuantumGateBase*, UINT>(
                &QuantumCircuit::add_gate_copy),
            "Add gate with copy", py::arg("gate"), py::arg("position"))
        .def("add_noise_gate", &QuantumCircuit::add_noise_gate_copy,
            "Add noise gate with copy", py::arg("gate"), py::arg("NoiseType"),
            py::arg("NoiseProbability"))
        .def("remove_gate", &QuantumCircuit::remove_gate, "Remove gate",
            py::arg("position"))

        .def(
            "get_gate",
            [](const QuantumCircuit& circuit, UINT index) -> QuantumGateBase* {
                if (index >= circuit.gate_list.size()) {
                    std::cerr << "Error: QuantumCircuit::get_gate(const "
                                 "QuantumCircuit&, UINT): gate index is "
                                 "out of range"
                              << std::endl;
                    return NULL;
                }
                return circuit.gate_list[index]->copy();
            },
            py::return_value_policy::take_ownership, "Get gate instance",
            py::arg("position"))
        .def(
            "merge_circuit", &QuantumCircuit::merge_circuit, py::arg("circuit"))
        .def(
            "get_gate_count",
            [](const QuantumCircuit& circuit) -> UINT {
                return (UINT)circuit.gate_list.size();
            },
            "Get gate count")
        .def(
            "get_qubit_count",
            [](const QuantumCircuit& circuit) -> UINT {
                return circuit.qubit_count;
            },
            "Get qubit count")

        .def("update_quantum_state",
            py::overload_cast<QuantumStateBase*>(
                &QuantumCircuit::update_quantum_state),
            "Update quantum state", py::arg("state"))
        .def("update_quantum_state",
            py::overload_cast<QuantumStateBase*, UINT, UINT>(
                &QuantumCircuit::update_quantum_state),
            "Update quantum state", py::arg("state"), py::arg("start"),
            py::arg("end"))
        .def("update_quantum_state",
            py::overload_cast<QuantumStateBase*, UINT>(
                &QuantumCircuit::update_quantum_state),
            "Update quantum state", py::arg("state"), py::arg("seed"))
        .def("update_quantum_state",
            py::overload_cast<QuantumStateBase*, UINT, UINT, UINT>(
                &QuantumCircuit::update_quantum_state),
            "Update quantum state", py::arg("state"), py::arg("start"),
            py::arg("end"), py::arg("seed"))
        .def("calculate_depth", &QuantumCircuit::calculate_depth,
            "Calculate depth of circuit")
        .def("to_string", &QuantumCircuit::to_string,
            "Get string representation")

        .def("add_X_gate", &QuantumCircuit::add_X_gate, "Add Pauli-X gate",
            py::arg("index"))
        .def("add_Y_gate", &QuantumCircuit::add_Y_gate, "Add Pauli-Y gate",
            py::arg("index"))
        .def("add_Z_gate", &QuantumCircuit::add_Z_gate, "Add Pauli-Z gate",
            py::arg("index"))
        .def("add_H_gate", &QuantumCircuit::add_H_gate, "Add Hadamard gate",
            py::arg("index"))
        .def("add_S_gate", &QuantumCircuit::add_S_gate, "Add pi/4 phase gate",
            py::arg("index"))
        .def("add_Sdag_gate", &QuantumCircuit::add_Sdag_gate,
            "Add adjoint of pi/4 phsae gate", py::arg("index"))
        .def("add_T_gate", &QuantumCircuit::add_T_gate, "Add pi/8 phase gate",
            py::arg("index"))
        .def("add_Tdag_gate", &QuantumCircuit::add_Tdag_gate,
            "Add adjoint of pi/8 phase gate", py::arg("index"))
        .def("add_sqrtX_gate", &QuantumCircuit::add_sqrtX_gate,
            "Add pi/4 Pauli-X rotation gate", py::arg("index"))
        .def("add_sqrtXdag_gate", &QuantumCircuit::add_sqrtXdag_gate,
            "Add adjoint of pi/4 Pauli-X rotation gate", py::arg("index"))
        .def("add_sqrtY_gate", &QuantumCircuit::add_sqrtY_gate,
            "Add pi/4 Pauli-Y rotation gate", py::arg("index"))
        .def("add_sqrtYdag_gate", &QuantumCircuit::add_sqrtYdag_gate,
            "Add adjoint of pi/4 Pauli-Y rotation gate", py::arg("index"))
        .def("add_P0_gate", &QuantumCircuit::add_P0_gate,
            "Add projection gate to |0> subspace", py::arg("index"))
        .def("add_P1_gate", &QuantumCircuit::add_P1_gate,
            "Add projection gate to |1> subspace", py::arg("index"))

        .def("add_CNOT_gate", &QuantumCircuit::add_CNOT_gate, "Add CNOT gate",
            py::arg("control"), py::arg("target"))
        .def("add_CZ_gate", &QuantumCircuit::add_CZ_gate, "Add CZ gate",
            py::arg("control"), py::arg("target"))
        .def("add_SWAP_gate", &QuantumCircuit::add_SWAP_gate, "Add SWAP gate",
            py::arg("target1"), py::arg("target2"))
        .def("add_FusedSWAP_gate", &QuantumCircuit::add_FusedSWAP_gate,
            "Add FusedSWAP gate", py::arg("target1"), py::arg("target2"),
            py::arg("block_size"))

        .def("add_RX_gate", &QuantumCircuit::add_RX_gate, R"(
Add Pauli-X rotation gate

Notes
-----
Matrix Representation

.. math::
    R_X(\theta) = \exp(i\frac{\theta}{2} X) =
        \begin{pmatrix}
        \cos(\frac{\theta}{2})  & i\sin(\frac{\theta}{2}) \\
        i\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
        \end{pmatrix}
)",
            py::arg("index"), py::arg("angle"))
        .def("add_RY_gate", &QuantumCircuit::add_RY_gate, R"(
Add Pauli-Y rotation gate

Notes
-----
Matrix Representation

.. math::
    R_Y(\theta) = \exp(i\frac{\theta}{2} Y) =
        \begin{pmatrix}
        \cos(\frac{\theta}{2})  & \sin(\frac{\theta}{2}) \\
        -\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
        \end{pmatrix}
)",
            py::arg("index"), py::arg("angle"))
        .def("add_RZ_gate", &QuantumCircuit::add_RZ_gate, R"(
Add Pauli-Z rotation gate

Notes
-----
Matrix Representation

.. math::
    R_Z(\theta) = \exp(i\frac{\theta}{2} Z) =
        \begin{pmatrix}
        e^{i\frac{\theta}{2}} & 0 \\
        0 & e^{-i\frac{\theta}{2}}
        \end{pmatrix}
)",
            py::arg("index"), py::arg("angle"))
        .def("add_RotInvX_gate", &QuantumCircuit::add_RotInvX_gate, R"(
Add Pauli-X rotation gate

Notes
-----
Matrix Representation

.. math::
    R_X(\theta) = \exp(i\frac{\theta}{2} X) =
        \begin{pmatrix}
        \cos(\frac{\theta}{2})  & i\sin(\frac{\theta}{2}) \\
        i\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
        \end{pmatrix}
)",
            py::arg("index"), py::arg("angle"))
        .def("add_RotInvY_gate", &QuantumCircuit::add_RotInvY_gate, R"(
Add Pauli-Y rotation gate

Notes
-----
Matrix Representation

.. math::
    R_Y(\theta) = \exp(i\frac{\theta}{2} Y) =
        \begin{pmatrix}
        \cos(\frac{\theta}{2})  & \sin(\frac{\theta}{2}) \\
        -\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
        \end{pmatrix}
)",
            py::arg("index"), py::arg("angle"))
        .def("add_RotInvZ_gate", &QuantumCircuit::add_RotInvZ_gate, R"(
Add Pauli-Z rotation gate

Notes
-----
Matrix Representation

.. math::
    R_Z(\theta) = \exp(i\frac{\theta}{2} Z) =
        \begin{pmatrix}
        e^{i\frac{\theta}{2}} & 0 \\
        0 & e^{-i\frac{\theta}{2}}
        \end{pmatrix}
)",
            py::arg("index"), py::arg("angle"))
        .def("add_RotX_gate", &QuantumCircuit::add_RotX_gate, R"(
Add Pauli-X rotation gate

Notes
-----
Matrix Representation

.. math::
    RotX(\theta) = \exp(-i\frac{\theta}{2} X) =
        \begin{pmatrix}
        \cos(\frac{\theta}{2})  & -i\sin(\frac{\theta}{2}) \\
        -i\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
        \end{pmatrix}
)",
            py::arg("index"), py::arg("angle"))
        .def("add_RotY_gate", &QuantumCircuit::add_RotY_gate, R"(
Add Pauli-Y rotation gate

Notes
-----
Matrix Representation

.. math::
    RotY(\theta) = \exp(-i\frac{\theta}{2} Y) =
        \begin{pmatrix}
        \cos(\frac{\theta}{2})  & -\sin(\frac{\theta}{2}) \\
        \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
        \end{pmatrix}
)",
            py::arg("index"), py::arg("angle"))
        .def("add_RotZ_gate", &QuantumCircuit::add_RotZ_gate, R"(
Add Pauli-Z rotation gate

Notes
-----
Matrix Representation

.. math::
    RotZ(\theta) = \exp(-i\frac{\theta}{2} Z) =
        \begin{pmatrix}
        e^{-i\frac{\theta}{2}} & 0 \\
        0 & e^{i\frac{\theta}{2}}
        \end{pmatrix}
)",
            py::arg("index"), py::arg("angle"))
        .def("add_U1_gate", &QuantumCircuit::add_U1_gate, "Add QASM U1 gate",
            py::arg("index"), py::arg("lambda_"))
        .def("add_U2_gate", &QuantumCircuit::add_U2_gate, "Add QASM U2 gate",
            py::arg("index"), py::arg("phi"), py::arg("lambda_"))
        .def("add_U3_gate", &QuantumCircuit::add_U3_gate, "Add QASM U3 gate",
            py::arg("index"), py::arg("theta"), py::arg("phi"),
            py::arg("lambda_"))

        .def("add_multi_Pauli_gate",
            py::overload_cast<std::vector<UINT>, std::vector<UINT>>(
                &QuantumCircuit::add_multi_Pauli_gate),
            "Add multi-qubit Pauli gate", py::arg("index_list"),
            py::arg("pauli_ids"))
        .def("add_multi_Pauli_gate",
            py::overload_cast<const PauliOperator&>(
                &QuantumCircuit::add_multi_Pauli_gate),
            "Add multi-qubit Pauli gate", py::arg("pauli"))
        .def("add_multi_Pauli_rotation_gate",
            py::overload_cast<std::vector<UINT>, std::vector<UINT>, double>(
                &QuantumCircuit::add_multi_Pauli_rotation_gate),
            "Add multi-qubit Pauli rotation gate", py::arg("index_list"),
            py::arg("pauli_ids"), py::arg("angle"))
        .def("add_multi_Pauli_rotation_gate",
            py::overload_cast<const PauliOperator&>(
                &QuantumCircuit::add_multi_Pauli_rotation_gate),
            "Add multi-qubit Pauli rotation gate", py::arg("pauli"))
        .def("add_dense_matrix_gate",
            py::overload_cast<UINT, const ComplexMatrix&>(
                &QuantumCircuit::add_dense_matrix_gate),
            "Add dense matrix gate", py::arg("index"), py::arg("matrix"))
        .def("add_dense_matrix_gate",
            py::overload_cast<std::vector<UINT>, const ComplexMatrix&>(
                &QuantumCircuit::add_dense_matrix_gate),
            "Add dense matrix gate", py::arg("index_list"), py::arg("matrix"))
        .def("add_random_unitary_gate",
            py::overload_cast<std::vector<UINT>>(
                &QuantumCircuit::add_random_unitary_gate),
            "Add random unitary gate", py::arg("index_list"))
        .def("add_random_unitary_gate",
            py::overload_cast<std::vector<UINT>, UINT>(
                &QuantumCircuit::add_random_unitary_gate),
            "Add random unitary gate", py::arg("index_list"), py::arg("seed"))
        .def("add_diagonal_observable_rotation_gate",
            &QuantumCircuit::add_diagonal_observable_rotation_gate,
            "Add diagonal observable rotation gate", py::arg("observable"),
            py::arg("angle"))
        .def("add_observable_rotation_gate",
            &QuantumCircuit::add_observable_rotation_gate,
            "Add observable rotation gate", py::arg("observable"),
            py::arg("angle"), py::arg("repeat"))
        .def("to_json",
            [](const QuantumCircuit& c) -> std::string {
                return ptree::to_json(c.to_ptree());
            })
        .def("get_inverse", &QuantumCircuit::get_inverse, "get inverse circuit")
        .def(
            "__str__", [](const QuantumCircuit& p) { return p.to_string(); },
            "to string")
        .def(py::pickle(
            [](const QuantumCircuit& c) -> std::string {
                return ptree::to_json(c.to_ptree());
            },
            [](std::string json) {
                boost::property_tree::ptree pt = ptree::from_json(json);
                return circuit::from_ptree(pt);
            }));

    py::class_<ParametricQuantumCircuit, QuantumCircuit>(
        m, "ParametricQuantumCircuit")
        .def(py::init<UINT>(), "Constructor", py::arg("qubit_count"))
        .def("copy", &ParametricQuantumCircuit::copy,
            py::return_value_policy::take_ownership, "Create copied instance")
        .def("add_parametric_gate",
            py::overload_cast<QuantumGate_SingleParameter*>(
                &ParametricQuantumCircuit::add_parametric_gate_copy),
            "Add parametric gate", py::arg("gate"))
        .def("add_parametric_gate",
            py::overload_cast<QuantumGate_SingleParameter*, UINT>(
                &ParametricQuantumCircuit::add_parametric_gate_copy),
            "Add parametric gate", py::arg("gate"), py::arg("position"))
        .def("add_gate",
            py::overload_cast<const QuantumGateBase*>(
                &ParametricQuantumCircuit::add_gate_copy),
            "Add gate", py::arg("gate"))
        .def("add_gate",
            py::overload_cast<const QuantumGateBase*, UINT>(
                &ParametricQuantumCircuit::add_gate_copy),
            "Add gate", py::arg("gate"), py::arg("position"))
        .def("get_parameter_count",
            &ParametricQuantumCircuit::get_parameter_count,
            "Get parameter count")
        .def("get_parameter", &ParametricQuantumCircuit::get_parameter,
            "Get parameter", py::arg("index"))
        .def("set_parameter", &ParametricQuantumCircuit::set_parameter,
            "Set parameter", py::arg("index"), py::arg("parameter"))
        .def("get_parametric_gate_position",
            &ParametricQuantumCircuit::get_parametric_gate_position,
            "Get parametric gate position", py::arg("index"))
        .def("remove_gate", &ParametricQuantumCircuit::remove_gate,
            "Remove gate", py::arg("position"))
        .def("merge_circuit", &ParametricQuantumCircuit::merge_circuit,
            "Merge another ParametricQuantumCircuit", py::arg("circuit"))

        .def("add_parametric_RX_gate",
            &ParametricQuantumCircuit::add_parametric_RX_gate,
            "Add parametric Pauli-X rotation gate", py::arg("index"),
            py::arg("angle"))
        .def("add_parametric_RY_gate",
            &ParametricQuantumCircuit::add_parametric_RY_gate,
            "Add parametric Pauli-Y rotation gate", py::arg("index"),
            py::arg("angle"))
        .def("add_parametric_RZ_gate",
            &ParametricQuantumCircuit::add_parametric_RZ_gate,
            "Add parametric Pauli-Z rotation gate", py::arg("index"),
            py::arg("angle"))
        .def("add_parametric_multi_Pauli_rotation_gate",
            &ParametricQuantumCircuit::add_parametric_multi_Pauli_rotation_gate,
            "Add parametric multi-qubit Pauli rotation gate",
            py::arg("index_list"), py::arg("pauli_ids"), py::arg("angle"))

        .def("backprop", &ParametricQuantumCircuit::backprop, "Do backprop",
            py::arg("obs"))
        .def("backprop_inner_product",
            &ParametricQuantumCircuit::backprop_inner_product,
            "Do backprop with innder product", py::arg("state"))

        .def(
            "__str__",
            [](const ParametricQuantumCircuit& p) { return p.to_string(); },
            "to string")
        .def(py::pickle(
            [](const ParametricQuantumCircuit& c) -> std::string {
                return ptree::to_json(c.to_ptree());
            },
            [](std::string json) {
                boost::property_tree::ptree pt = ptree::from_json(json);
                return circuit::parametric_circuit_from_ptree(pt);
            }));

    py::class_<GradCalculator>(m, "GradCalculator")
        .def(py::init<>())
        .def("calculate_grad",
            py::overload_cast<ParametricQuantumCircuit&, Observable&>(
                &GradCalculator::calculate_grad),
            "Calculate Grad", py::arg("parametric_circuit"),
            py::arg("observable"))
        .def("calculate_grad",
            py::overload_cast<ParametricQuantumCircuit&, Observable&,
                std::vector<double>>(&GradCalculator::calculate_grad),
            "Calculate Grad", py::arg("parametric_circuit"),
            py::arg("observable"), py::arg("angles_of_gates"));

    auto mcircuit = m.def_submodule("circuit");
    mcircuit.def(
        "from_json",
        [](const std::string& json) -> QuantumCircuit* {
            boost::property_tree::ptree pt = ptree::from_json(json);
            if (pt.get<std::string>("name") == "ParametricQuantumCircuit") {
                return circuit::parametric_circuit_from_ptree(pt);
            } else {
                return circuit::from_ptree(pt);
            }
        },
        "from json string", py::return_value_policy::take_ownership);

    py::class_<QuantumCircuitOptimizer>(mcircuit, "QuantumCircuitOptimizer")
        .def(py::init<UINT>(), "Constructor", py::arg("mpi_size") = 0)
        .def("optimize", &QuantumCircuitOptimizer::optimize,
            "Optimize quantum circuit", py::arg("circuit"),
            py::arg("block_size"), py::arg("swap_level") = 0)
        .def("optimize_light", &QuantumCircuitOptimizer::optimize_light,
            "Optimize quantum circuit with light method", py::arg("circuit"),
            py::arg("swap_level") = 0)
        .def("merge_all", &QuantumCircuitOptimizer::merge_all,
            py::return_value_policy::take_ownership, py::arg("circuit"));

    py::class_<QuantumCircuitSimulator>(m, "QuantumCircuitSimulator")
        .def(py::init<QuantumCircuit*, QuantumStateBase*>(), "Constructor",
            py::arg("circuit"), py::arg("state"))
        .def("initialize_state", &QuantumCircuitSimulator::initialize_state,
            "Initialize state")
        .def("initialize_random_state",
            py::overload_cast<>(
                &QuantumCircuitSimulator::initialize_random_state),
            "Initialize state with random pure state")
        .def("initialize_random_state",
            py::overload_cast<UINT>(
                &QuantumCircuitSimulator::initialize_random_state),
            "Initialize state with random pure state", py::arg("seed"))
        .def("simulate", &QuantumCircuitSimulator::simulate, "Simulate circuit")
        .def("simulate_range", &QuantumCircuitSimulator::simulate_range,
            "Simulate circuit", py::arg("start"), py::arg("end"))
        .def("get_expectation_value",
            &QuantumCircuitSimulator::get_expectation_value,
            "Get expectation value", py::arg("observable"))
        .def("get_gate_count", &QuantumCircuitSimulator::get_gate_count,
            "Get gate count")
        .def("copy_state_to_buffer",
            &QuantumCircuitSimulator::copy_state_to_buffer,
            "Copy state to buffer")
        .def("copy_state_from_buffer",
            &QuantumCircuitSimulator::copy_state_from_buffer,
            "Copy buffer to state")
        .def("swap_state_and_buffer",
            &QuantumCircuitSimulator::swap_state_and_buffer,
            "Swap state and buffer");

    py::class_<CausalConeSimulator>(m, "CausalConeSimulator")
        .def(py::init<ParametricQuantumCircuit&, Observable&>(), "Constructor")
        .def("build", &CausalConeSimulator::build, "Build")
        .def("get_expectation_value",
            &CausalConeSimulator::get_expectation_value,
            "Return expectation_value")
        .def("get_circuit_list", &CausalConeSimulator::get_circuit_list,
            "Return circuit_list")
        .def("get_pauli_operator_list",
            &CausalConeSimulator::get_pauli_operator_list,
            "Return pauli_operator_list")
        .def("get_coef_list", &CausalConeSimulator::get_coef_list,
            "Return coef_list");

    py::class_<NoiseSimulator::Result>(m, "SimulationResult")
        .def(
            "get_count",
            [](const NoiseSimulator::Result& result) -> UINT {
                return result.result.size();
            },
            "get state count")
        .def(
            "get_state",
            [](const NoiseSimulator::Result& result, UINT i) -> QuantumState* {
                return result.result[i].first->copy();
            },
            "get state", py::return_value_policy::take_ownership)
        .def(
            "get_frequency",
            [](const NoiseSimulator::Result& result, UINT i) -> UINT {
                return result.result[i].second;
            },
            "get state frequency");

    py::class_<NoiseSimulator>(m, "NoiseSimulator")
        .def(py::init<QuantumCircuit*, QuantumState*>(), "Constructor")
        .def("execute", &NoiseSimulator::execute,
            "Sampling & Return result [array]",
            py::return_value_policy::take_ownership)
        .def("execute_and_get_result", &NoiseSimulator::execute_and_get_result,
            "Simulate & Return ressult [array of (state, frequency)]");
}
