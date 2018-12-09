
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>

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
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/circuit_optimizer.hpp>
#include <cppsim/simulator.hpp>

#include <vqcsim/parametric_gate_factory.hpp>
#include <vqcsim/parametric_circuit.hpp>

namespace py = pybind11;
PYBIND11_MODULE(qulacs, m) {
    m.doc() = "cppsim python interface";

    py::class_<PauliOperator>(m, "PauliOperator")
        .def(py::init<double>())
        .def(py::init<std::string, double>())
        .def(py::init<std::vector<unsigned int>&, std::string, double>())
        .def(py::init<std::vector<unsigned int>&, double>())
        .def("get_index_list", &PauliOperator::get_index_list)
        .def("get_pauli_id_list", &PauliOperator::get_pauli_id_list)
        .def("get_coef", &PauliOperator::get_coef)
        .def("add_single_Pauli", &PauliOperator::add_single_Pauli)
        .def("get_expectation_value", &PauliOperator::get_expectation_value)
        .def("get_transition_amplitude", &PauliOperator::get_transition_amplitude)
        .def("copy", &PauliOperator::copy, pybind11::return_value_policy::automatic_reference)
        ;

    py::class_<Observable>(m, "Observable")
        .def(py::init<unsigned int>())
        // .def(py::init<std::string>())
        .def("add_operator", (void (Observable::*)(const PauliOperator*)) &Observable::add_operator)
        .def("add_operator", (void (Observable::*)(double coef, std::string))&Observable::add_operator)
        .def("get_qubit_count", &Observable::get_qubit_count)
        .def("get_state_dim", &Observable::get_state_dim)
        .def("get_term_count", &Observable::get_term_count)
        .def("get_term", &Observable::get_term, pybind11::return_value_policy::automatic_reference)
        .def("get_expectation_value", &Observable::get_expectation_value)
        .def("get_transition_amplitude", &Observable::get_transition_amplitude)
        //.def_static("get_split_Observable", &(Observable::get_split_observable));
        ;
    auto mobservable = m.def_submodule("observable");
    mobservable.def("create_observable_from_openfermion_file", &observable::create_observable_from_openfermion_file, pybind11::return_value_policy::automatic_reference);
    mobservable.def("create_observable_from_openfermion_text", &observable::create_observable_from_openfermion_text, pybind11::return_value_policy::automatic_reference);
    mobservable.def("create_split_observable", &observable::create_split_observable, pybind11::return_value_policy::automatic_reference);


    py::class_<QuantumStateBase>(m, "QuantumStateBase");
    py::class_<QuantumState,QuantumStateBase>(m, "QuantumState")
        .def(py::init<unsigned int>())
        .def("set_zero_state", &QuantumState::set_zero_state)
        .def("set_computational_basis", &QuantumState::set_computational_basis)
        .def("set_Haar_random_state", (void (QuantumState::*)(void))&QuantumState::set_Haar_random_state)
        .def("set_Haar_random_state", (void (QuantumState::*)(UINT))&QuantumState::set_Haar_random_state)
        .def("get_zero_probability", &QuantumState::get_zero_probability)
        .def("get_marginal_probability", &QuantumState::get_marginal_probability)
        .def("get_entropy", &QuantumState::get_entropy)
        .def("get_norm", &QuantumState::get_norm)
        .def("normalize", &QuantumState::normalize)
        .def("allocate_buffer", &QuantumState::allocate_buffer, pybind11::return_value_policy::automatic_reference)
        .def("copy", &QuantumState::copy, pybind11::return_value_policy::automatic_reference)
        .def("load", (void (QuantumState::*)(const QuantumStateBase*))&QuantumState::load)
        .def("load", (void (QuantumState::*)(const std::vector<CPPCTYPE>&))&QuantumState::load)
        .def("get_device_name", &QuantumState::get_device_name)
        .def("data_cpp", &QuantumState::data_cpp)
        .def("data_c", &QuantumState::data_c)
        .def("get_classical_value", &QuantumState::get_classical_value)
        .def("set_classical_value", &QuantumState::set_classical_value)
        .def("to_string",&QuantumState::to_string)
        .def("sampling",&QuantumState::sampling)

        .def("get_vector", [](const QuantumState& state) {
        Eigen::VectorXcd vec = Eigen::Map<Eigen::VectorXcd>(state.data_cpp(), state.dim);
        return vec;
        })
        .def("__repr__", [](const QuantumState &p) {return p.to_string();});
        ;
    auto mstate = m.def_submodule("state");
    mstate.def("inner_product", &state::inner_product);

    py::class_<QuantumGateBase>(m, "QuantumGateBase")
        .def("update_quantum_state", &QuantumGateBase::update_quantum_state)
        .def("copy",&QuantumGateBase::copy, pybind11::return_value_policy::automatic_reference)
        .def("to_string", &QuantumGateBase::to_string)

        .def("get_matrix", [](const QuantumGateBase& gate) {
        ComplexMatrix mat;
        gate.set_matrix(mat);
        return mat;
        })
        .def("__repr__", [](const QuantumGateBase &p) {return p.to_string(); });
        ;

    py::class_<QuantumGateMatrix,QuantumGateBase>(m, "QuantumGateMatrix")
        .def("update_quantum_state", &QuantumGateMatrix::update_quantum_state)
        .def("add_control_qubit", &QuantumGateMatrix::add_control_qubit)
        .def("multiply_scalar", &QuantumGateMatrix::multiply_scalar)
        .def("copy", &QuantumGateMatrix::copy, pybind11::return_value_policy::automatic_reference)
        .def("to_string", &QuantumGateMatrix::to_string)

        .def("get_matrix", [](const QuantumGateMatrix& gate) {
        ComplexMatrix mat;
        gate.set_matrix(mat);
        return mat;
        })
        .def("__repr__", [](const QuantumGateMatrix &p) {return p.to_string(); });
        ;

    auto mgate = m.def_submodule("gate");
    mgate.def("Identity", &gate::Identity, pybind11::return_value_policy::automatic_reference);
    mgate.def("X", &gate::X, pybind11::return_value_policy::automatic_reference);
    mgate.def("Y", &gate::Y, pybind11::return_value_policy::automatic_reference);
    mgate.def("Z", &gate::Z, pybind11::return_value_policy::automatic_reference);
    mgate.def("H", &gate::H, pybind11::return_value_policy::automatic_reference);
    mgate.def("S", &gate::S, pybind11::return_value_policy::automatic_reference);
    mgate.def("Sdag", &gate::Sdag, pybind11::return_value_policy::automatic_reference);
    mgate.def("T", &gate::T, pybind11::return_value_policy::automatic_reference);
    mgate.def("Tdag", &gate::Tdag, pybind11::return_value_policy::automatic_reference);
    mgate.def("sqrtX", &gate::sqrtX, pybind11::return_value_policy::automatic_reference);
    mgate.def("sqrtXdag", &gate::sqrtXdag, pybind11::return_value_policy::automatic_reference);
    mgate.def("sqrtY", &gate::sqrtY, pybind11::return_value_policy::automatic_reference);
    mgate.def("sqrtYdag", &gate::sqrtYdag, pybind11::return_value_policy::automatic_reference);
    mgate.def("P0", &gate::P0, pybind11::return_value_policy::automatic_reference);
    mgate.def("P1", &gate::P1, pybind11::return_value_policy::automatic_reference);

    mgate.def("U1", &gate::U1, pybind11::return_value_policy::automatic_reference);
    mgate.def("U2", &gate::U2, pybind11::return_value_policy::automatic_reference);
    mgate.def("U3", &gate::U3, pybind11::return_value_policy::automatic_reference);

    mgate.def("RX", &gate::RX, pybind11::return_value_policy::automatic_reference);
    mgate.def("RY", &gate::RY, pybind11::return_value_policy::automatic_reference);
    mgate.def("RZ", &gate::RZ, pybind11::return_value_policy::automatic_reference);

    mgate.def("CNOT", &gate::CNOT, pybind11::return_value_policy::automatic_reference);
    mgate.def("CZ", &gate::CZ, pybind11::return_value_policy::automatic_reference);
    mgate.def("SWAP", &gate::SWAP, pybind11::return_value_policy::automatic_reference);

    mgate.def("Pauli", &gate::Pauli, pybind11::return_value_policy::automatic_reference);
    mgate.def("PauliRotation", &gate::PauliRotation, pybind11::return_value_policy::automatic_reference);

    QuantumGateMatrix*(*ptr1)(unsigned int, ComplexMatrix) = &gate::DenseMatrix;
    QuantumGateMatrix*(*ptr2)(std::vector<unsigned int>, ComplexMatrix) = &gate::DenseMatrix;
    mgate.def("DenseMatrix", ptr1, pybind11::return_value_policy::automatic_reference);
    mgate.def("DenseMatrix", ptr2, pybind11::return_value_policy::automatic_reference);

    mgate.def("BitFlipNoise", &gate::BitFlipNoise);
    mgate.def("DephasingNoise", &gate::DephasingNoise);
    mgate.def("IndependentXZNoise", &gate::IndependentXZNoise);
    mgate.def("DepolarizingNoise", &gate::DepolarizingNoise);
    mgate.def("Measurement", &gate::Measurement);

    QuantumGateMatrix*(*ptr3)(const QuantumGateBase*, const QuantumGateBase*) = &gate::merge;
    mgate.def("merge", ptr3, pybind11::return_value_policy::automatic_reference);

    QuantumGateMatrix*(*ptr4)(std::vector<const QuantumGateBase*>) = &gate::merge;
    mgate.def("merge", ptr4, pybind11::return_value_policy::automatic_reference);

    QuantumGateMatrix*(*ptr5)(const QuantumGateBase*, const QuantumGateBase*) = &gate::add;
    mgate.def("add", ptr5, pybind11::return_value_policy::automatic_reference);

    QuantumGateMatrix*(*ptr6)(std::vector<const QuantumGateBase*>) = &gate::add;
    mgate.def("add", ptr6, pybind11::return_value_policy::automatic_reference);

    mgate.def("to_matrix_gate", &gate::to_matrix_gate, pybind11::return_value_policy::automatic_reference);
    mgate.def("Probabilistic", &gate::Probabilistic, pybind11::return_value_policy::automatic_reference);
    mgate.def("CPTP", &gate::CPTP, pybind11::return_value_policy::automatic_reference);
    mgate.def("Instrument", &gate::Instrument, pybind11::return_value_policy::automatic_reference);
    mgate.def("Adaptive", &gate::Adaptive, pybind11::return_value_policy::automatic_reference);


    mgate.def("ParametricRX", &gate::ParametricRX);
    mgate.def("ParametricRY", &gate::ParametricRY);
    mgate.def("ParametricRZ", &gate::ParametricRZ);
    mgate.def("ParametricPauliRotation", &gate::ParametricPauliRotation);


    py::class_<QuantumCircuit>(m, "QuantumCircuit")
        .def(py::init<unsigned int>())
        .def("copy", &QuantumCircuit::copy, pybind11::return_value_policy::automatic_reference)
        // In order to avoid double release, we force using add_gate_copy in python
        .def("add_gate_consume", (void (QuantumCircuit::*)(QuantumGateBase*))&QuantumCircuit::add_gate)
        .def("add_gate_consume", (void (QuantumCircuit::*)(QuantumGateBase*, unsigned int))&QuantumCircuit::add_gate)
        .def("add_gate", (void (QuantumCircuit::*)(const QuantumGateBase&))&QuantumCircuit::add_gate_copy)
        .def("add_gate", (void (QuantumCircuit::*)(const QuantumGateBase&, unsigned int))&QuantumCircuit::add_gate_copy)
        .def("remove_gate", &QuantumCircuit::remove_gate)

        .def("get_gate", [](const QuantumCircuit& circuit, unsigned int index) -> QuantumGateBase* { return circuit.gate_list[index]->copy(); }, pybind11::return_value_policy::automatic_reference)

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
        .def("add_diagonal_observable_rotation_gate", &QuantumCircuit::add_diagonal_observable_rotation_gate)
        .def("add_observable_rotation_gate", &QuantumCircuit::add_observable_rotation_gate)
        .def("__repr__", [](const QuantumCircuit &p) {return p.to_string(); });
    ;

    py::class_<ParametricQuantumCircuit, QuantumCircuit>(m, "ParametricQuantumCircuit")
        .def(py::init<unsigned int>())
        .def("add_parametric_gate", (void (ParametricQuantumCircuit::*)(QuantumGate_SingleParameter* gate))  &ParametricQuantumCircuit::add_parametric_gate)
        .def("add_parametric_gate", (void (ParametricQuantumCircuit::*)(QuantumGate_SingleParameter* gate, UINT))  &ParametricQuantumCircuit::add_parametric_gate)
        .def("add_gate", (void (ParametricQuantumCircuit::*)(QuantumGateBase* gate))  &ParametricQuantumCircuit::add_gate )
        .def("add_gate", (void (ParametricQuantumCircuit::*)(QuantumGateBase* gate, unsigned int))  &ParametricQuantumCircuit::add_gate)
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
        .def("merge_all", &QuantumCircuitOptimizer::merge_all, pybind11::return_value_policy::automatic_reference)
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



