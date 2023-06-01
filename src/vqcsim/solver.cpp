#include "solver.hpp"

QuantumCircuitEnergyMinimizationSolver::QuantumCircuitEnergyMinimizationSolver(
    const std::function<ParametricQuantumCircuit*(UINT, UINT)>*
        circuit_generator,
    UINT param_count) {
    _circuit_construction = circuit_generator;
    _param_count = param_count;
    _circuit = NULL;
    verbose = false;
}

QuantumCircuitEnergyMinimizationSolver::
    ~QuantumCircuitEnergyMinimizationSolver() {
    if (_circuit != NULL) {
        delete _circuit;
    }
}

void QuantumCircuitEnergyMinimizationSolver::solve(
    EnergyMinimizationProblem* instance, UINT max_iteration,
    std::string optimizer_name,
    std::string differentiation_method) {
    if (_circuit != NULL) {
        delete _circuit;
        _circuit = NULL;
    }
    _circuit =
        (*_circuit_construction)(instance->get_qubit_count(), _param_count);
    auto _simulator = new ParametricQuantumCircuitSimulator(_circuit);
    _param_count = _simulator->get_parametric_gate_count();

    _parameter = std::vector<double>(_param_count, 0.);
    std::vector<double> gradient(_param_count);
    Random random;
    random.set_seed(0);
    for (auto& val : _parameter) val = random.uniform() * acos(0.0) * 4;

    GradientBasedOptimizer* optimizer;
    QuantumCircuitGradientDifferentiation* differentiation;

    if (optimizer_name == "Adam") {
        optimizer = new AdamOptimizer(_param_count);
    } else if (optimizer_name == "GD") {
        optimizer = new GradientDecentOptimizer(_param_count);
    } else
        return;
    if (differentiation_method == "HalfPi") {
        differentiation = new GradientByHalfPi();
    }
    std::vector<double> old_param;
    for (UINT iteration = 0; iteration < max_iteration; ++iteration) {
        loss = differentiation->compute_gradient(
            _simulator, instance, _parameter, &gradient);

        if (verbose) {
            std::cout << " *** epoch " << iteration << " *** " << std::endl;
            std::cout << " * loss = " << loss << std::endl;
            old_param = _parameter;
        }

        optimizer->apply_gradient(&_parameter, gradient);

        if (verbose) {
            for (UINT i = 0; i < _param_count; ++i) {
                std::cout << " ** id " << i << " para = " << old_param[i]
                          << " -> " << _parameter[i]
                          << " , grad = " << gradient[i] << std::endl;
            }
        }
    }
    delete optimizer;
    delete differentiation;
    delete _simulator;
}

double QuantumCircuitEnergyMinimizationSolver::get_loss() { return loss; }

std::vector<double> QuantumCircuitEnergyMinimizationSolver::get_parameter() {
    return _parameter;
}

ParametricQuantumCircuitSimulator*
QuantumCircuitEnergyMinimizationSolver::get_quantum_circuit_simulator() {
    return new ParametricQuantumCircuitSimulator(_circuit);
}

DiagonalizationEnergyMinimizationSolver::
    DiagonalizationEnergyMinimizationSolver() {
    verbose = false;
}

DiagonalizationEnergyMinimizationSolver::
    ~DiagonalizationEnergyMinimizationSolver() {}

void DiagonalizationEnergyMinimizationSolver::solve(
    EnergyMinimizationProblem* instance) {
    const UINT qubit_count = instance->get_qubit_count();
    const UINT term_count = instance->get_term_count();
    const ITYPE matrix_dim = 1ULL << qubit_count;

    ComplexMatrix observable_matrix =
        ComplexMatrix::Zero(matrix_dim, matrix_dim);
    for (UINT term_index = 0; term_index < term_count; ++term_index) {
        auto Pauli_operator = instance->get_Pauli_operator(term_index);
        CPPCTYPE coef = Pauli_operator->get_coef();
        auto target_index_list = Pauli_operator->get_index_list();
        auto pauli_id_list = Pauli_operator->get_pauli_id_list();

        std::vector<UINT> whole_pauli_id_list(qubit_count, 0);
        for (UINT i = 0; i < target_index_list.size(); ++i) {
            whole_pauli_id_list[target_index_list[i]] = pauli_id_list[i];
        }

        ComplexMatrix pauli_matrix;
        get_Pauli_matrix(pauli_matrix, whole_pauli_id_list);
        observable_matrix += coef * pauli_matrix;
    }

    observable_matrix.eigenvalues();
    Eigen::SelfAdjointEigenSolver<ComplexMatrix> eigen_solver(
        observable_matrix);
    loss = eigen_solver.eigenvalues()[0];

    if (verbose)
        std::cout << "Eigenvalues : " << std::endl
                  << eigen_solver.eigenvalues() << std::endl;
}

double DiagonalizationEnergyMinimizationSolver::get_loss() { return loss; }
