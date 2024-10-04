#include "gate_noisy_evolution.hpp"

#include <Eigen/Dense>
#include <csim/stat_ops.hpp>
#include <csim/update_ops.hpp>
#include <csim/update_ops_dm.hpp>
#include <cstring>
#include <fstream>
#include <numeric>

#include "exception.hpp"
#include "gate.hpp"
#include "gate_factory.hpp"
#include "gate_merge.hpp"
#include "gate_named_pauli.hpp"
#include "general_quantum_operator.hpp"
#include "observable.hpp"
#include "pauli_operator.hpp"
#include "state.hpp"
#include "type.hpp"
#include "utility.hpp"

double ClsNoisyEvolution::_find_collapse(QuantumStateBase* k1,
    QuantumStateBase* k2, QuantumStateBase* k3, QuantumStateBase* k4,
    QuantumStateBase* prev_state, QuantumStateBase* now_state,
    double target_norm, double dt, bool use_secant_method) {
    if (!use_secant_method) {
        return _find_collapse_original(
            k1, k2, k3, k4, prev_state, now_state, target_norm, dt);
    }
    auto mae_norm = prev_state->get_squared_norm_single_thread();
    auto now_norm = now_state->get_squared_norm_single_thread();
    double t_mae = 0;
    double t_now = dt;

    // mae now で挟み撃ちする
    int search_count = 0;

    if (std::abs(mae_norm - target_norm) < _norm_tol) {
        now_state->load(prev_state);
        return 0;
    }
    if (std::abs(now_norm - target_norm) < _norm_tol) {
        return dt;
    }
    if (mae_norm < target_norm) {
        throw std::runtime_error("must be prev_state.norm() >= target_norm. ");
    }
    if (now_norm > target_norm) {
        throw std::runtime_error("must be now_state.norm() <= target_norm. ");
    }

    QuantumStateBase* mae_state = prev_state->copy();
    double target_norm_log = std::log(target_norm);
    double mae_norm_log = std::log(mae_norm);
    double now_norm_log = std::log(now_norm);
    QuantumStateBase* buf_state = prev_state->copy();
    QuantumStateBase* bufB_state = prev_state->copy();
    while (true) {
        //  we expect norm to reduce as Ae^-a*dt, so use log.

        double t_guess = 0;
        if (search_count <= 20) {
            // use secant method
            t_guess = t_mae + (t_now - t_mae) *
                                  (mae_norm_log - target_norm_log) /
                                  (mae_norm_log - now_norm_log);
        } else {
            // use bisection method
            t_guess = (t_mae + t_now) / 2;
        }

        // evolve by time t_guess
        buf_state->load(prev_state);
        _evolve_one_step(k1, k2, k3, k4, bufB_state, buf_state, t_guess);

        double buf_norm = buf_state->get_squared_norm_single_thread();
        if (std::abs(buf_norm - target_norm) < _norm_tol) {
            now_state->load(buf_state);
            delete mae_state;
            delete buf_state;
            delete bufB_state;

            return t_guess;
        } else if (buf_norm < target_norm) {
            now_state->load(buf_state);
            t_now = t_guess;
            now_norm = now_state->get_squared_norm_single_thread();
            now_norm_log = std::log(now_norm);
        } else {
            mae_state->load(buf_state);
            t_mae = t_guess;
            mae_norm = mae_state->get_squared_norm_single_thread();
            mae_norm_log = std::log(mae_norm);
        }

        search_count++;
        // avoid infinite loop
        // It sometimes fails to find t_guess to reach the target norm.
        // More likely to happen when dt is not small enough compared to the
        // relaxation times
        if (search_count >= _find_collapse_max_steps) {
            throw std::runtime_error(
                "Failed to find the exact jump time. Try with "
                "smaller dt.");
        }
    }
    // ここには来ない
    throw std::runtime_error(
        "unexpectedly come to end of _find_collapse function.");
}

/**
 * \~japanese-en collapse が起こるタイミング (norm = rになるタイミング)
 * を見つける。 この関数内で norm=rになるタイミングまでの evolution
 * が行われる。
 *
 * 割線法を利用する場合、normは時間に対して広義単調減少であることが必要。
 */
double ClsNoisyEvolution::_find_collapse_original(QuantumStateBase* k1,
    QuantumStateBase* k2, QuantumStateBase* k3, QuantumStateBase* k4,
    QuantumStateBase* prev_state, QuantumStateBase* now_state,
    double target_norm, double dt) {
    auto now_norm = now_state->get_squared_norm_single_thread();
    auto prev_norm = prev_state->get_squared_norm_single_thread();
    auto t_guess = dt;
    int search_count = 0;
    while (std::abs(now_norm - target_norm) > _norm_tol) {
        // we expect norm to reduce as Ae^-a*dt, so we first calculate the
        // coefficient a
        auto a = std::log(now_norm / prev_norm) / t_guess;
        // then guess the time to become target norm as (target_norm) =
        // (prev_norm)e^-a*(t_guess) which means -a*t_guess =
        // log(target_norm/prev_norm)
        t_guess = std::log(target_norm / prev_norm) / a;
        // evolve by time t_guess
        now_state->load(prev_state);
        _evolve_one_step(k1, k2, k3, k4, prev_state, now_state, t_guess);
        now_norm = now_state->get_squared_norm_single_thread();

        search_count++;
        // avoid infinite loop
        // It sometimes fails to find t_guess to reach the target norm.
        // More likely to happen when dt is not small enough compared to the
        // relaxation times
        if (search_count >= _find_collapse_max_steps) {
            throw std::runtime_error(
                "Failed to find the exact jump time. Try with "
                "smaller dt.");
        }
    }
    return t_guess;
}

/**
 * \~japanese-en runge-kutta を 1 step 進める
 * この関数が終わった時点で、時間発展前の状態は buffer に格納される。
 */
void ClsNoisyEvolution::_evolve_one_step(QuantumStateBase* k1,
    QuantumStateBase* k2, QuantumStateBase* k3, QuantumStateBase* k4,
    QuantumStateBase* buffer, QuantumStateBase* state, double dt) {
    // Runge-Kutta evolution
    // k1
    _effective_hamiltonian->apply_to_state_single_thread(state, k1);

    // k2
    buffer->load(state);
    buffer->add_state_with_coef_single_thread(-1.i * dt / 2., k1);
    _effective_hamiltonian->apply_to_state_single_thread(buffer, k2);

    // k3
    buffer->load(state);
    buffer->add_state_with_coef_single_thread(-1.i * dt / 2., k2);
    _effective_hamiltonian->apply_to_state_single_thread(buffer, k3);

    // k4
    buffer->load(state);
    buffer->add_state_with_coef_single_thread(-1.i * dt, k3);
    _effective_hamiltonian->apply_to_state_single_thread(buffer, k4);

    // store the previous state in buffer
    buffer->load(state);

    // add them together
    state->add_state_with_coef_single_thread(-1.i * dt / 6., k1);
    state->add_state_with_coef_single_thread(-1.i * dt / 3., k2);
    state->add_state_with_coef_single_thread(-1.i * dt / 3., k3);
    state->add_state_with_coef_single_thread(-1.i * dt / 6., k4);
}

ClsNoisyEvolution::ClsNoisyEvolution(Observable* hamiltonian,
    std::vector<GeneralQuantumOperator*> c_ops, double time, double dt) {
    _hamiltonian = hamiltonian->copy();
    for (auto const& op : c_ops) {
        _c_ops.push_back(op->copy());
        _c_ops_dagger.push_back(op->get_dagger());
    }

    // HermitianQuantumOperatorは、add_operatorの呼び出し時に、
    // 追加するPauliOperatorがHermitianであるかのチェックが入る。
    // _effective_hamiltonianに追加するPauliOperatorはHermitianとは限らないので、
    // チェックに失敗してしまう可能性がある。
    // したがって、このチェックを回避するためにGeneralQuantumOperatorを生成し、
    // hamiltonianの中身をコピーする。
    //
    // HermitianQuantumOperatorをGeneralQuantumOperatorにキャストする方法では、
    // インスタンスがHermitianQuantumOperatorのままであるため、
    // HermitianQuantumOperator側のadd_operatorが呼び出されてしまい、問題が解決できない。
    // したがって、GeneralQuantumOperatorを実際に作成する必要がある。
    _effective_hamiltonian =
        new GeneralQuantumOperator(hamiltonian->get_qubit_count());
    for (auto pauli : hamiltonian->get_terms()) {
        _effective_hamiltonian->add_operator_copy(pauli);
    }

    for (size_t k = 0; k < _c_ops.size(); k++) {
        auto cdagc = (*_c_ops_dagger[k]) * (*_c_ops[k]) * (-.5i);
        *_effective_hamiltonian += cdagc;
    }
    _time = time;
    _dt = dt;
}
ClsNoisyEvolution::~ClsNoisyEvolution() {
    delete _hamiltonian;
    delete _effective_hamiltonian;
    for (size_t k = 0; k < _c_ops.size(); k++) {
        delete _c_ops[k];
        delete _c_ops_dagger[k];
    }
}

/**
 * \~japanese-en 量子状態を更新する
 *
 * @param state 更新する量子状態
 */
void ClsNoisyEvolution::update_quantum_state(QuantumStateBase* state) {
    double initial_squared_norm = state->get_squared_norm();
    double r = _random.uniform();
    std::vector<double> cumulative_dist(_c_ops.size());
    double prob_sum = 0;
    auto k1 = state->copy();  // vector for Runge-Kutta k
    auto k2 = state->copy();  // vector for Runge-Kutta k
    auto k3 = state->copy();  // vector for Runge-Kutta k
    auto k4 = state->copy();  // vector for Runge-Kutta k
    auto buffer = state->copy();
    double t = 0;

    while (
        std::abs(t - _time) > 1e-10 * _time) {  // For machine precision error.
        // For final time, we modify the step size to match the total
        // execution time
        auto dt = _dt;
        if (t + _dt > _time) {
            dt = _time - t;
        }
        _evolve_one_step(k1, k2, k3, k4, buffer, state, dt);
        // check if the jump should occur or not
        auto norm = state->get_squared_norm();
        if (norm <= r) {  // jump occured
            // evolve the state to the time such that norm=r
            double dt_target_norm;
            dt_target_norm =
                _find_collapse(k1, k2, k3, k4, buffer, state, r, dt);

            // get cumulative distribution
            prob_sum = 0.;
            for (size_t k = 0; k < _c_ops.size(); k++) {
                _c_ops[k]->apply_to_state_single_thread(state, buffer);
                cumulative_dist[k] =
                    buffer->get_squared_norm_single_thread() + prob_sum;
                prob_sum = cumulative_dist[k];
            }

            // determine which collapse operator to be applied
            const auto jump_r = _random.uniform() * prob_sum;
            const auto ite = std::lower_bound(
                cumulative_dist.begin(), cumulative_dist.end(), jump_r);
            const auto index = static_cast<size_t>(
                std::distance(cumulative_dist.begin(), ite));

            // apply the collapse operator and normalize the state
            // ルンゲクッタ法の誤差により、normが1にならない場合があります。
            // その場合に本来collapseが起こらないはずなのに起きる分岐に入ってしまいます。
            // そして、c_opsが空の場合に不正アクセスとなります。
            // coefが0の場合でも、各collapseが発生する確率は0なので、0除算が発生してしまいます。
            if (_c_ops.size() > index) {
                _c_ops[index]->apply_to_state_single_thread(state, buffer);
                buffer->normalize(buffer->get_squared_norm_single_thread());
                state->load(buffer);
            }

            // update dt to be consistent with the step size
            t += dt_target_norm;

            // update random variable
            r = _random.uniform();
        } else {  // if jump did not occur, update t to the next time
            t += dt;
        }
    }

    // normalize the state and finish
    state->normalize_single_thread(
        state->get_squared_norm_single_thread() / initial_squared_norm);
    delete k1;
    delete k2;
    delete k3;
    delete k4;
    delete buffer;
}

boost::property_tree::ptree ClsNoisyEvolution::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "NoisyEvolutionGate");
    pt.put_child("hamiltonian", _hamiltonian->to_ptree());
    boost::property_tree::ptree c_ops_pt;
    for (const GeneralQuantumOperator* c_op : _c_ops) {
        c_ops_pt.push_back(std::make_pair("", c_op->to_ptree()));
    }
    pt.put_child("c_ops", c_ops_pt);
    pt.put("time", _time);
    pt.put("dt", _dt);
    return pt;
}

// ここからfast

double ClsNoisyEvolution_fast::_find_collapse(QuantumStateBase* prev_state,
    QuantumStateBase* now_state, double target_norm, double t_step) {
    auto mae_norm = prev_state->get_squared_norm_single_thread();
    auto now_norm = now_state->get_squared_norm_single_thread();
    double t_mae = 0;
    double t_now = t_step;

    // mae now で挟み撃ちする
    int search_count = 0;

    if (std::abs(mae_norm - target_norm) < _norm_tol) {
        now_state->load(prev_state);
        return 0;
    }
    if (std::abs(now_norm - target_norm) < _norm_tol) {
        return t_step;
    }
    if (mae_norm < target_norm) {
        throw std::runtime_error("must be prev_state.norm() >= target_norm. ");
    }
    if (now_norm > target_norm) {
        throw std::runtime_error("must be now_state.norm() <= target_norm. ");
    }

    QuantumStateBase* mae_state = prev_state->copy();
    double target_norm_log = std::log(target_norm);
    double mae_norm_log = std::log(mae_norm);
    double now_norm_log = std::log(now_norm);
    QuantumStateBase* buf_state = prev_state->copy();
    while (true) {
        double t_guess = 0;
        if (search_count <= 20) {
            // use secant method
            double mae_kyo = (mae_norm_log - target_norm_log) /
                             (mae_norm_log - now_norm_log);
            double ato_kyo = (target_norm_log - now_norm_log) /
                             (mae_norm_log - now_norm_log);

            if ((search_count - 2) % 3 != 2) {  // 5,8,11,14,17は下の式を採用
                t_guess =
                    t_mae + (t_now - t_mae) * mae_kyo / (mae_kyo + ato_kyo);
            } else {
                t_guess = t_mae + (t_now - t_mae) * sqrt(mae_kyo) /
                                      (sqrt(mae_kyo) + sqrt(ato_kyo));
            }

        } else {
            // use bisection method
            t_guess = (t_mae + t_now) / 2;
        }

        // evolve by time t_guess
        buf_state->load(prev_state);
        _evolve_one_step(buf_state, t_guess);

        double buf_norm = buf_state->get_squared_norm_single_thread();
        if (std::abs(buf_norm - target_norm) < _norm_tol) {
            now_state->load(buf_state);
            delete mae_state;
            delete buf_state;

            return t_guess;
        } else if (buf_norm < target_norm) {
            now_state->load(buf_state);
            t_now = t_guess;
            now_norm = now_state->get_squared_norm_single_thread();
            now_norm_log = std::log(now_norm);
        } else {
            mae_state->load(buf_state);
            t_mae = t_guess;
            mae_norm = mae_state->get_squared_norm_single_thread();
            mae_norm_log = std::log(mae_norm);
        }

        search_count++;
        // avoid infinite loop
        // It sometimes fails to find t_guess to reach the target norm.
        // More likely to happen when t_step is not small enough compared to
        // the relaxation times
        if (search_count >= _find_collapse_max_steps) {
            throw std::runtime_error(
                "Failed to find the exact jump time. Try with "
                "smaller t_step.");
        }
    }
    // ここには来ない
    throw std::runtime_error(
        "unexpectedly come to end of _find_collapse function.");
}

void ClsNoisyEvolution_fast::_evolve_one_step(
    QuantumStateBase* state, double t_step) {
    // 対角化したものを持っておき、　ここではそれを使う
    eigenMatrixRevGate->update_quantum_state(state);
    ITYPE dim = eigenvalue_mto.size();

    ComplexVector eigenvalues(dim);

    for (ITYPE i = 0; i < dim; i++) {
        eigenvalues[i] = std::exp(eigenvalue_mto[i] * t_step);
    }
    // eigenvalues[i] を掛ける必要がある
    QuantumGateBase* eigenValueGate =
        gate::DiagonalMatrix(this->get_target_index_list(), eigenvalues);

    eigenValueGate->update_quantum_state(state);

    eigenMatrixGate->update_quantum_state(state);

    delete eigenValueGate;
}

ClsNoisyEvolution_fast::ClsNoisyEvolution_fast(Observable* hamiltonian,
    std::vector<GeneralQuantumOperator*> c_ops, double time) {
    _hamiltonian = hamiltonian->copy();

    for (auto const& op : c_ops) {
        _c_ops.push_back(op->copy());
        _c_ops_dagger.push_back(op->get_dagger());
        auto aaa = op->get_terms();
    }

    // HermitianQuantumOperatorは、add_operatorの呼び出し時に、
    // 追加するPauliOperatorがHermitianであるかのチェックが入る。
    // _effective_hamiltonianに追加するPauliOperatorはHermitianとは限らないので、
    // チェックに失敗してしまう可能性がある。
    // したがって、このチェックを回避するためにGeneralQuantumOperatorを生成し、
    // hamiltonianの中身をコピーする。
    //
    // HermitianQuantumOperatorをGeneralQuantumOperatorにキャストする方法では、
    // インスタンスがHermitianQuantumOperatorのままであるため、
    // HermitianQuantumOperator側のadd_operatorが呼び出されてしまい、問題が解決できない。
    // したがって、GeneralQuantumOperatorを実際に作成する必要がある。
    _effective_hamiltonian =
        new GeneralQuantumOperator(hamiltonian->get_qubit_count());

    for (auto pauli : hamiltonian->get_terms()) {
        _effective_hamiltonian->add_operator_copy(pauli);
    }
    for (size_t k = 0; k < _c_ops.size(); k++) {
        auto cdagc = (*_c_ops_dagger[k]) * (*_c_ops[k]) * (-.5i);
        *_effective_hamiltonian += cdagc;
    }
    _time = time;

    std::vector<QuantumGateBase*> gate_list;
    std::vector<bool> bit_list(hamiltonian->get_qubit_count());
    for (auto pauli : _effective_hamiltonian->get_terms()) {
        // 要素をゲートのマージで作る
        auto pauli_gate =
            gate::Pauli(pauli->get_index_list(), pauli->get_pauli_id_list());

        auto aaaa = gate::to_matrix_gate(pauli_gate);

        aaaa->multiply_scalar(-pauli->get_coef() * 1.0i);

        gate_list.push_back(aaaa);
        for (auto it : pauli->get_index_list()) {
            bit_list[it] = true;
        }
        delete pauli_gate;
    }
    int iH_bit_num = 0;
    for (UINT i = 0; i < hamiltonian->get_qubit_count(); i++) {
        if (bit_list[i]) {
            iH_bit_num++;
        }
    }
    if (iH_bit_num > 8) {
        throw std::runtime_error(
            "too much use hamiltonian qubit. "
            " If there is an independent bit, please separate gate");
    }
    if (gate_list.size() == 0) {
        gate_none = true;
        return;
    }
    // effective_hamiltonianをもとに、 -iHを求める
    //(iHという名前だが、実際には-iH)

    QuantumGateMatrix* iH = gate::add(gate_list);
    auto target_index_list = iH->get_target_index_list();
    _target_qubit_list.clear();
    _target_qubit_list.reserve(target_index_list.size());
    for (UINT idx : target_index_list) {
        _target_qubit_list.emplace_back(idx);
    }
    // 注意　このtarget_index_listはPや対角行列　に対してのlistである。
    // このlistに入っていないが、c_opsには入っている　というパターンもありうる。
    // ここで、　対角化を求める
    //  A = PBP^-1 のとき、　e^A = P e^B P^-1
    //  の性質を使って,e^-iHを計算する

    ComplexMatrix hamilMatrix;
    iH->set_matrix(hamilMatrix);
    delete iH;
    Eigen::ComplexEigenSolver<ComplexMatrix> eigen_solver(hamilMatrix);

    auto result = eigen_solver.info();

    if (result != Eigen::Success) {
        throw std::runtime_error("sorry, This complex don't have eigenvalue.");
    } else {
        const auto eigenvectors = eigen_solver.eigenvectors();
        eigenvalue_mto = eigen_solver.eigenvalues();

        eigenMatrixGate =
            gate::DenseMatrix(this->get_target_index_list(), eigenvectors);
        eigenMatrixRevGate = gate::DenseMatrix(
            this->get_target_index_list(), eigenvectors.inverse());
    }

    for (auto it : gate_list) {
        delete it;
    }
}

void ClsNoisyEvolution_fast::update_quantum_state(QuantumStateBase* state) {
    if (gate_none) {
        return;
    }
    double initial_squared_norm = state->get_squared_norm();
    double r = _random.uniform();
    std::vector<double> cumulative_dist(_c_ops.size());
    double prob_sum = 0;
    auto buffer = state->copy();
    double t = 0;

    while (
        std::abs(t - _time) > 1e-10 * _time) {  // For machine precision error.
        // For final time, we modify the step size to match the total
        // execution time
        auto t_step = _time - t;

        _evolve_one_step(state, t_step);
        // check if the jump should occur or not
        auto norm = state->get_squared_norm();
        if (norm <= r) {  // jump occured
            // evolve the state to the time such that norm=r
            double at_target_norm;
            at_target_norm = _find_collapse(buffer, state, r, t_step);

            // get cumulative distribution
            prob_sum = 0.;
            for (size_t k = 0; k < _c_ops.size(); k++) {
                _c_ops[k]->apply_to_state_single_thread(state, buffer);
                cumulative_dist[k] =
                    buffer->get_squared_norm_single_thread() + prob_sum;
                prob_sum = cumulative_dist[k];
            }

            // determine which collapse operator to be applied
            auto jump_r = _random.uniform() * prob_sum;
            auto ite = std::lower_bound(
                cumulative_dist.begin(), cumulative_dist.end(), jump_r);
            auto index = std::distance(cumulative_dist.begin(), ite);

            // apply the collapse operator and normalize the state
            _c_ops[index]->apply_to_state_single_thread(state, buffer);
            buffer->normalize(buffer->get_squared_norm_single_thread());
            state->load(buffer);

            // update t_step to be consistent with the step size
            t += at_target_norm;

            // update random variable
            r = _random.uniform();
        } else {  // if jump did not occur, update t to the next time
            t += t_step;
        }
    }

    // normalize the state and finish
    state->normalize_single_thread(
        state->get_squared_norm_single_thread() / initial_squared_norm);

    delete buffer;
}

boost::property_tree::ptree ClsNoisyEvolution_fast::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "NoisyEvolutionFastGate");
    pt.put_child("hamiltonian", _hamiltonian->to_ptree());
    boost::property_tree::ptree c_ops_pt;
    for (const GeneralQuantumOperator* c_op : _c_ops) {
        c_ops_pt.push_back(std::make_pair("", c_op->to_ptree()));
    }
    pt.put_child("c_ops", c_ops_pt);
    pt.put("time", _time);
    return pt;
}
