
#pragma once

#include "gate.hpp"
#include "general_quantum_operator.hpp"
#include "observable.hpp"
#include "state.hpp"
#include "utility.hpp"

class ClsNoisyEvolution : public QuantumGateBase {
private:
    Random _random;
    Observable* _hamiltonian;
    GeneralQuantumOperator* _effective_hamiltonian;
    std::vector<GeneralQuantumOperator*> _c_ops;  // collapse operator
    std::vector<GeneralQuantumOperator*>
        _c_ops_dagger;        // collapse operator dagger
    double _time;             // evolution time
    double _dt;               // step size for runge kutta evolution
    double _norm_tol = 1e-8;  // accuracy in solving <psi|psi>=r
    int _find_collapse_max_steps =
        1000;  // maximum number of steps for while loop in _find_collapse

    /**
     * \~japanese-en collapse が起こるタイミング (norm = rになるタイミング)
     * を見つける。 この関数内で norm=rになるタイミングまでの evolution
     * が行われる。
     * normは時間に対して、広義単調減少であることが必要。
     * 内部では割線法を利用している。
     */
    virtual double _find_collapse(QuantumStateBase* k1, QuantumStateBase* k2,
        QuantumStateBase* k3, QuantumStateBase* k4,
        QuantumStateBase* prev_state, QuantumStateBase* now_state,
        double target_norm, double dt) {
        auto mae_norm = prev_state->get_squared_norm();
        auto now_norm = now_state->get_squared_norm();
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
            throw std::runtime_error(
                "must be prev_state.norm() >= target_norm. ");
        }
        if (now_norm > target_norm) {
            throw std::runtime_error(
                "must be now_state.norm() <= target_norm. ");
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
                t_guess = (t_mae + t_now) / 2;
            }

            // evolve by time t_guess
            buf_state->load(prev_state);
            _evolve_one_step(k1, k2, k3, k4, bufB_state, buf_state, t_guess);

            double buf_norm = buf_state->get_squared_norm();
            if (std::abs(buf_norm - target_norm) < _norm_tol) {
                now_state->load(buf_state);
                delete mae_state;
                delete buf_state;
                delete bufB_state;

                return t_guess;
            } else if (buf_norm < target_norm) {
                now_state->load(buf_state);
                t_now = t_guess;
                now_norm = now_state->get_squared_norm();
                now_norm_log = std::log(now_norm);
            } else {
                mae_state->load(buf_state);
                t_mae = t_guess;
                mae_norm = mae_state->get_squared_norm();
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
        //ここには来ない
        throw std::runtime_error(
            "unexpectedly come to end of _find_collapse function.");
    }

    /**
     * \~japanese-en runge-kutta を 1 step 進める
     * この関数が終わった時点で、時間発展前の状態は buffer に格納される。
     */
    virtual void _evolve_one_step(QuantumStateBase* k1, QuantumStateBase* k2,
        QuantumStateBase* k3, QuantumStateBase* k4, QuantumStateBase* buffer,
        QuantumStateBase* state, double dt) {
        // Runge-Kutta evolution
        // k1
        _effective_hamiltonian->apply_to_state(state, k1);

        // k2
        buffer->load(state);
        buffer->add_state_with_coef(-1.i * dt / 2., k1);
        _effective_hamiltonian->apply_to_state(buffer, k2);

        // k3
        buffer->load(state);
        buffer->add_state_with_coef(-1.i * dt / 2., k2);
        _effective_hamiltonian->apply_to_state(buffer, k3);

        // k4
        buffer->load(state);
        buffer->add_state_with_coef(-1.i * dt, k3);
        _effective_hamiltonian->apply_to_state(buffer, k4);

        // store the previous state in buffer
        buffer->load(state);

        // add them together
        state->add_state_with_coef(-1.i * dt / 6., k1);
        state->add_state_with_coef(-1.i * dt / 3., k2);
        state->add_state_with_coef(-1.i * dt / 3., k3);
        state->add_state_with_coef(-1.i * dt / 6., k4);
    }

public:
    ClsNoisyEvolution(Observable* hamiltonian,
        std::vector<GeneralQuantumOperator*> c_ops, double time,
        double dt = 1e-6) {
        _hamiltonian = dynamic_cast<Observable*>(hamiltonian->copy());
        for (auto const& op : c_ops) {
            _c_ops.push_back(op->copy());
            _c_ops_dagger.push_back(op->get_dagger());
        }
        _effective_hamiltonian = hamiltonian->copy();
        for (size_t k = 0; k < _c_ops.size(); k++) {
            auto cdagc = (*_c_ops_dagger[k]) * (*_c_ops[k]) * (-.5i);
            *_effective_hamiltonian += cdagc;
        }
        _time = time;
        _dt = dt;
    };
    ~ClsNoisyEvolution() {
        delete _hamiltonian;
        delete _effective_hamiltonian;
        for (size_t k = 0; k < _c_ops.size(); k++) {
            delete _c_ops[k];
            delete _c_ops_dagger[k];
        }
    };

    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        std::stringstream error_message_stream;
        error_message_stream
            << "* Warning : Gate-matrix of noisy evolution cannot be "
               "defined. Nothing has been done.";
        throw std::invalid_argument(error_message_stream.str());
    }

    /**
     * \~japanese-en 乱数シードをセットする
     *
     * @param seed シード値
     */
    virtual void set_seed(int seed) override { _random.set_seed(seed); };

    virtual QuantumGateBase* copy() const override {
        return new ClsNoisyEvolution(_hamiltonian, _c_ops, _time, _dt);
    }

    /**
     * \~japanese-en NoisyEvolution が使用する有効ハミルトニアンを得る
     */
    virtual GeneralQuantumOperator* get_effective_hamiltonian() const {
        return _effective_hamiltonian->copy();
    }

    /**
     * \~japanese-en collapse 時間を探すときに許す最大ループ数をセットする
     *
     * @param n ステップ数
     */
    virtual void set_find_collapse_max_steps(int n) {
        this->_find_collapse_max_steps = n;
    }

    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) {
        double r = _random.uniform();
        std::vector<double> cumulative_dist(_c_ops.size());
        double prob_sum = 0;
        auto k1 = state->copy();  // vector for Runge-Kutta k
        auto k2 = state->copy();  // vector for Runge-Kutta k
        auto k3 = state->copy();  // vector for Runge-Kutta k
        auto k4 = state->copy();  // vector for Runge-Kutta k
        auto buffer = state->copy();
        double t = 0;

        while (std::abs(t - _time) >
               1e-10 * _time) {  // For machine precision error.
            // For final time, we modify the step size to match the total
            // execution time
            auto dt = _dt;
            if (t + _dt > _time) {
                dt = _time - t;
            }
            while (std::abs(dt) >
                   1e-10 * _dt) {  // dt is decreased by dt_target_norm if jump
                                   // occurs and if jump did not occure dt will
                                   // be set to zero.
                // evolve the state by dt
                _evolve_one_step(k1, k2, k3, k4, buffer, state, dt);
                // check if the jump should occur or not
                auto norm = state->get_squared_norm();
                if (norm <= r) {  // jump occured
                    // evolve the state to the time such that norm=r
                    double dt_target_norm;
                    try {
                        dt_target_norm = _find_collapse(
                            k1, k2, k3, k4, buffer, state, r, dt);
                    } catch (std::runtime_error& e) {
                        throw std::runtime_error(
                            "_find_collapse failed. Result is "
                            "unreliable.");
                        return;
                    }

                    // get cumulative distribution
                    prob_sum = 0.;
                    for (size_t k = 0; k < _c_ops.size(); k++) {
                        _c_ops[k]->apply_to_state(state, buffer);
                        cumulative_dist[k] =
                            buffer->get_squared_norm() + prob_sum;
                        prob_sum = cumulative_dist[k];
                    }

                    // determine which collapse operator to be applied
                    auto jump_r = _random.uniform() * prob_sum;
                    auto ite = std::lower_bound(
                        cumulative_dist.begin(), cumulative_dist.end(), jump_r);
                    auto index = std::distance(cumulative_dist.begin(), ite);

                    // apply the collapse operator and normalize the state
                    _c_ops[index]->apply_to_state(state, buffer);
                    buffer->normalize(buffer->get_squared_norm());
                    state->load(buffer);

                    // update dt to be consistent with the step size
                    t += dt_target_norm;
                    dt -= dt_target_norm;

                    // update random variable
                    r = _random.uniform();
                } else {  // if jump did not occur, update t to the next time
                          // and break the loop
                    t += dt;
                    dt = 0.;
                }
            }
        }

        // normalize the state and finish
        state->normalize(state->get_squared_norm());
        delete k1;
        delete k2;
        delete k3;
        delete k4;
        delete buffer;
    }
};
