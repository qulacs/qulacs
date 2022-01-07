
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
    std::vector<GeneralQuantumOperator*> _c_ops;
    std::vector<GeneralQuantumOperator*> _c_ops_dagger;
    double _time;
    double _dt;
    double _norm_tol = 1e-8;

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
        std::cerr << "* Warning : Gate-matrix of noisy evolution cannot be "
                     "obtained. Identity matrix is returned."
                  << std::endl;
        matrix = Eigen::MatrixXcd::Ones(1, 1);
    }
    virtual void set_seed(int seed) override { _random.set_seed(seed); };

    virtual QuantumGateBase* copy() const override {
        return new ClsNoisyEvolution(_hamiltonian, _c_ops, _time, _dt);
    }

    virtual GeneralQuantumOperator* get_effective_hamiltonian() const {
        return _effective_hamiltonian->copy();
    }

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
                    auto dt_target_norm =
                        _find_collapse(k1, k2, k3, k4, buffer, state, r, dt);

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

    /**
     * \~japanese-en collapse が起こるタイミング (norm = rになるタイミング)
     * を見つける。 この関数内で norm=rになるタイミングまでの evolution
     * が行われる。
     */
    virtual double _find_collapse(QuantumStateBase* k1, QuantumStateBase* k2,
        QuantumStateBase* k3, QuantumStateBase* k4,
        QuantumStateBase* prev_state, QuantumStateBase* now_state,
        double target_norm, double dt) {
        auto now_norm = now_state->get_squared_norm();
        auto prev_norm = prev_state->get_squared_norm();
        auto t_guess = dt;
        int search_count = 0;
        // std::cout << "prev norm" << prev_state->get_squared_norm() <<
        // std::endl; std::cout << "now state: " << now_norm << " prev state: "
        // << prev_norm << target_norm << std::endl;
        while (std::abs(now_norm - target_norm) > _norm_tol) {
            // we expect norm to reduce as Ae^-a*dt, so we first calculate the
            // coefficient a
            auto a = std::log(now_norm / prev_norm) / t_guess;
            // then guess the time to become target norm as (target_norm) =
            // (prev_norm)e^-a*(t_guess) which means -a*t_guess =
            // log(target_norm/prev_norm)
            t_guess = std::log(target_norm / prev_norm) / a;
            // evole by time t_guess
            now_state->load(prev_state);
            _evolve_one_step(k1, k2, k3, k4, prev_state, now_state, t_guess);
            now_norm = now_state->get_squared_norm();
            // std::cout << "prev norm" << prev_state->get_squared_norm() <<
            // std::endl;
            search_count++;
            // std::cout << "search count" << search_count << std::endl;
        }
        return t_guess;
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
};
