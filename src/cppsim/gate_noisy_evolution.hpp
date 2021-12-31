
#pragma once

#include "gate.hpp"
#include "state.hpp"
#include "utility.hpp"
#include "observable.hpp"
#include "general_quantum_operator.hpp"


class ClsNoisyEvolution : public QuantumGateBase
{
private:
    Random _random;
    Observable* _hamiltonian;
    GeneralQuantumOperator* _effective_hamiltonian;
    std::vector<GeneralQuantumOperator*> _c_ops;
    std::vector<GeneralQuantumOperator*> _c_ops_dagger;
    double _time;
    double _dt;
    

public:
    ClsNoisyEvolution(Observable* hamiltonian, 
                   std::vector<GeneralQuantumOperator*> c_ops,
                   double time,
                   double dt=1e-6){
        _hamiltonian = dynamic_cast<Observable*>(hamiltonian->copy());
        for (auto const & op: c_ops){
            _c_ops.push_back(op->copy());
            _c_ops_dagger.push_back(op->get_dagger());
        } 
        _effective_hamiltonian = hamiltonian->copy();
        for (size_t k=0; k<_c_ops.size(); k++){
            auto cdagc = (*_c_ops_dagger[k])*(*_c_ops[k])*(-0.5i);
            for (UINT j=0; j<cdagc.get_term_count(); j++){
                _effective_hamiltonian->add_operator(cdagc.get_term(j));
            }
        }
        _time = time;
        _dt = dt;
    };
    ~ClsNoisyEvolution(){
        delete _hamiltonian;
        delete _effective_hamiltonian;
        for (size_t k=0; k<_c_ops.size(); k++){
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

    virtual void update_quantum_state(QuantumStateBase* state) {
        double r = _random.uniform();
        std::vector<double> cumulative_dist(_c_ops.size());
        double prob_sum = 0;
        auto k1 = state->copy(); //vector for Runge-Kutta k
        auto k2 = state->copy(); //vector for Runge-Kutta k
        auto k3 = state->copy(); //vector for Runge-Kutta k
        auto k4 = state->copy(); //vector for Runge-Kutta k
        auto buffer = state->copy();
        auto dt = _dt;
        for (double t=0; t <= _time; t += _dt){
            // for final time, we modify the step size to match the total execution time
            if (t + _dt > _time){
                dt = _time-t;
            }
            // Runge-Kutta evolution
            // k1
            _effective_hamiltonian -> apply_to_state(state, k1);

            //k2
            buffer -> load(state);
            buffer -> add_state_with_coef(-1.i*dt/2., k1);
            _effective_hamiltonian -> apply_to_state(buffer, k2);
            
            // k3
            buffer -> load(state);
            buffer -> add_state_with_coef(-1.i*dt/2., k2);
            _effective_hamiltonian -> apply_to_state(buffer, k3);
            
            // k4
            buffer -> load(state);
            buffer -> add_state_with_coef(-1.i*dt, k3);
            _effective_hamiltonian -> apply_to_state(buffer, k4);

            // add them together
            state->add_state_with_coef(-1.i*dt/6., k1);
            state->add_state_with_coef(-1.i*dt/3., k2);
            state->add_state_with_coef(-1.i*dt/3., k3);
            state->add_state_with_coef(-1.i*dt/6., k4);

            auto norm = state -> get_squared_norm();
            if (norm <= r){
                // get cumulative distribution
                for (size_t k=0; k<_c_ops.size(); k++){
                    _c_ops[k] -> apply_to_state(state, buffer);
                    cumulative_dist[k] = buffer->get_squared_norm()+prob_sum;
                    prob_sum = cumulative_dist[k];
                }
                auto jump_r = _random.uniform() * cumulative_dist[cumulative_dist.size()-1];
                auto ite =
                    std::lower_bound(cumulative_dist.begin(), cumulative_dist.end(), jump_r);
                auto index = std::distance(cumulative_dist.begin(), ite);
                _c_ops[index] -> apply_to_state(state, buffer);
                buffer -> normalize(buffer -> get_squared_norm());
                state -> load(buffer);
                r = _random.uniform();
            }
        }
        delete k1;
        delete k2;
        delete k3;
        delete k4;
        delete buffer;
    }
    
};



