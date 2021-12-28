
#pragma once

#include "gate.hpp"
#include "state.hpp"
#include "utility.hpp"
#include "observable.hpp"
#include "general_quantum_operator.hpp"


class NoisyEvolution : public QuantumGateBase
{
private:
    Random _random;
    const Observable* _hamiltonian;
    const GeneralQuantumOperator* _effective_hamiltoian;
    const std::vector<GeneralQuantumOperator*> _c_ops;
    const std::vector<GeneralQuantumOperator*> _c_ops_dagger;
    const double _time;
    const double _dt;
    

public:
    NoisyEvolution(Observable* hamiltonian, 
                   std::vector<GeneralQuantumOperator*> c_ops,
                   double time,
                   double dt=1e-6){
        _hamiltonian = hamiltonian->copy();
        _c_ops = c_ops;
        for (auto const & op: _c_ops){
            _c_ops_dagger.push_back(op->get_dagger());
        } 
        _effective_hamiltoian = hamiltonian->copy();
        for (int k=0; k<_c_ops.size(); k++){
            auto cdagc = (*_c_ops_dagger[k])*(*_c_ops[k])/2*(-1.i)
            for (int j=0; j<cdagc.get_term_count(); j++){
                _effective_hamiltoian->add_operator(cdagc.get_term(j))
            }
        }
        _time = time;
        _dt = dt;
    };
    ~NoisyEvolution(){
        delete _hamiltonian;
        delete _effective_hamiltoian;
        for (int k=0; k<_c_ops.size(); k++){
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
        return new NoisyEvolution(_hamiltonian, _c_ops, _time, _dt);
    }

    virtual void update_quantum_state(QuantumStateBase* state) {
        double t = 0;
        double r;
        auto n_qubits = state->qubit_count;
        auto k1 = state->copy(); //vector for Runge-Kutta k
        auto k2 = state->copy(); //vector for Runge-Kutta k
        auto k3 = state->copy(); //vector for Runge-Kutta k
        auto k4 = state->copy(); //vector for Runge-Kutta k
        auto tmp_state_1 = state->copy();
        auto tmp_state_2 = state->copy();
        auto work_state = state->copy();
        for (double t=0; t <= _time; t += dt){
            double r = random.uniform();
            // Runge-Kutta evolution
            // k1
            _effective_hamiltonian.apply_to_state(work_state, *tmp_state_2, k1);
            k1 -> multiply_coef(-1.i);

            //k2
            tmp_state_1.load(k1); 
            tmp_state_1 -> multiply_coef(_dt/2);
            tmp_state_2.add_state(tmp_state_1);
            _effective_hamiltonian.apply_to_state(work_state, *tmp_state_2, k2);
            k2 -> multiply_coef(-1.i);
            
            // k3
            tmp_state_1 -> load(k2);
            tmp_state_1 -> multiply_coef(_dt/2);
            tmp_state_2.load(state);
            tmp_state_2.add_state(tmp_state_1);
            _effective_hamiltonian.apply_to_state(work_state, *tmp_state_2, k3);
            k3 -> multiply_coef(-1.i);
            
            // k4
            tmp_state_1 -> load(k3);
            tmp_state_1 -> multiply_coef(_dt);
            tmp_state_2.load(state);
            tmp_state_2.add_state(tmp_state_1);
            _effective_hamiltonian.apply_to_state(work_state, *tmp_state_2, k4);
            k4 -> multiply_coef(-1.i);

            // add them together
            k1->multiply_coef(_dt/6);
            k2->multiply_coef(_dt/3);
            k3->multiply_coef(_dt/3);
            k4->multiply_coef(_dt/6);
            state->add_state(k1);
            state->add_state(k2);
            state->add_state(k3);
            state->add_state(k4);
            
            auto norm = state -> get_squared_norm();
            if (norm <= r){
                std::vector<double> cumulative_dist(_c_ops.size())
                double prob_sum = 0.;
                // get cumulative distribution
                _c_ops[0] -> apply_to_state(work_state, *state, tmp_state_1);
                cumulative_dist[0] = tmp_state_1->get_squared_norm();    
                for (int k=1; k<_c_ops.size(); k++){
                    _c_ops[k] -> apply_to_state(work_state, *state, tmp_state_1);
                    cumulative_dist[k] = tmp_state_1->get_squared_norm()+cumulative_dist[k-1];
                }
                jump_r = _random.uniform() * cumulative_dist[cumulative_dist.size()-1]
            }
        }
    }
};



