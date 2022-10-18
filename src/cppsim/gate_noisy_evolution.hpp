#pragma once

#include <Eigen/Dense>
#include <csim/stat_ops.hpp>
#include <csim/update_ops.hpp>
#include <csim/update_ops_dm.hpp>
#include <cstring>
#include <fstream>
#include <numeric>

#include "exception.hpp"
#include "gate.hpp"
#include "gate_merge.hpp"
#include "gate_named_pauli.hpp"
#include "general_quantum_operator.hpp"
#include "observable.hpp"
#include "pauli_operator.hpp"
#include "state.hpp"
#include "type.hpp"
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
     *
     * 割線法を利用する場合、normは時間に対して広義単調減少であることが必要。
     */
    virtual double _find_collapse(QuantumStateBase* k1, QuantumStateBase* k2,
        QuantumStateBase* k3, QuantumStateBase* k4,
        QuantumStateBase* prev_state, QuantumStateBase* now_state,
        double target_norm, double dt, bool use_secant_method = true);
    virtual double _find_collapse_original(QuantumStateBase* k1,
        QuantumStateBase* k2, QuantumStateBase* k3, QuantumStateBase* k4,
        QuantumStateBase* prev_state, QuantumStateBase* now_state,
        double target_norm, double dt);
    virtual void _evolve_one_step(QuantumStateBase* k1, QuantumStateBase* k2,
        QuantumStateBase* k3, QuantumStateBase* k4, QuantumStateBase* buffer,
        QuantumStateBase* state, double dt);

public:
    ClsNoisyEvolution(Observable* hamiltonian,
        std::vector<GeneralQuantumOperator*> c_ops, double time,
        double dt = 1e-6);
    ~ClsNoisyEvolution();

    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix&) const override {
        throw NotImplementedException(
            "Error: "
            "ClsNoisyEvolution::set_matrix(ComplexMatrix&): Gate-matrix of "
            "noisy evolution cannot be defined.");
    }

    /**
     * \~japanese-en 乱数シードをセットする
     *
     * @param seed シード値
     */
    virtual void set_seed(int seed) override { _random.set_seed(seed); };

    virtual ClsNoisyEvolution* copy() const override {
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

    virtual void update_quantum_state(QuantumStateBase* state);
};

/*
元のバージョンで、時間発展をルンゲクッタ法を使っていたところを、　行列を抜き出して対角化で求めます。
性質上、　
・　hamiltonianやc_opsに出現するビットの種類数が多いと遅い、　多分3~4ビットあたりが境目？
・　なので、操作するビットがたくさんある場合、それが独立なビット集合に分けられる場合、　集合ごとにこのクラスを使ってください
・　ゲートを作るときに重い操作を行うので、同じゲートに対して何回も演算すると速い

*/

class ClsNoisyEvolution_fast : public QuantumGateBase {
private:
    Random _random;
    Observable* _hamiltonian;
    GeneralQuantumOperator* _effective_hamiltonian;
    std::vector<GeneralQuantumOperator*> _c_ops;  // collapse operator
    std::vector<GeneralQuantumOperator*>
        _c_ops_dagger;        // collapse operator dagger
    double _time;             // evolution time
    double _norm_tol = 1e-4;  // accuracy in solving <psi|psi>=r
    // rが乱数なら精度不要なので1e-4にした
    int _find_collapse_max_steps = 100;
    ComplexVector eigenvalue_mto;
    QuantumGateMatrix* eigenMatrixGate = nullptr;
    QuantumGateMatrix* eigenMatrixRevGate = nullptr;
    bool gate_none = false;

    /**
     * \~japanese-en collapse が起こるタイミング (norm = rになるタイミング)
     * を見つける。 この関数内で norm=rになるタイミングまでの evolution
     * が行われる。
     *
     * 割線法を利用する場合、normは時間に対して広義単調減少であることが必要。
     */
    virtual double _find_collapse(QuantumStateBase* prev_state,
        QuantumStateBase* now_state, double target_norm, double t_step);

    /**
     * \~japanese-en
     * この関数が終わった時点で、時間発展前の状態は buffer に格納される。
     */
    virtual void _evolve_one_step(QuantumStateBase* state, double t_step);

public:
    ClsNoisyEvolution_fast(Observable* hamiltonian,
        std::vector<GeneralQuantumOperator*> c_ops, double time);

    ~ClsNoisyEvolution_fast() {
        delete _hamiltonian;
        delete _effective_hamiltonian;
        for (size_t k = 0; k < _c_ops.size(); k++) {
            delete _c_ops[k];
            delete _c_ops_dagger[k];
        }
        delete eigenMatrixGate;
        delete eigenMatrixRevGate;
    };

    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        throw NotImplementedException(
            "Error: "
            "ClsNoisyEvolution::set_matrix(ComplexMatrix&): Gate-matrix of "
            "noisy evolution cannot be defined.");
    }

    /**
     * \~japanese-en 乱数シードをセットする
     *
     * @param seed シード値
     */
    virtual void set_seed(int seed) override { _random.set_seed(seed); };

    virtual ClsNoisyEvolution_fast* copy() const override {
        return new ClsNoisyEvolution_fast(_hamiltonian, _c_ops, _time);
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
    virtual void update_quantum_state(QuantumStateBase* state);
};
