
#pragma once

#include "gate.hpp"
#include "state.hpp"
#include "utility.hpp"


/**
 * \~japanese-en 1量子ビットを対象とする回転角固定のゲートのクラス
 */
class QuantumGate_OneQubit : public QuantumGateBase{
protected:
    typedef void (T_UPDATE_FUNC)(UINT, CTYPE*, ITYPE);
    T_UPDATE_FUNC* _update_func;
    ComplexMatrix _matrix_element;

    QuantumGate_OneQubit() {};
public:
    /**
     * \~japanese-en 量子状態を更新する
     * 
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override{
        _update_func(this->_target_qubit_list[0].index(), state->data_c(), state->dim);
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     * 
     * @return 自身のディープコピー
     */
    virtual QuantumGateBase* copy() const override{
        return new QuantumGate_OneQubit(*this);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     * 
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override{
        matrix = this->_matrix_element;
    }
};

/**
 * \~japanese-en 2量子ビットを対象とする回転角固定のゲートのクラス
 */
class QuantumGate_TwoQubit : public QuantumGateBase{
protected:
    typedef void (T_UPDATE_FUNC)(UINT, UINT, CTYPE*, ITYPE);
    T_UPDATE_FUNC* _update_func;
    ComplexMatrix _matrix_element;

    QuantumGate_TwoQubit() {};
public:
    /**
     * \~japanese-en 量子状態を更新する
     * 
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override{
        _update_func(this->_target_qubit_list[0].index(), this->_target_qubit_list[1].index(), state->data_c(), state->dim);
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     * 
     * @return 自身のディープコピー
     */
    virtual QuantumGateBase* copy() const override{
        return new QuantumGate_TwoQubit(*this);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     * 
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        matrix = this->_matrix_element;
    }
};

/**
 * \~japanese-en 1量子ビットを対象とし1量子ビットにコントロールされる回転角固定のゲートのクラス
 */
class QuantumGate_OneControlOneTarget : public QuantumGateBase {
protected:
    typedef void (T_UPDATE_FUNC)(UINT, UINT, CTYPE*, ITYPE);
    T_UPDATE_FUNC* _update_func;
    ComplexMatrix _matrix_element;

    QuantumGate_OneControlOneTarget() {};
public:
    /**
     * \~japanese-en 量子状態を更新する
     * 
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
        _update_func(this->_control_qubit_list[0].index(), this->_target_qubit_list[0].index(), state->data_c(), state->dim);
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     * 
     * @return 自身のディープコピー
     */
    virtual QuantumGateBase* copy() const override {
        return new QuantumGate_OneControlOneTarget(*this);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     * 
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        matrix = this->_matrix_element;
    }
};

/**
 * \~japanese-en 1量子ビットを対象とする回転ゲートのクラス
 */
class QuantumGate_OneQubitRotation : public QuantumGateBase{
protected:
    typedef void (T_UPDATE_FUNC)(UINT, double, CTYPE*, ITYPE);
    T_UPDATE_FUNC* _update_func;
    ComplexMatrix _matrix_element;
    double _angle;

    QuantumGate_OneQubitRotation(double angle) : _angle(angle) {};
public:
    /**
     * \~japanese-en 量子状態を更新する
     * 
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override{
        _update_func(this->_target_qubit_list[0].index(), _angle, state->data_c(), state->dim);
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     * 
     * @return 自身のディープコピー
     */
    virtual QuantumGateBase* copy() const override{
        return new QuantumGate_OneQubitRotation(*this);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     * 
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        matrix = this->_matrix_element;
    }
};


