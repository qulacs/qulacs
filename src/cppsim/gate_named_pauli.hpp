#pragma once

#include <csim/update_ops.hpp>
#include <csim/update_ops_dm.hpp>

#include "gate.hpp"
#include "pauli_operator.hpp"
#include "state.hpp"
#include "utility.hpp"
#ifdef _USE_GPU
#include <gpusim/update_ops_cuda.h>
#endif
/**
 * \~japanese-en 複数の量子ビットに作用するPauli演算子を作用させるゲート
 */
class ClsPauliGate : public QuantumGateBase {
protected:
    PauliOperator* _pauli;

public:
    /**
     * \~japanese-en コンストラクタ
     *
     * 使用したパウリ演算子はクラスにて解放される
     * @param pauli 作用させるパウリ演算子
     */
    explicit ClsPauliGate(PauliOperator* pauli) {
        _pauli = pauli;
        this->_name = "Pauli";
        auto target_index_list = _pauli->get_index_list();
        auto pauli_id_list = _pauli->get_pauli_id_list();
        for (UINT index = 0; index < target_index_list.size(); ++index) {
            UINT commutation_relation = 0;
            if (pauli_id_list[index] == 1)
                commutation_relation = FLAG_X_COMMUTE;
            else if (pauli_id_list[index] == 2)
                commutation_relation = FLAG_Y_COMMUTE;
            else if (pauli_id_list[index] == 3)
                commutation_relation = FLAG_Z_COMMUTE;
            this->_target_qubit_list.push_back(TargetQubitInfo(
                target_index_list[index], commutation_relation));
        }
    };
    /**
     * \~japanese-en デストラクタ
     */
    virtual ~ClsPauliGate() { delete _pauli; }
    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
        auto target_index_list = _pauli->get_index_list();
        auto pauli_id_list = _pauli->get_pauli_id_list();
        if (state->is_state_vector()) {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                multi_qubit_Pauli_gate_partial_list_host(
                    target_index_list.data(), pauli_id_list.data(),
                    (UINT)target_index_list.size(), state->data(), state->dim,
                    state->get_cuda_stream(), state->device_number);
                // _update_func_gpu(this->_target_qubit_list[0].index(), _angle,
                // state->data(), state->dim);
            } else {
                multi_qubit_Pauli_gate_partial_list(target_index_list.data(),
                    pauli_id_list.data(), (UINT)target_index_list.size(),
                    state->data_c(), state->dim);
            }
#else
            multi_qubit_Pauli_gate_partial_list(target_index_list.data(),
                pauli_id_list.data(), (UINT)target_index_list.size(),
                state->data_c(), state->dim);
#endif
        } else {
            dm_multi_qubit_Pauli_gate_partial_list(target_index_list.data(),
                pauli_id_list.data(), (UINT)target_index_list.size(),
                state->data_c(), state->dim);
        }
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual ClsPauliGate* copy() const override {
        return new ClsPauliGate(_pauli->copy());
    };

    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        get_Pauli_matrix(matrix, _pauli->get_pauli_id_list());
    }

    /**
     * \~japanese-en ptreeに変換する
     *
     * @return ptree
     */
    virtual boost::property_tree::ptree to_ptree() const override {
        boost::property_tree::ptree pt;
        pt.add("name", "PauliGate");
        pt.add_child("pauli", _pauli->to_ptree());
        return pt;
    }

    virtual QuantumGateBase* create_gate_whose_qubit_indices_are_replaced(
        const std::vector<UINT>& target_index_list,
        const std::vector<UINT>& control_index_list) const override {
        if (_target_qubit_list.size() != target_index_list.size()) {
            throw InvalidQubitCountException(
                "Error: "
                "QuantumGateBase::create_gate_whose_qubit_indices_is_"
                "replaced\n qubit count of target_index_list does not match.");
        }
        if (_control_qubit_list.size() != control_index_list.size()) {
            throw InvalidQubitCountException(
                "Error: "
                "QuantumGateBase::create_gate_whose_qubit_indices_is_"
                "replaced\n qubit count of control_index_list does not match.");
        }
        PauliOperator* pauli = new PauliOperator(
            target_index_list, _pauli->get_pauli_id_list(), _pauli->get_coef());
        auto ret = new ClsPauliGate(pauli);
        delete pauli;
        return ret;
    }

    virtual QuantumGateBase* get_inverse() const override { return copy(); }
};

/**
 * \~japanese-en 複数の量子ビットに作用するPauli演算子で回転するゲート
 */
class ClsPauliRotationGate : public QuantumGateBase {
protected:
    double _angle;
    PauliOperator* _pauli;

public:
    /**
     * \~japanese-en コンストラクタ
     *
     * 使用したパウリ演算子はクラスにて解放される
     * @param angle 回転角
     * @param pauli 作用させるパウリ演算子
     */
    ClsPauliRotationGate(double angle, PauliOperator* pauli) : _angle(angle) {
        _pauli = pauli;
        this->_name = "Pauli-rotation";
        auto target_index_list = _pauli->get_index_list();
        auto pauli_id_list = _pauli->get_pauli_id_list();
        for (UINT index = 0; index < target_index_list.size(); ++index) {
            UINT commutation_relation = 0;
            if (pauli_id_list[index] == 1)
                commutation_relation = FLAG_X_COMMUTE;
            else if (pauli_id_list[index] == 2)
                commutation_relation = FLAG_Y_COMMUTE;
            else if (pauli_id_list[index] == 3)
                commutation_relation = FLAG_Z_COMMUTE;
            this->_target_qubit_list.push_back(TargetQubitInfo(
                target_index_list[index], commutation_relation));
        }
    };
    /**
     * \~japanese-en デストラクタ
     */
    virtual ~ClsPauliRotationGate() { delete _pauli; }
    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
        auto target_index_list = _pauli->get_index_list();
        auto pauli_id_list = _pauli->get_pauli_id_list();
        if (state->is_state_vector()) {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                multi_qubit_Pauli_rotation_gate_partial_list_host(
                    target_index_list.data(), pauli_id_list.data(),
                    (UINT)target_index_list.size(), _angle, state->data(),
                    state->dim, state->get_cuda_stream(), state->device_number);
            } else {
                multi_qubit_Pauli_rotation_gate_partial_list(
                    target_index_list.data(), pauli_id_list.data(),
                    (UINT)target_index_list.size(), _angle, state->data_c(),
                    state->dim);
            }
#else
            multi_qubit_Pauli_rotation_gate_partial_list(
                target_index_list.data(), pauli_id_list.data(),
                (UINT)target_index_list.size(), _angle, state->data_c(),
                state->dim);
#endif
        } else {
            dm_multi_qubit_Pauli_rotation_gate_partial_list(
                target_index_list.data(), pauli_id_list.data(),
                (UINT)target_index_list.size(), _angle, state->data_c(),
                state->dim);
        }
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual ClsPauliRotationGate* copy() const override {
        return new ClsPauliRotationGate(_angle, _pauli->copy());
    };

    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        get_Pauli_matrix(matrix, _pauli->get_pauli_id_list());
        std::complex<double> imag_unit(0, 1);
        matrix = cos(_angle / 2) *
                     ComplexMatrix::Identity(matrix.rows(), matrix.cols()) +
                 imag_unit * sin(_angle / 2) * matrix;
    }

    /**
     * \~japanese-en ptreeに変換する
     *
     * @return ptree
     */
    virtual boost::property_tree::ptree to_ptree() const override {
        boost::property_tree::ptree pt;
        pt.add("name", "PauliRotationGate");
        pt.add("angle", _angle);
        pt.add_child("pauli", _pauli->to_ptree());
        return pt;
    }

    virtual QuantumGateBase* create_gate_whose_qubit_indices_are_replaced(
        const std::vector<UINT>& target_index_list,
        const std::vector<UINT>& control_index_list) const override {
        if (_target_qubit_list.size() != target_index_list.size()) {
            throw InvalidQubitCountException(
                "Error: "
                "QuantumGateBase::create_gate_whose_qubit_indices_is_"
                "replaced\n qubit count of target_index_list does not match.");
        }
        if (_control_qubit_list.size() != control_index_list.size()) {
            throw InvalidQubitCountException(
                "Error: "
                "QuantumGateBase::create_gate_whose_qubit_indices_is_"
                "replaced\n qubit count of control_index_list does not match.");
        }
        PauliOperator* pauli = new PauliOperator(
            target_index_list, _pauli->get_pauli_id_list(), _pauli->get_coef());
        auto ret = new ClsPauliRotationGate(_angle, pauli);
        delete pauli;
        return ret;
    }

    virtual ClsPauliRotationGate* get_inverse(void) const override {
        return new ClsPauliRotationGate(-this->_angle, this->_pauli->copy());
    }
};
