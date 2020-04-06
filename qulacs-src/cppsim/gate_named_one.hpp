#pragma once

#ifndef _MSC_VER
extern "C"{
#include <csim/update_ops.h>
#include <csim/update_ops_dm.h>
}
#else
#include <csim/update_ops.h>
#include <csim/update_ops_dm.h>
#endif

#include "gate_named.hpp"
#include <cmath>

#ifdef _USE_GPU
#include <gpusim/update_ops_cuda.h>
#endif

/**
 * \~japanese-en Identityゲート
 */
/**
 * \~english Identity gate
 */
class ClsIGate : public QuantumGate_OneQubit{
    static void idling(UINT,CTYPE*,ITYPE){};
	static void idling_gpu(UINT, void*, ITYPE, void*, UINT) {};
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     */
    ClsIGate(UINT target_qubit_index) {
        this->_update_func = ClsIGate::idling;
		this->_update_func_dm = ClsIGate::idling;
		this->_update_func_gpu = ClsIGate::idling_gpu;
		this->_name = "I";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, FLAG_X_COMMUTE | FLAG_Y_COMMUTE | FLAG_Z_COMMUTE ));
        this->_gate_property = FLAG_PAULI | FLAG_CLIFFORD | FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2,2);
        this->_matrix_element << 1,0,0,1;
    }
};

/**
 * \~japanese-en Pauli-\f$X\f$ゲート
 */
/**
 * \~english Pauli-\f$X\f$ gate
 */
class ClsXGate : public QuantumGate_OneQubit{
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     */
    ClsXGate(UINT target_qubit_index) {
        this->_update_func = X_gate;
		this->_update_func_dm = dm_X_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = X_gate_host;
#endif
        this->_name = "X";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, FLAG_X_COMMUTE ));
        this->_gate_property = FLAG_PAULI | FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2,2);
        this->_matrix_element << 0,1,1,0;
    }
};

/**
 * \~japanese-en Pauli-\f$Y\f$ゲート
 */
/**
 * \~english Pauli-\f$Y\f$ gate
 */
class ClsYGate : public QuantumGate_OneQubit{
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     */
    ClsYGate(UINT target_qubit_index) {
        this->_update_func = Y_gate;
		this->_update_func_dm = dm_Y_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = Y_gate_host;
#endif
        this->_name = "Y";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, FLAG_Y_COMMUTE ));
        this->_gate_property = FLAG_PAULI | FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2,2);
        this->_matrix_element << 0,-1.i,1.i,0;
    }
};

/**
 * \~japanese-en Pauli-\f$Z\f$ゲート
 */
/**
 * \~english Pauli-\f$Z\f$ gate
 */
class ClsZGate : public QuantumGate_OneQubit{
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     */
    ClsZGate(UINT target_qubit_index){
        this->_update_func = Z_gate;
		this->_update_func_dm = dm_Z_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = Z_gate_host;
#endif
	    this->_name = "Z";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, FLAG_Z_COMMUTE ));
        this->_gate_property = FLAG_PAULI | FLAG_CLIFFORD | FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2,2);
        this->_matrix_element << 1,0,0,-1;
    }
};

/**
 * \~japanese-en Pauli-\f$H\f$ゲート
 */
/**
 * \~english Pauli-\f$H\f$ gate
 */
class ClsHGate : public QuantumGate_OneQubit{
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     */
    ClsHGate(UINT target_qubit_index) {
        this->_update_func = H_gate;
		this->_update_func_dm = dm_H_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = H_gate_host;
#endif
	    this->_name = "H";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, 0));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 1, 1, -1;
        this->_matrix_element /= sqrt(2.);
    }
};

/**
 * \~japanese-en Pauli-\f$S\f$ゲート
 */
/**
 * \~english Pauli-\f$S\f$ gate
 */
class ClsSGate : public QuantumGate_OneQubit{
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     */
    ClsSGate(UINT target_qubit_index){
        this->_update_func = S_gate;
		this->_update_func_dm = dm_S_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = S_gate_host;
#endif
        this->_name = "S";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, 0));
        this->_gate_property = FLAG_CLIFFORD | FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 0, 0, 1.i;
    }
};

/**
 * \~japanese-en Pauli-\f$S^{\dagger}\f$ゲート
 */
/**
 * \~english Pauli-\f$S^{\dagger}\f$ gate
 */
class ClsSdagGate : public QuantumGate_OneQubit{
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     */
    ClsSdagGate(UINT target_qubit_index){
        this->_update_func = Sdag_gate;
		this->_update_func_dm = dm_Sdag_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = Sdag_gate_host;
#endif
        this->_name = "Sdag";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, 0));
        this->_gate_property = FLAG_CLIFFORD | FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 0, 0, -1.i;
    }
};

/**
 * \~japanese-en Pauli-\f$T\f$ゲート
 */
/**
 * \~english Pauli-\f$T\f$ gate
 */
class ClsTGate : public QuantumGate_OneQubit{
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     */
    ClsTGate(UINT target_qubit_index){
        this->_update_func = T_gate;
		this->_update_func_dm = dm_T_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = T_gate_host;
#endif
        this->_name = "T";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, 0));
        this->_gate_property = FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 0, 0, (1.+1.i)/sqrt(2.);
    }
};

/**
 * \~japanese-en Pauli-\f$T^{\dagger}\f$ゲート
 */
/**
 * \~english Pauli-\f$T^{\dagger}\f$ gate
 */
class ClsTdagGate : public QuantumGate_OneQubit{
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     */
    ClsTdagGate(UINT target_qubit_index){
        this->_update_func = Tdag_gate;
		this->_update_func_dm = dm_Tdag_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = Tdag_gate_host;
#endif
        this->_name = "Tdag";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, 0));
        this->_gate_property = FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 0, 0, (1. - 1.i) / sqrt(2.);
    }
};

/**
 * \~japanese-en Pauli-\f$\sqrt{X}\f$ゲート
 */
/**
 * \~english Pauli-\f$\sqrt{X}\f$ gate
 */
class ClsSqrtXGate : public QuantumGate_OneQubit {
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     */
    ClsSqrtXGate(UINT target_qubit_index) {
        this->_update_func = sqrtX_gate;
		this->_update_func_dm = dm_sqrtX_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = sqrtX_gate_host;
#endif
        this->_name = "sqrtX";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, FLAG_X_COMMUTE));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
    }
};

/**
 * \~japanese-en Pauli-\f$\sqrt{X}^{\dagger}\f$ゲート
 */
/**
 * \~english Pauli-\f$\sqrt{X}^{\dagger}\f$ gate
 */
class ClsSqrtXdagGate : public QuantumGate_OneQubit {
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     */
    ClsSqrtXdagGate(UINT target_qubit_index) {
        this->_update_func = sqrtXdag_gate;
		this->_update_func_dm = dm_sqrtXdag_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = sqrtXdag_gate_host;
#endif
        this->_name = "sqrtXdag";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, FLAG_X_COMMUTE));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 0.5-0.5i, 0.5+0.5i, 0.5+0.5i, 0.5-0.5i;
    }
};

/**
 * \~japanese-en Pauli-\f$\sqrt{Y}\f$ゲート
 */
/**
 * \~english Pauli-\f$\sqrt{Y}\f$ gate
 */
class ClsSqrtYGate : public QuantumGate_OneQubit {
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     */
    ClsSqrtYGate(UINT target_qubit_index) {
        this->_update_func = sqrtY_gate;
		this->_update_func_dm = dm_sqrtY_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = sqrtY_gate_host;
#endif
        this->_name = "sqrtY";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, FLAG_Y_COMMUTE));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 0.5+0.5i, -0.5-0.5i, 0.5+0.5i, 0.5+0.5i;
    }
};

/**
 * \~japanese-en Pauli-\f$\sqrt{Y}^{\dagger}\f$ゲート
 */
/**
 * \~english Pauli-\f$\sqrt{Y}^{\dagger}\f$ gate
 */
class ClsSqrtYdagGate : public QuantumGate_OneQubit {
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     */
    ClsSqrtYdagGate(UINT target_qubit_index) {
        this->_update_func = sqrtYdag_gate;
		this->_update_func_dm = dm_sqrtYdag_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = sqrtYdag_gate_host;
#endif
        this->_name = "sqrtYdag";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, FLAG_Y_COMMUTE));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 0.5-0.5i, 0.5-0.5i, -0.5+0.5i, 0.5-0.5i;
    }
};

/**
 * \~japanese-en 作用する量子ビットを0状態へ射影するゲート
 */
/**
 * \~english A gate that projects a working qubit into state 0
 */
class ClsP0Gate : public QuantumGate_OneQubit{
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     */
    ClsP0Gate(UINT target_qubit_index){
        this->_update_func = P0_gate;
		this->_update_func_dm = dm_P0_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = P0_gate_host;
#endif
        this->_name = "Projection-0";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, 0));
        this->_gate_property = FLAG_CLIFFORD | FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 0, 0, 0;
    }
};

/**
 * \~japanese-en 作用する量子ビットを1状態へ射影するゲート
 */
/**
 * \~english A gate that projects a working qubit into state 1
 */
class ClsP1Gate : public QuantumGate_OneQubit{
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     */
    ClsP1Gate(UINT target_qubit_index){
        this->_update_func = P1_gate;
		this->_update_func_dm = dm_P1_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = P1_gate_host;
#endif
        this->_name = "Projection-1";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, 0));
        this->_gate_property = FLAG_CLIFFORD | FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 0, 0, 0, 1;
    }
};

/**
 * \~japanese-en \f$X\f$回転ゲート
 */ 
/**
 * \~english \f$X\f$ rotation gate
 */ 
class ClsRXGate : public QuantumGate_OneQubitRotation{
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     * @param angle 回転角
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     * @param angle Rotation angle
     */
    ClsRXGate(UINT target_qubit_index, double angle) : QuantumGate_OneQubitRotation(angle) {
        this->_update_func = RX_gate;
		this->_update_func_dm = dm_RX_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = RX_gate_host;
#endif
        this->_name = "X-rotation";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, FLAG_X_COMMUTE ));
        this->_matrix_element = ComplexMatrix::Zero(2,2);
        this->_matrix_element << cos(_angle/2), sin(_angle/2) * 1.i, sin(_angle/2) * 1.i, cos(_angle/2);
    }
};

/**
 * \~japanese-en \f$Y\f$回転ゲート
 */ 
/**
 * \~english \f$Y\f$ rotation gate
 */ 
class ClsRYGate : public QuantumGate_OneQubitRotation{
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     * @param angle 回転角
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     * @param angle Rotation angle
     */
    ClsRYGate(UINT target_qubit_index, double angle): QuantumGate_OneQubitRotation(angle){
        this->_update_func = RY_gate;
		this->_update_func_dm = dm_RY_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = RY_gate_host;
#endif
        this->_name = "Y-rotation";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, FLAG_Y_COMMUTE));
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << cos(_angle/2), sin(_angle/2), -sin(_angle/2), cos(_angle/2);
    }
};

/**
 * \~japanese-en \f$Z\f$回転ゲート
 */ 
/**
 * \~english \f$Z\f$ rotation gate
 */ 
class ClsRZGate : public QuantumGate_OneQubitRotation{
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param target_qubit_index ターゲットの量子ビットの添え字
     * @param angle 回転角
     */
    /**
     * \~english Construct
     * 
     * @param target_qubit_index Subscript of target qubit
     * @param angle Rotation angle
     */
    ClsRZGate(UINT target_qubit_index, double angle): QuantumGate_OneQubitRotation(angle){
        this->_update_func = RZ_gate;
		this->_update_func_dm = dm_RZ_gate;
#ifdef _USE_GPU
		this->_update_func_gpu = RZ_gate_host;
#endif
        this->_name = "Z-rotation";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, FLAG_Z_COMMUTE));
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << cos(_angle/2)+1.i*sin(_angle/2), 0, 0, cos(_angle/2) - 1.i * sin(_angle/2);
    }
};
