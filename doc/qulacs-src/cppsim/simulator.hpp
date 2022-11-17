
#pragma once

#include <cstdlib>
#include "type.hpp"
#include "circuit.hpp"
class QuantumStateBase;
class HermitianQuantumOperator;
typedef HermitianQuantumOperator Observable;


/**
 * \~japanese-en 量子回路をシミュレートするためのクラス
 */
/**
 * \~english A class for simulating quantum circuits
 */
class DllExport QuantumCircuitSimulator {
private:
    QuantumCircuit* _circuit;
    QuantumStateBase* _state;
    QuantumStateBase* _buffer;
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param circuit シミュレートする量子回路 
     * @param initial_state 初期量子状態。デフォルト値はNULLで、NULLの場合は0状態に初期化される。
     */
    /**
     * \~english Constructor
     * 
     * @param circuit Simulating quantum circuit 
     * @param initial_state Initial quantum state. Default is NULL, and it is initialized to 0 state.
     */
    QuantumCircuitSimulator(QuantumCircuit* circuit, QuantumStateBase* initial_state = NULL);

    /**
     * \~japanese-en デストラクタ
     */
    /**
     * \~english Destructor
     */
    ~QuantumCircuitSimulator();

    /**
     * \~japanese-en 量子状態を計算基底に初期化する
     * 
     * @param computationl_basis 初期化する計算基底を二進数にした値
     */
    /**
     * \~english Initialize quantum states to computational basis
     * 
     * @param computationl_basis Binary number of the calculation base to be initialized
     */
    void initialize_state(ITYPE computationl_basis = 0);
    /**
     * \~japanese-en ランダムな量子状態に初期化する。
     */
    /**
     * \~english Initialize to random quantum state
     */
    void initialize_random_state();

    /**
     * \~japanese-en 量子回路全体シミュレートする。
     */
    /**
     * \~english Simulate the entire quantum circuit
     */
    void simulate();

    /**
     * \~japanese-en 量子回路を<code>start</code>から<code>end</code>までの区間シミュレートする
     * 
     * @param start シミュレートを開始する添え字
     * @param end シミュレートを終了する添え字
     */
    /**
     * \~english Simulate quantum circuit in the range from <code>start</code> to <code>end</code>
     * 
     * @param start Index to start simulation
     * @param end Index to end simulation
     */
    void simulate_range(UINT start, UINT end);

    /**
     * \~japanese-en 現在の量子状態の受け取ったオブザーバブルの期待値を計算する。
     * 
     * @param observable オブザーバブル
     * @return 期待値
     */
    /**
     * \~english Calculate the expectation value of the received observable for the current quantum state.
     * 
     * @param observable Observable
     * @return Expectation value
     */
    CPPCTYPE get_expectation_value(const Observable* observable);

    /**
     * \~japanese-en 量子回路中のゲートの数を取得する
     * 
     * @return ゲートの数
     */
    /**
     * \~english Obtain the number of gates in a quantum circuit
     * 
     * @return Number of gates
     */
    UINT get_gate_count();

    /**
     * \~japanese-en 量子状態をバッファへコピーする
     */
    /**
     * \~english Copy quantum state to buffer
     */
    void copy_state_to_buffer();
    /**
     * \~japanese-en バッファの量子状態を現在の量子状態へコピーする
     */
    /**
     * \~english Copy buffer quantum state to current quantum state
     */
    void copy_state_from_buffer();
    /**
     * \~japanese-en 現在の量子状態をバッファと交換する
     */
    /**
     * \~english Exchange the currentum quantum state to buffer quantum state
     */
    void swap_state_and_buffer();

    /**
     * \~japanese-en 現在の量子状態のポインタを取得する
     * 
     * @return 量子状態のポインタ
     */
    /**
     * \~english Obtain pointer to current quantum state
     * 
     * @return Pointer of quantum state
     */
    const QuantumStateBase* get_state_ptr() const { return _state; }
};


