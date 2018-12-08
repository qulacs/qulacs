
#pragma once

#ifndef _MSC_VER
extern "C"{
#include <csim/memory_ops.h>
#include <csim/stat_ops.h>
#include <csim/update_ops.h>
}
#else
#include <csim/memory_ops.h>
#include <csim/stat_ops.h>
#include <csim/update_ops.h>
#endif

#include "type.hpp"
#include "utility.hpp"
#include <vector>

/**
 * \~japanese-en 量子状態の基底クラス
 */
class QuantumStateBase{
protected:
    ITYPE _dim;
    UINT _qubit_count;
    std::vector<UINT> _classical_register;
public:
    const UINT& qubit_count = _qubit_count; /**< \~japanese-en 量子ビット数 */
    const ITYPE& dim = _dim; /**< \~japanese-en 量子状態の次元 */
    const std::vector<UINT>& classical_register = _classical_register; /**< \~japanese-en 古典ビットのレジスタ */

    /**
     * \~japanese-en コンストラクタ
     * 
     * @param qubit_count_ 量子ビット数
     */
    QuantumStateBase(UINT qubit_count_){
        this->_qubit_count = qubit_count_;
        this->_dim = 1ULL << qubit_count_;
    }
    /**
     * \~japanese-en デストラクタ
     */
    virtual ~QuantumStateBase(){}

    /**
     * \~japanese-en 量子状態を計算基底の0状態に初期化する
     */
    virtual void set_zero_state() = 0;

    /**
     * \~japanese-en 量子状態を<code>comp_basis</code>の基底状態に初期化する
     * 
     * @param comp_basis 初期化する基底を表す整数
     */
    virtual void set_computational_basis(ITYPE comp_basis) = 0;

    /**
     * \~japanese-en 量子状態をHaar randomにサンプリングされた量子状態に初期化する
     */
    virtual void set_Haar_random_state() = 0;

    /**
     * \~japanese-en 量子状態をシードを用いてHaar randomにサンプリングされた量子状態に初期化する
     */
    virtual void set_Haar_random_state(UINT seed) = 0;

    /**
     * \~japanese-en <code>target_qubit_index</code>の添え字の量子ビットを測定した時、0が観測される確率を計算する。
     * 
     * 量子状態は変更しない。
     * @param target_qubit_index 
     * @return double 
     */
    virtual double get_zero_probability(UINT target_qubit_index) const = 0;

    /**
     * \~japanese-en 複数の量子ビットを測定した時の周辺確率を計算する
     * 
     * @param measured_values 量子ビット数と同じ長さの0,1,2の配列。0,1はその値が観測され、2は測定をしないことを表す。
     * @return 計算された周辺確率
     */
    virtual double get_marginal_probability(std::vector<UINT> measured_values) const = 0;

    /**
     * \~japanese-en 計算基底で測定した時得られる確率分布のエントロピーを計算する。
     * 
     * @return エントロピー
     */
    virtual double get_entropy() const = 0;

    /**
     * \~japanese-en 量子状態のノルムを計算する
     * 
     * 量子状態のノルムは非ユニタリなゲートを作用した時に小さくなる。
     * @return ノルム
     */
    virtual double get_norm() const = 0;

    /**
     * \~japanese-en 量子状態を正規化する
     *
     * @param norm 自身のノルム
     */
    virtual void normalize(double norm) = 0;

    /**
     * \~japanese-en バッファとして同じサイズの量子状態を作成する。
     * 
     * @return 生成された量子状態
     */
    virtual QuantumStateBase* allocate_buffer() const = 0;

    /**
     * \~japanese-en 自身の状態のディープコピーを生成する
     * 
     * @return 自身のディープコピー
     */
    virtual QuantumStateBase* copy() const = 0;

    /**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
    virtual void load(const QuantumStateBase* state) = 0;

    /**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
    virtual void load(const std::vector<CPPCTYPE>& state) = 0;

	/**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
	virtual void load(const CPPCTYPE* state) = 0;

    /**
     * \~japanese-en 量子状態が配置されているメモリを保持するデバイス名を取得する。
     */
    virtual const char* get_device_name() const = 0;

    /**
     * \~japanese-en 量子状態をC++の<code>std::complex\<double\></code>の配列として取得する
     * 
     * @return 複素ベクトルのポインタ
     */
    virtual CPPCTYPE* data_cpp() const = 0;

    /**
     * \~japanese-en 量子状態をcsimのComplex型の配列として取得する
     * 
     * @return 複素ベクトルのポインタ
     */
    virtual CTYPE* data_c() const = 0;

    /**
     * \~japanese-en 指定した添え字の古典レジスタの値を取得する。
     * 
     * @param index セットするレジスタの添え字
     * @return 複素ベクトルのポインタ
     */
    virtual UINT get_classical_value(UINT index) {
        if(_classical_register.size() <= index) {
            _classical_register.resize(index+1,0);
        }
        return _classical_register[index];
    }

    /**
     * \~japanese-en 指定した添え字の古典レジスタに値をセットする
     * 
     * @param index セットするレジスタの添え字
     * @param val セットする値
     * @return 複素ベクトルのポインタ
     */
    virtual void set_classical_value(UINT index, UINT val) {
        if(_classical_register.size() <= index) {
            _classical_register.resize(index+1,0);
        }
        _classical_register[index] = val;
    }

    /**
     * \~japanese-en 古典レジスタのベクトルを返す
     *
     * @return 古典レジスタ
     */
    virtual const std::vector<UINT> get_classical_register() const {
        return _classical_register;
    }

    /**
     * \~japanese-en 量子状態を測定した際の計算基底のサンプリングを行う
     *
     * @param[in] sampling_count サンプリングを行う回数
     * @return サンプルされた値のリスト
     */
    virtual std::vector<ITYPE> sampling(UINT sampling_count) = 0;
    

    /**
     * \~japanese-en 量子回路のデバッグ情報の文字列を生成する
     *
     * @return 生成した文字列
     */
    virtual std::string to_string() const {
        std::stringstream os;
        ComplexVector eigen_state(this->dim);
        auto data = this->data_cpp();
        for (UINT i = 0; i < this->dim; ++i) eigen_state[i] = data[i];
        os << " *** Quantum State ***" << std::endl;
        os << " * Qubit Count : " << this->qubit_count << std::endl;
        os << " * Dimension   : " << this->dim << std::endl;
        os << " * State vector : \n" << eigen_state << std::endl;
        return os.str();
    }
    
    /**
     * \~japanese-en 量子状態のデバッグ情報を出力する。
     * 
     * @return 受け取ったストリーム
     */
    friend std::ostream& operator<<(std::ostream& os, const QuantumStateBase& state) {
        os << state.to_string();
        return os;
    }

    /**
     * \~japanese-en 量子状態のデバッグ情報を出力する。
     * 
     * @return 受け取ったストリーム
     */
    friend std::ostream& operator<<(std::ostream& os, const QuantumStateBase* state) {
        os << *state;
        return os;
    }
};

class QuantumStateCpu : public QuantumStateBase{
private:
    CPPCTYPE* _state_vector;
    Random random;
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param qubit_count_ 量子ビット数
     */
    QuantumStateCpu(UINT qubit_count_) : QuantumStateBase(qubit_count_){
        this->_state_vector = reinterpret_cast<CPPCTYPE*>(allocate_quantum_state(this->_dim));
        initialize_quantum_state(this->data_c(), _dim);
    }
    /**
     * \~japanese-en デストラクタ
     */
    virtual ~QuantumStateCpu(){
        release_quantum_state(this->data_c());
    }
    /**
     * \~japanese-en 量子状態を計算基底の0状態に初期化する
     */
    virtual void set_zero_state() override{
        initialize_quantum_state(this->data_c(), _dim);
    }
    /**
     * \~japanese-en 量子状態を<code>comp_basis</code>の基底状態に初期化する
     * 
     * @param comp_basis 初期化する基底を表す整数
     */
    virtual void set_computational_basis(ITYPE comp_basis)  override {
        set_zero_state();
        _state_vector[0] = 0.;
        _state_vector[comp_basis] = 1.;
    }
    /**
     * \~japanese-en 量子状態をHaar randomにサンプリングされた量子状態に初期化する
     */
    virtual void set_Haar_random_state() override{
        initialize_Haar_random_state_with_seed(this->data_c(), _dim, random.int32());
    }
    /**
     * \~japanese-en 量子状態をシードを用いてHaar randomにサンプリングされた量子状態に初期化する
     */
    virtual void set_Haar_random_state(UINT seed) override {
        initialize_Haar_random_state_with_seed(this->data_c(), _dim,seed);
    }
    /**
     * \~japanese-en <code>target_qubit_index</code>の添え字の量子ビットを測定した時、0が観測される確率を計算する。
     * 
     * 量子状態は変更しない。
     * @param target_qubit_index 
     * @return double 
     */
    virtual double get_zero_probability(UINT target_qubit_index) const override {
        return M0_prob(target_qubit_index, this->data_c(), _dim);
    }
    /**
     * \~japanese-en 複数の量子ビットを測定した時の周辺確率を計算する
     * 
     * @param measured_values 量子ビット数と同じ長さの0,1,2の配列。0,1はその値が観測され、2は測定をしないことを表す。
     * @return 計算された周辺確率
     */
    virtual double get_marginal_probability(std::vector<UINT> measured_values) const override {
        std::vector<UINT> target_index;
        std::vector<UINT> target_value;
        for (UINT i = 0; i < measured_values.size(); ++i) {
            if (i == 0 || i == 1) {
                target_index.push_back(i);
                target_value.push_back(measured_values[i]);
            }
        }
        return marginal_prob(target_index.data(), target_value.data(), (UINT)target_index.size(), this->data_c(), _dim);
    }
    /**
     * \~japanese-en 計算基底で測定した時得られる確率分布のエントロピーを計算する。
     * 
     * @return エントロピー
     */
    virtual double get_entropy() const override{
        return measurement_distribution_entropy(this->data_c(), _dim);
    }

    /**
     * \~japanese-en 量子状態のノルムを計算する
     * 
     * 量子状態のノルムは非ユニタリなゲートを作用した時に小さくなる。
     * @return ノルム
     */
    virtual double get_norm() const override {
        return state_norm(this->data_c(),_dim);
    }

    /**
     * \~japanese-en 量子状態を正規化する
     *
     * @param norm 自身のノルム
     */
    virtual void normalize(double norm) override{
        ::normalize(norm, this->data_c(), _dim);
    }


    /**
     * \~japanese-en バッファとして同じサイズの量子状態を作成する。
     * 
     * @return 生成された量子状態
     */
    virtual QuantumStateBase* allocate_buffer() const override {
        QuantumStateCpu* new_state = new QuantumStateCpu(this->_qubit_count);
        return new_state;
    }
    /**
     * \~japanese-en 自身の状態のディープコピーを生成する
     * 
     * @return 自身のディープコピー
     */
    virtual QuantumStateBase* copy() const override {
        QuantumStateCpu* new_state = new QuantumStateCpu(this->_qubit_count);
        memcpy(new_state->data_cpp(), _state_vector, (size_t)(sizeof(CPPCTYPE)*_dim));
        for(UINT i=0;i<_classical_register.size();++i) new_state->set_classical_value(i,_classical_register[i]);
        return new_state;
    }
    /**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
    virtual void load(const QuantumStateBase* _state) {
        this->_classical_register = _state->classical_register;
        memcpy(this->data_cpp(), _state->data_cpp(), (size_t)(sizeof(CPPCTYPE)*_dim));
    }
	/**
	 * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
	 */
	virtual void load(const std::vector<CPPCTYPE>& _state) {
		assert(_state.size() == _dim);
		memcpy(this->data_cpp(), _state.data(), (size_t)(sizeof(CPPCTYPE)*_dim));
	}

	/**
	 * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
	 */
	virtual void load(const CPPCTYPE* _state) {
		memcpy(this->data_cpp(), _state, (size_t)(sizeof(CPPCTYPE)*_dim));
	}
	
	/**
     * \~japanese-en 量子状態が配置されているメモリを保持するデバイス名を取得する。
     */
    virtual const char* get_device_name() const override {return "cpu";}
    /**
     * \~japanese-en 量子状態をC++の<code>std::complex\<double\></code>の配列として取得する
     * 
     * @return 複素ベクトルのポインタ
     */
    virtual CPPCTYPE* data_cpp() const override { return this->_state_vector; }
    /**
     * \~japanese-en 量子状態をcsimのComplex型の配列として取得する
     * 
     * @return 複素ベクトルのポインタ
     */
    virtual CTYPE* data_c() const override {
        return reinterpret_cast<CTYPE*>(this->_state_vector);
    }

    /**
     * \~japanese-en 量子状態を測定した際の計算基底のサンプリングを行う
     *
     * @param[in] sampling_count サンプリングを行う回数
     * @return サンプルされた値のリスト
     */
    virtual std::vector<ITYPE> sampling(UINT sampling_count) override{
        std::vector<double> stacked_prob;
        std::vector<ITYPE> result;
        double sum = 0.;
        auto ptr = this->data_cpp();
        stacked_prob.push_back(0.);
        for (UINT i = 0; i < this->dim; ++i) {
            sum += norm(ptr[i]);
            stacked_prob.push_back(sum);
        }

        for (UINT count = 0; count < sampling_count; ++count) {
            double r = random.uniform();
            auto ite = std::lower_bound(stacked_prob.begin(), stacked_prob.end(), r);
            auto index = std::distance(stacked_prob.begin(), ite) - 1;
            result.push_back(index);
        }
        return result;
    }

};

typedef QuantumStateCpu QuantumState; /**< QuantumState is an alias of QuantumStateCPU */

namespace state {
    /**
     * \~japanese-en 量子状態間の内積を計算する
     *
     * @param[in] state_bra 内積のブラ側の量子状態
     * @param[in] state_ket 内積のケット側の量子状態
     * @return 内積の値
     */
    CPPCTYPE DllExport inner_product(const QuantumState* state_bra, const QuantumState* state_ket);
}
