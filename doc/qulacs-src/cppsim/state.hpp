
#pragma once

#ifndef _MSC_VER
extern "C"{
#include <csim/memory_ops.h>
#include <csim/stat_ops.h>
#include <csim/update_ops.h>
#include <csim/init_ops.h>
}
#else
#include <csim/memory_ops.h>
#include <csim/stat_ops.h>
#include <csim/update_ops.h>
#include <csim/init_ops.h>
#endif

#include "type.hpp"
#include "utility.hpp"
#include <vector>
#include <iostream>

/**
 * \~japanese-en 量子状態の基底クラス
 */
/**
 * \~english Quantum state basis class
 */
class QuantumStateBase{
protected:
    ITYPE _dim;
    UINT _qubit_count;
	bool _is_state_vector;
    std::vector<UINT> _classical_register;
    UINT _device_number;
	void* _cuda_stream;
public:
    const UINT& qubit_count; /**< \~japanese-en 量子ビット数 */
	/**< \~english Qubit number */
    const UINT& qubit_count; /**< \~japanese-en 量子ビット数 */
	/**< \~english Qubit number */
    const ITYPE& dim; /**< \~japanese-en 量子状態の次元 */
	/**< \~english Dimension of quantum state */
    const std::vector<UINT>& classical_register; /**< \~japanese-en 古典ビットのレジスタ */
	/**< \~english Register of classic bit */
    const UINT& device_number;
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param qubit_count_ 量子ビット数
     */
    /**
     * \~english Constructor
     * 
     * @param qubit_count_ Qubit number
     */
    QuantumStateBase(UINT qubit_count_, bool is_state_vector):
        qubit_count(_qubit_count), dim(_dim), classical_register(_classical_register), device_number(_device_number)
    {
        this->_qubit_count = qubit_count_;
        this->_dim = 1ULL << qubit_count_;
		this->_is_state_vector = is_state_vector;
        this->_device_number=0;
    }
    QuantumStateBase(UINT qubit_count_, bool is_state_vector, UINT device_number_):
        qubit_count(_qubit_count), dim(_dim), classical_register(_classical_register), device_number(_device_number)
    {
        this->_qubit_count = qubit_count_;
        this->_dim = 1ULL << qubit_count_;
		this->_is_state_vector = is_state_vector;
        this->_device_number = device_number_;
    }
    /**
     * \~japanese-en デストラクタ
     */
    /**
     * \~english Destructor
     */
    virtual ~QuantumStateBase(){}

	/**
	 * \~japanese-en 量子状態が状態ベクトルか密度行列かを判定する
	 */
	/**
	 * \~english Determine if quantum state is state vector or density matrix
	 */
	virtual bool is_state_vector() const {
		return this->_is_state_vector;
	}

    /**
     * \~japanese-en 量子状態を計算基底の0状態に初期化する
     */
    /**
     * \~english Initialize the quantum state to the 0 state of the computational basis
     */
    virtual void set_zero_state() = 0;

    /**
     * \~japanese-en 量子状態を<code>comp_basis</code>の基底状態に初期化する
     * 
     * @param comp_basis 初期化する基底を表す整数
     */
    /**
     * \~english Initialize the quantum state to basis state of <code>comp_basis</code>
     * 
     * @param comp_basis An integer indicating the initializing basis
     */
    virtual void set_computational_basis(ITYPE comp_basis) = 0;

    /**
     * \~japanese-en 量子状態をHaar randomにサンプリングされた量子状態に初期化する
     */
     /**
     * \~english Initialize the quantum state to Haar random sampled quantum states
     */
    virtual void set_Haar_random_state() = 0;

    /**
     * \~japanese-en 量子状態をシードを用いてHaar randomにサンプリングされた量子状態に初期化する
     */
     /**
     * \~english Initialize the quantum state to Haar random sampled quantum states using seed
     */
    virtual void set_Haar_random_state(UINT seed) = 0;

    /**
     * \~japanese-en <code>target_qubit_index</code>の添え字の量子ビットを測定した時、0が観測される確率を計算する。
     * 
     * 量子状態は変更しない。
     * @param target_qubit_index 
     * @return double 
     */
    /**
     * \~english Calculate the probability of observing 0 when measuring the qubit at the index of <code>target_qubit_index</code>.
     * 
     * The quantum state does not change.
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
    /**
     * \~english Calculate marginal probabilities when measuring multiple qubits
     * 
     * @param measured_values An array of 0,1,2 with the same length as the number of qubits. 0,1 means that the value is observed, and 2 means no measurement.
     * @return Calculated marginal probabilities
     */
    virtual double get_marginal_probability(std::vector<UINT> measured_values) const = 0;

    /**
     * \~japanese-en 計算基底で測定した時得られる確率分布のエントロピーを計算する。
     * 
     * @return エントロピー
     */
    /**
     * \~english Calculate the entropy of the probability distribution obtained when measuring on the calculation basis.
     * 
     * @return Entropy
     */
    virtual double get_entropy() const = 0;

    /**
     * \~japanese-en 量子状態のノルムを計算する
     * 
     * 量子状態のノルムは非ユニタリなゲートを作用した時に小さくなる。
     * @return ノルム
     */
    /**
     * \~english Calculate norm of quantum state
     * 
     * The norm of the quantum state becomes smaller when a non-unitary gate is applied.
     * @return Norm
     */
    virtual double get_squared_norm() const = 0;

    /**
     * \~japanese-en 量子状態を正規化する
     *
     * @param norm 自身のノルム
     */    
    /**
     * \~english Normalize quantum states
     *
     * @param norm Norm of itself
     */
    virtual void normalize(double squared_norm) = 0;

    /**
     * \~japanese-en バッファとして同じサイズの量子状態を作成する。
     * 
     * @return 生成された量子状態
     */
    /**
     * \~english Create a quantum state of the same size as a buffer.
     * 
     * @return Created quantum state
     */
    virtual QuantumStateBase* allocate_buffer() const = 0;

    /**
     * \~japanese-en 自身の状態のディープコピーを生成する
     * 
     * @return 自身のディープコピー
     */
    /**
     * \~english Generate a deep copy of itself
     * 
     * @return Deep copy of itself
     */
    virtual QuantumStateBase* copy() const = 0;

    /**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
    /**
     * \~english Copy quantum state <code>state</code> to itself
     */
    virtual void load(const QuantumStateBase* state) = 0;

    /**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
    /**
     * \~english Copy quantum state <code>state</code> to itself
     */
    virtual void load(const std::vector<CPPCTYPE>& state) = 0;

    /**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
    /**
     * \~english Copy quantum state <code>state</code> to itself
     */
    virtual void load(const CPPCTYPE* state) = 0;

    /**
     * \~japanese-en 量子状態が配置されているメモリを保持するデバイス名を取得する。
     */
    /**
     * \~english Obtain the name of the device that holds the memory where the quantum state is located.
     */
    virtual const std::string get_device_name() const = 0;

    virtual void* data() const = 0;

    /**
     * \~japanese-en 量子状態をC++の<code>std::complex\<double\></code>の配列として取得する
     * 
     * @return 複素ベクトルのポインタ
     */
    /**
     * \~english Obtain quantum state as an array of <code>std::complex\<double\></code> of C++
     * 
     * @return Pointer of complex vector
     */
    virtual CPPCTYPE* data_cpp() const = 0;

    /**
     * \~japanese-en 量子状態をcsimのComplex型の配列として取得する
     * 
     * @return 複素ベクトルのポインタ
     */
    /**
     * \~english Obtain quantum state as an array of complex type of csim
     * 
     * @return Pointer of complex vector
     */
    virtual CTYPE* data_c() const = 0;

	/**
	 * \~japanese-en 量子状態をC++の<code>std::complex\<double\></code>の配列として新たに確保する
	 *
	 * @return 複素ベクトルのポインタ
	 */
       /**
         * \~english Secure quantum state as a new array of <code>std::complex\<double\></code> of C++
         * 
         * @return Pointer of complex vector
         */
	virtual CPPCTYPE* duplicate_data_cpp() const = 0;

	/**
	 * \~japanese-en 量子状態をcsimのComplex型の配列として新たに確保する。
	 *
	 * @return 複素ベクトルのポインタ
	 */
        /**
         * \~english Secure quantum state as a new array of complex type of csim
         * 
         * @return Pointer of complex vector
         */
	virtual CTYPE* duplicate_data_c() const = 0;

    /**
     * \~japanese-en 量子状態を足しこむ
     */
    /**
     * \~english Add quantum state
     */
    virtual void add_state(const QuantumStateBase* state) = 0;
    /**
     * \~japanese-en 複素数をかける
     */
    /**
     * \~english Multiply complex number
     */
    virtual void multiply_coef(CPPCTYPE coef) = 0;
    
    virtual void multiply_elementwise_function(const std::function<CPPCTYPE(ITYPE)> &func) = 0;
    
    /**
     * \~japanese-en 指定した添え字の古典レジスタの値を取得する。
     * 
     * @param index セットするレジスタの添え字
     * @return 複素ベクトルのポインタ
     */
    /**
     * \~english Obtain the value of the classical register with the specified index.
     * 
     * @param index Index of register to be set
     * @return Ponter of complex vector
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
    /**
     * \~english Set the value of the classical register with the specified index.
     * 
     * @param index Index of register to be set
     * @param val Value to be set
     * @return Ponter of complex vector
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
    /**
     * \~english Return vectore of classic register
     *
     * @return Classic register
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
    /**
     * \~english Sampling the computational basis when measuring the quantum state
     *
     * @param[in] sampling_count Number of times sampling is performed
     * @return List of sampled values
     */
    virtual std::vector<ITYPE> sampling(UINT sampling_count) = 0;
    

    /**
     * \~japanese-en 量子回路のデバッグ情報の文字列を生成する
     *
     * @return 生成した文字列
     */
    /**
     * \~english Generate a string of debug information for a quantum circuit
     *
     * @return Generated string
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
    /**
     * \~english Output quantum state debug information.
     * 
     * @return String received
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
    /**
     * \~english Output quantum state debug information.
     * 
     * @return String received
     */
    friend std::ostream& operator<<(std::ostream& os, const QuantumStateBase* state) {
        os << *state;
        return os;
    }

	virtual void* get_cuda_stream() const {
		return this->_cuda_stream;
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
    /**
     * \~english Constructor
     * 
     * @param qubit_count_ Qubit number
     */
    QuantumStateCpu(UINT qubit_count_) : QuantumStateBase(qubit_count_, true){
        this->_state_vector = reinterpret_cast<CPPCTYPE*>(allocate_quantum_state(this->_dim));
        initialize_quantum_state(this->data_c(), _dim);
    }
    /**
     * \~japanese-en デストラクタ
     */
    /**
     * \~english Destructor
     */
    virtual ~QuantumStateCpu(){
        release_quantum_state(this->data_c());
    }
    /**
     * \~japanese-en 量子状態を計算基底の0状態に初期化する
     */
    /**
     * \~english Initialize the quantum state to the 0 state of the computational basis
     */
    virtual void set_zero_state() override{
        initialize_quantum_state(this->data_c(), _dim);
    }
    /**
     * \~japanese-en 量子状態を<code>comp_basis</code>の基底状態に初期化する
     * 
     * @param comp_basis 初期化する基底を表す整数
     */
    /**
     * \~english Initialize quantum state to basis state of <code>comp_basis</code>
     * 
     * @param comp_basis An integer indicating the initializing basis
     */
    virtual void set_computational_basis(ITYPE comp_basis)  override {
        if (comp_basis >= (ITYPE)(1ULL << this->qubit_count)) {
            std::cerr << "Error: QuantumStateCpu::set_computational_basis(ITYPE): index of computational basis must be smaller than 2^qubit_count" << std::endl;
            return;
        }
        set_zero_state();
        _state_vector[0] = 0.;
        _state_vector[comp_basis] = 1.;
    }
    /**
     * \~japanese-en 量子状態をHaar randomにサンプリングされた量子状態に初期化する
     */
     /**
     * \~english Initialize the quantum state to Haar random sampled quantum states
     */
    virtual void set_Haar_random_state() override{
        initialize_Haar_random_state_with_seed(this->data_c(), _dim, random.int32());
    }
    /**
     * \~japanese-en 量子状態をシードを用いてHaar randomにサンプリングされた量子状態に初期化する
     */
     /**
     * \~english Initialize the quantum state to Haar random sampled quantum states using seed
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
    /**
     * \~english Calculate the probability of observing 0 when measuring the qubit at the index of <code>target_qubit_index</code>.
     * 
     * The quantum state does not change.
     * @param target_qubit_index 
     * @return double 
     */
    virtual double get_zero_probability(UINT target_qubit_index) const override {
        if (target_qubit_index >= this->qubit_count) {
            std::cerr << "Error: QuantumStateCpu::get_zero_probability(UINT): index of target qubit must be smaller than qubit_count" << std::endl;
            return 0.;
        }
        return M0_prob(target_qubit_index, this->data_c(), _dim);
    }
    /**
     * \~japanese-en 複数の量子ビットを測定した時の周辺確率を計算する
     * 
     * @param measured_values 量子ビット数と同じ長さの0,1,2の配列。0,1はその値が観測され、2は測定をしないことを表す。
     * @return 計算された周辺確率
     */
    /**
     * \~english Calculate marginal probabilities when measuring multiple qubits
     * 
     * @param measured_values An array of 0,1,2 with the same length as the number of qubits. 0,1 means that the value is observed, and 2 means no measurement.
     * @return Calculated marginal probabilities
     */
    virtual double get_marginal_probability(std::vector<UINT> measured_values) const override {
        if (measured_values.size() != this->qubit_count) {
            std::cerr << "Error: QuantumStateCpu::get_marginal_probability(vector<UINT>): the length of measured_values must be equal to qubit_count" << std::endl;
            return 0.;
        }
        
        std::vector<UINT> target_index;
        std::vector<UINT> target_value;
        for (UINT i = 0; i < measured_values.size(); ++i) {
            UINT measured_value = measured_values[i];
            if (measured_value== 0 || measured_value == 1) {
                target_index.push_back(i);
                target_value.push_back(measured_value);
            }
        }
        return marginal_prob(target_index.data(), target_value.data(), (UINT)target_index.size(), this->data_c(), _dim);
    }
    /**
     * \~japanese-en 計算基底で測定した時得られる確率分布のエントロピーを計算する。
     * 
     * @return エントロピー
     */
    /**
     * \~english Calculate the entropy of the probability distribution obtained when measuring on the calculation basis.
     * 
     * @return Entropy
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
    /**
     * \~english Calculate norm of quantum state
     * 
     * The norm of the quantum state becomes smaller when a non-unitary gate is applied.
     * @return Norm
     */
    virtual double get_squared_norm() const override {
        return state_norm_squared(this->data_c(),_dim);
    }

    /**
     * \~japanese-en 量子状態を正規化する
     *
     * @param norm 自身のノルム
     */
    /**
     * \~english Normalize quantum states
     *
     * @param norm Norm of itself
     */
    virtual void normalize(double squared_norm) override{
        ::normalize(squared_norm, this->data_c(), _dim);
    }


    /**
     * \~japanese-en バッファとして同じサイズの量子状態を作成する。
     * 
     * @return 生成された量子状態
     */
    /**
     * \~english Create a quantum state of the same size as a buffer.
     * 
     * @return Created quantum state
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
    /**
     * \~english Generate a deep copy of itself
     * 
     * @return Deep copy of itself
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
    /**
     * \~english Copy quantum state <code>state</code> to itself
     */
    virtual void load(const QuantumStateBase* _state) {
        if (_state->qubit_count != this->qubit_count) {
            std::cerr << "Error: QuantumStateCpu::load(const QuantumStateBase*): invalid qubit count" << std::endl;
            return;
        }
        
        this->_classical_register = _state->classical_register;
		if (_state->get_device_name() == "gpu") {
			auto ptr = _state->duplicate_data_cpp();
			memcpy(this->data_cpp(), ptr, (size_t)(sizeof(CPPCTYPE)*_dim));
			free(ptr);
		}
		else {
			memcpy(this->data_cpp(), _state->data_cpp(), (size_t)(sizeof(CPPCTYPE)*_dim));
		}
    }
    /**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
    /**
     * \~english Copy quantum state <code>state</code> to itself
     */
    virtual void load(const std::vector<CPPCTYPE>& _state) {
        if (_state.size() != _dim) {
            std::cerr << "Error: QuantumStateCpu::load(vector<Complex>&): invalid length of state" << std::endl;
            return;
        }
        memcpy(this->data_cpp(), _state.data(), (size_t)(sizeof(CPPCTYPE)*_dim));
    }

    /**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
    /**
     * \~english Copy quantum state <code>state</code> to itself
     */
    virtual void load(const CPPCTYPE* _state) {
        memcpy(this->data_cpp(), _state, (size_t)(sizeof(CPPCTYPE)*_dim));
    }
    
    /**
     * \~japanese-en 量子状態が配置されているメモリを保持するデバイス名を取得する。
     */
    /**
     * \~english Obtain the name of the device that holds the memory where the quantum state is located.
     */
    virtual const std::string get_device_name() const override {return "cpu";}

    /**
     * \~japanese-en 量子状態のポインタをvoid*型として返す
     */
    /**
     * \~english Return the pointer of quantum state as void* type
     */
    virtual void* data() const override {
        return reinterpret_cast<void*>(this->_state_vector);
    }
    /**
     * \~japanese-en 量子状態をC++の<code>std::complex\<double\></code>の配列として取得する
     * 
     * @return 複素ベクトルのポインタ
     */
    /**
     * \~english Obtain quantum state as an array of <code>std::complex\<double\></code> of C++
     * 
     * @return Pointer of complex vector
     */

    virtual CPPCTYPE* data_cpp() const override { return this->_state_vector; }
    /**
     * \~japanese-en 量子状態をcsimのComplex型の配列として取得する
     * 
     * @return 複素ベクトルのポインタ
     */
    /**
     * \~english Obtain quantum state as an array of complex type of csim
     * 
     * @return Pointer of complex vector
     */

    virtual CTYPE* data_c() const override {
        return reinterpret_cast<CTYPE*>(this->_state_vector);
    }

	virtual CTYPE* duplicate_data_c() const override {
		CTYPE* new_data = (CTYPE*)malloc(sizeof(CTYPE)*_dim);
		memcpy(new_data, this->data(), (size_t)(sizeof(CTYPE)*_dim));
		return new_data;
	}

	virtual CPPCTYPE* duplicate_data_cpp() const override {
		CPPCTYPE* new_data = (CPPCTYPE*)malloc(sizeof(CPPCTYPE)*_dim);
		memcpy(new_data, this->data(), (size_t)(sizeof(CPPCTYPE)*_dim));
		return new_data;
	}



    /**
     * \~japanese-en 量子状態を足しこむ
     */
    /**
     * \~english Add quantum state
     */
    virtual void add_state(const QuantumStateBase* state) override{
		if (state->get_device_name() == "gpu") {
			std::cerr << "State vector on GPU cannot be added to that on CPU" << std::endl;
			return;
		}
        state_add(state->data_c(), this->data_c(), this->dim);
    }
    /**
     * \~japanese-en 複素数をかける
     */
    /**
     * \~english Multiply complex number
     */
    virtual void multiply_coef(CPPCTYPE coef) override{
#ifdef _MSC_VER
        state_multiply(coef, this->data_c(), this->dim);
#else
        CTYPE c_coef = {coef.real(), coef.imag()};
        state_multiply(c_coef, this->data_c(), this->dim);
#endif
    }

    virtual void multiply_elementwise_function(const std::function<CPPCTYPE(ITYPE)> &func) override{
		CPPCTYPE* state = this->data_cpp();
		for (ITYPE idx = 0; idx < dim; ++idx) {
			state[idx] *= (CPPCTYPE)func(idx);
		}
	}

    /**
     * \~japanese-en 量子状態を測定した際の計算基底のサンプリングを行う
     *
     * @param[in] sampling_count サンプリングを行う回数
     * @return サンプルされた値のリスト
     */
    /**
     * \~english Sampling the computational basis when measuring the quantum state
     *
     * @param[in] sampling_count Number of times sampling is performed
     * @return List of sampled values
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
    /**
     * \~english Calculate inner product between quantum states
     *
     * @param[in] state_bra Bra side quantum state of inner product
     * @param[in] state_ket Ket side quantum state of inner product
     * @return Value of inner product
     */
    CPPCTYPE DllExport inner_product(const QuantumState* state_bra, const QuantumState* state_ket);

}
