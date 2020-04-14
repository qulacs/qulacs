#pragma once

#include "state.hpp"

#ifdef _USE_GPU

#include <gpusim/update_ops_cuda.h>
#include <gpusim/memory_ops.h>
#include <gpusim/stat_ops.h>
#include <gpusim/util_func.h>

class QuantumStateGpu : public QuantumStateBase {
private:
	void* _state_vector; // void* is assumed as GTYPE* 
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
	QuantumStateGpu(UINT qubit_count_) : QuantumStateBase(qubit_count_, true, 0) {
		set_device(0);
		this->_cuda_stream = allocate_cuda_stream_host(1, 0);
		this->_state_vector = reinterpret_cast<void*>(allocate_quantum_state_host(this->_dim, 0));
		initialize_quantum_state_host(this->data(), _dim, _cuda_stream, device_number);
	}

	QuantumStateGpu(UINT qubit_count_, UINT device_number_) : QuantumStateBase(qubit_count_, true, device_number_) {
		int num_device = get_num_device();
		assert(device_number_ < num_device);
		set_device(device_number_);
		this->_cuda_stream = allocate_cuda_stream_host(1, device_number_);
		this->_state_vector = reinterpret_cast<void*>(allocate_quantum_state_host(this->_dim, device_number_));
		initialize_quantum_state_host(this->data(), _dim, _cuda_stream, device_number_);
	}

	/**
	 * \~japanese-en デストラクタ
	 */
        /**
         * \~english Destructor
         */
	virtual ~QuantumStateGpu() {
		release_quantum_state_host(this->data(), device_number);
		release_cuda_stream_host(this->_cuda_stream, 1, device_number);
	}
	/**
	 * \~japanese-en 量子状態を計算基底の0状態に初期化する
	 */
        /**
         * \~english Initialize the quantum state to the 0 state of the computational basis
         */
	virtual void set_zero_state() override {
		initialize_quantum_state_host(this->data(), _dim, _cuda_stream, device_number);
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
		set_computational_basis_host(comp_basis, _state_vector, _dim, _cuda_stream, device_number);
	}
	/**
	 * \~japanese-en 量子状態をHaar randomにサンプリングされた量子状態に初期化する
	 */
        /**
         * \~english Initialize the quantum state to Haar random sampled quantum states
         */
	virtual void set_Haar_random_state() override {
		initialize_Haar_random_state_with_seed_host(this->data(), _dim, random.int32(), _cuda_stream, device_number);
	}
	/**
	 * \~japanese-en 量子状態をシードを用いてHaar randomにサンプリングされた量子状態に初期化する
	 */
        /**
         * \~english Initialize the quantum state to Haar random sampled quantum states using seed
         */
	virtual void set_Haar_random_state(UINT seed) override {
		initialize_Haar_random_state_with_seed_host(this->data(), _dim, seed, _cuda_stream, device_number);
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
		return M0_prob_host(target_qubit_index, this->data(), _dim, _cuda_stream, device_number);
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
		std::vector<UINT> target_index;
		std::vector<UINT> target_value;
		for (UINT i = 0; i < measured_values.size(); ++i) {
			UINT measured_value = measured_values[i];
			if (measured_value == 0 || measured_value == 1) {
				target_index.push_back(i);
				target_value.push_back(measured_value);
			}
		}
		return marginal_prob_host(target_index.data(), target_value.data(), (UINT)target_index.size(), this->data(), _dim, _cuda_stream, device_number);
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
	virtual double get_entropy() const override {
		return measurement_distribution_entropy_host(this->data(), _dim, _cuda_stream, device_number);
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
		return state_norm_squared_host(this->data(), _dim, _cuda_stream, device_number);
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
	virtual void normalize(double squared_norm) override {
		normalize_host(squared_norm, this->data(), _dim, _cuda_stream, device_number);
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
		QuantumStateGpu* new_state = new QuantumStateGpu(this->_qubit_count, device_number);
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
		QuantumStateGpu* new_state = new QuantumStateGpu(this->_qubit_count, device_number);
		copy_quantum_state_from_device_to_device(new_state->data(), _state_vector, _dim, _cuda_stream, device_number);
		return new_state;
	}
	/**
	 * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
	 */
        /**
         * \~english Copy quantum state <code>state</code> to itself
         */
	virtual void load(const QuantumStateBase* _state) override {
		if (_state->get_device_name() == "gpu") {
			copy_quantum_state_from_device_to_device(this->data(), _state->data(), dim, _cuda_stream, device_number);
		}
		else {
			this->load(_state->data_cpp());
		}
		this->_classical_register = _state->classical_register;
	}

	/**
	* \~japanese-en <code>state</code>の量子状態を自身へコピーする。
	*/
        /**
         * \~english Copy quantum state <code>state</code> to itself
         */
	virtual void load(const std::vector<CPPCTYPE>& _state) override {
		copy_quantum_state_from_cppstate_host(this->data(), _state.data(), dim, _cuda_stream, device_number);
	}

	/**
	* \~japanese-en <code>state</code>の量子状態を自身へコピーする。
	*/
        /**
         * \~english Copy quantum state <code>state</code> to itself
         */
	virtual void load(const CPPCTYPE* _state) override {
		copy_quantum_state_from_cppstate_host(this->data(), _state, dim, _cuda_stream, device_number);
	}
	/**
	 * \~japanese-en 量子状態が配置されているメモリを保持するデバイス名を取得する。
	 */
        /**
         * \~english Obtain the name of the device that holds the memory where the quantum state is located.
         */
	virtual const std::string get_device_name() const override { return "gpu"; }

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
	virtual CPPCTYPE* data_cpp() const override {
		std::cerr << "Cannot reinterpret state vector on GPU to cpp complex vector. Use duplicate_data_cpp instead." << std::endl;
		return NULL;
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
	virtual CTYPE* data_c() const override {
		std::cerr << "Cannot reinterpret state vector on GPU to C complex vector. Use duplicate_data_cpp instead." << std::endl;
		return NULL;
	}

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

	virtual void* data() const override {
		return reinterpret_cast<void*>(this->_state_vector);
	}

	virtual CTYPE* duplicate_data_c() const override {
		CTYPE* _copy_state = (CTYPE*)malloc(sizeof(CTYPE) * dim);
		get_quantum_state_host(this->_state_vector, _copy_state, dim, _cuda_stream, device_number);
		return _copy_state;
	}

	virtual CPPCTYPE* duplicate_data_cpp() const override {
		CPPCTYPE* _copy_state = (CPPCTYPE*)malloc(sizeof(CPPCTYPE) * dim);
		get_quantum_state_host(this->_state_vector, _copy_state, dim, _cuda_stream, device_number);
		return _copy_state;
	}

	/**
	 * \~japanese-en 量子状態を足しこむ
	 */
	/**
         * \~english Add quantum state
         */
	virtual void add_state(const QuantumStateBase* state) override {
		state_add_host(state->data(), this->data(), this->dim, _cuda_stream, device_number);
	}
	/**
	 * \~japanese-en 複素数をかける
	 */
        /**
         * \~english Multiply complex number
         */
	virtual void multiply_coef(CPPCTYPE coef) override {
		state_multiply_host(coef, this->data(), this->dim, _cuda_stream, device_number);
	}
    
    virtual void multiply_elementwise_function(const std::function<CPPCTYPE(ITYPE)> &func) override{
		std::vector<CPPCTYPE> diagonal_matrix(dim);
		for (ITYPE i = 0; i < dim; ++i) {
			diagonal_matrix[i] = func(i);
		}
		multi_qubit_diagonal_matrix_gate_host(diagonal_matrix.data(), this->data(), dim, _cuda_stream, device_number);
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
	virtual std::vector<ITYPE> sampling(UINT sampling_count) override {
		std::vector<double> stacked_prob;
		std::vector<ITYPE> result;
		double sum = 0.;
		auto ptr = this->duplicate_data_cpp();
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
		free(ptr);
		return result;
	}

	virtual std::string to_string() const {
		std::stringstream os;
		ComplexVector eigen_state(this->dim);
		auto data = this->duplicate_data_cpp();
		for (UINT i = 0; i < this->dim; ++i) eigen_state[i] = data[i];
		os << " *** Quantum State ***" << std::endl;
		os << " * Qubit Count : " << this->qubit_count << std::endl;
		os << " * Dimension   : " << this->dim << std::endl;
		os << " * State vector : \n" << eigen_state << std::endl;
		free(data);
		return os.str();
	}
};

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
	CPPCTYPE DllExport inner_product(const QuantumStateGpu* state_bra, const QuantumStateGpu* state_ket);
}

#endif // _USE_GPU
