
#pragma once

#ifndef _MSC_VER
extern "C" {
#include <csim/memory_ops_dm.h>
#include <csim/stat_ops_dm.h>
#include <csim/update_ops_dm.h>
}
#else
#include <csim/memory_ops_dm.h>
#include <csim/stat_ops_dm.h>
#include <csim/update_ops_dm.h>
#endif

#include "state.hpp"

class DensityMatrixCpu : public QuantumStateBase{
private:
    CPPCTYPE* _density_matrix;
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
    DensityMatrixCpu(UINT qubit_count_) : QuantumStateBase(qubit_count_, false){
        this->_density_matrix = reinterpret_cast<CPPCTYPE*>(dm_allocate_quantum_state(this->_dim));
        dm_initialize_quantum_state(this->data_c(), _dim);
    }
    /**
     * \~japanese-en デストラクタ
     */
    /**
     * \~english Destructor
     */
    virtual ~DensityMatrixCpu(){
        dm_release_quantum_state(this->data_c());
    }
    /**
     * \~japanese-en 量子状態を計算基底の0状態に初期化する
     */
    /**
     * \~english Initialize the quantum state to the 0 state of the computational basis
     */
    virtual void set_zero_state() override{
        dm_initialize_quantum_state(this->data_c(), _dim);
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
            std::cerr << "Error: DensityMatrixCpu::set_computational_basis(ITYPE): index of computational basis must be smaller than 2^qubit_count" << std::endl;
            return;
        }
        set_zero_state();
		_density_matrix[0] = 0.;
        _density_matrix[comp_basis*dim+comp_basis] = 1.;
    }
    /**
     * \~japanese-en 量子状態をHaar randomにサンプリングされた量子状態に初期化する
     */
     /**
     * \~english Initialize the quantum state to Haar random sampled quantum states
     */
    virtual void set_Haar_random_state() override{
		this->set_Haar_random_state(random.int32());
    }
    /**
     * \~japanese-en 量子状態をシードを用いてHaar randomにサンプリングされた量子状態に初期化する
     */
     /**
     * \~english Initialize the quantum state to Haar random sampled quantum states using seed
     */
    virtual void set_Haar_random_state(UINT seed) override {
		QuantumStateCpu* pure_state = new QuantumStateCpu(qubit_count);
		pure_state->set_Haar_random_state(seed);
		dm_initialize_with_pure_state(this->data_c(), pure_state->data_c(), _dim);
		delete pure_state;
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
            std::cerr << "Error: DensityMatrixCpu::get_zero_probability(UINT): index of target qubit must be smaller than qubit_count" << std::endl;
            return 0.;
        }
        return dm_M0_prob(target_qubit_index, this->data_c(), _dim);
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
            std::cerr << "Error: DensityMatrixCpu::get_marginal_probability(vector<UINT>): the length of measured_values must be equal to qubit_count" << std::endl;
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
        return dm_marginal_prob(target_index.data(), target_value.data(), (UINT)target_index.size(), this->data_c(), _dim);
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
        return dm_measurement_distribution_entropy(this->data_c(), _dim);
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
        return dm_state_norm_squared(this->data_c(),_dim);
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
        dm_normalize(squared_norm, this->data_c(), _dim);
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
        DensityMatrixCpu* new_state = new DensityMatrixCpu(this->_qubit_count);
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
        DensityMatrixCpu* new_state = new DensityMatrixCpu(this->_qubit_count);
        memcpy(new_state->data_cpp(), _density_matrix, (size_t)(sizeof(CPPCTYPE)*_dim*_dim));
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
            std::cerr << "Error: DensityMatrixCpu::load(const QuantumStateBase*): invalid qubit count" << std::endl;
            return;
        }
		if (_state->is_state_vector()) {
			if (_state->get_device_name() == "gpu") {
				auto ptr = _state->duplicate_data_c();
				dm_initialize_with_pure_state(this->data_c(), ptr, dim);
				free(ptr);
			}
			else {
				dm_initialize_with_pure_state(this->data_c(), _state->data_c(), dim);
			}
		}else {
			memcpy(this->data_cpp(), _state->data_cpp(), (size_t)(sizeof(CPPCTYPE)*_dim*_dim));
		}
		this->_classical_register = _state->classical_register;
	}
    /**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
    /**
     * \~english Copy quantum state <code>state</code> to itself
     */
    virtual void load(const std::vector<CPPCTYPE>& _state) {
        if (_state.size() != _dim && _state.size() != _dim*_dim) {
            std::cerr << "Error: DensityMatrixCpu::load(vector<Complex>&): invalid length of state" << std::endl;
            return;
        }
		if (_state.size() == _dim) {
			dm_initialize_with_pure_state(this->data_c(), (const CTYPE*)_state.data(), dim);
		}
		else {
			memcpy(this->data_cpp(), _state.data(), (size_t)(sizeof(CPPCTYPE)*_dim*_dim));
		}
    }

	virtual void load(const Eigen::VectorXcd& _state) {
		if (_state.size() != _dim && _state.size() != _dim * _dim) {
			std::cerr << "Error: DensityMatrixCpu::load(vector<Complex>&): invalid length of state" << std::endl;
			return;
		}
		if (_state.size() == _dim) {
			dm_initialize_with_pure_state(this->data_c(), (const CTYPE*)_state.data(), dim);
		}
		else {
			memcpy(this->data_cpp(), _state.data(), (size_t)(sizeof(CPPCTYPE)*_dim*_dim));
		}
	}

	virtual void load(const ComplexMatrix& _state) {
		if (_state.cols() != _dim && _state.rows() != _dim * _dim) {
			std::cerr << "Error: DensityMatrixCpu::load(ComplexMatrix&): invalid length of state" << std::endl;
			return;
		}
		memcpy(this->data_cpp(), _state.data(), (size_t)(sizeof(CPPCTYPE)*_dim*_dim));
	}

    /**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
    /**
     * \~english Copy quantum state <code>state</code> to itself
     */
    virtual void load(const CPPCTYPE* _state) {
        memcpy(this->data_cpp(), _state, (size_t)(sizeof(CPPCTYPE)*_dim*_dim));
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
        return reinterpret_cast<void*>(this->_density_matrix);
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
    virtual CPPCTYPE* data_cpp() const override { return this->_density_matrix; }
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
        return reinterpret_cast<CTYPE*>(this->_density_matrix);
    }

	virtual CTYPE* duplicate_data_c() const override {
		CTYPE* new_data = (CTYPE*)malloc(sizeof(CTYPE)*_dim*_dim);
		memcpy(new_data, this->data(), (size_t)(sizeof(CTYPE)*_dim*_dim));
		return new_data;
	}

	virtual CPPCTYPE* duplicate_data_cpp() const override {
		CPPCTYPE* new_data = (CPPCTYPE*)malloc(sizeof(CPPCTYPE)*_dim*_dim);
		memcpy(new_data, this->data(), (size_t)(sizeof(CPPCTYPE)*_dim*_dim));
		return new_data;
	}


    /**
     * \~japanese-en 量子状態を足しこむ
     */
    /**
     * \~english Add quantum state
     */
	virtual void add_state(const QuantumStateBase* state) override {
		if (state->is_state_vector()) {
			std::cerr << "add state between density matrix and state vector is not implemented" << std::endl;
			return;
		}
		dm_state_add(state->data_c(), this->data_c(), this->dim);
	}
	/**
	 * \~japanese-en 複素数をかける
	 */
        /**
         * \~english Multiply complex number
         */
	virtual void multiply_coef(CPPCTYPE coef) override {
#ifdef _MSC_VER
		dm_state_multiply(coef, this->data_c(), this->dim);
#else
		CTYPE c_coef = { coef.real(), coef.imag() };
		dm_state_multiply(c_coef, this->data_c(), this->dim);
#endif
	}

    virtual void multiply_elementwise_function(const std::function<CPPCTYPE(ITYPE)> &func) override{
			std::cerr << "multiply_elementwise_function between density matrix and state vector is not implemented" << std::endl;
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
            sum += norm(ptr[i*dim+i]);
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

	virtual std::string to_string() const {
		std::stringstream os;
		ComplexMatrix eigen_state(this->dim, this->dim);
		auto data = this->data_cpp();
		for (UINT i = 0; i < this->dim; ++i) {
			for (UINT j = 0; j < this->dim; ++j) {
				eigen_state(i,j) = data[i*dim+j];
			}
		}
		os << " *** Density Matrix ***" << std::endl;
		os << " * Qubit Count : " << this->qubit_count << std::endl;
		os << " * Dimension   : " << this->dim << std::endl;
		os << " * Density matrix : \n" << eigen_state << std::endl;
		return os.str();
	}
};

typedef DensityMatrixCpu DensityMatrix; /**< QuantumState is an alias of QuantumStateCPU */

