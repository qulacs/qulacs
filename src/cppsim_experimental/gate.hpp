#pragma once

#include <map>
#include <set>
#include <stdexcept>
#include <string>

#ifndef _MSC_VER
extern "C" {
#endif
#include <csim/constant.h>
#include <csim/update_ops.h>
#include <csim/update_ops_dm.h>
#ifndef _MSC_VER
}
#endif

#include <csim/update_ops_cpp.hpp>

#ifdef _USE_GPU
#include <gpusim/update_ops_cuda.h>
#endif

#include "observable.hpp"
#include "state.hpp"
#include "type.hpp"

enum MapType {
    Basic,
    Sequence,
    Probabilistic,
    CPTP,
    Instrument,
};

enum GateMatrixType {
    DenseMatrix,
    DiagonalMatrix,
    SparseMatrix,
    PauliMatrix,
    PermutationMatrix
};

enum SpecialFuncType {
    None,
    GateI,
    GateX,
    GateY,
    GateZ,
    GateSqrtX,
    GateSqrtY,
    GateSqrtXdag,
    GateSqrtYdag,
    GateRX,
    GateRY,
    GateRZ,
    GateH,
    GateS,
    GateSdag,
    GateT,
    GateTdag,
    GateP0,
    GateP1,
    GateCX,
    GateCY,
    GateCZ,
    GateSWAP,
};

#define FLAG_CLIFFORD 0x01
#define FLAG_GAUSSIAN 0x02
#define FLAG_PARAMETRIC 0x04
#define FLAG_NOISE 0x8

#define FLAG_COMMUTE_X 0x01
#define FLAG_COMMUTE_Y 0x02
#define FLAG_COMMUTE_Z 0x04

class DllExport QuantumGateBase {
protected:
    MapType _map_type;
    std::map<std::string, double*> _parameter;
    QuantumGateBase(MapType map_type) : _map_type(map_type){};

public:
    QuantumGateBase(){};
    template <class Archive>
    void save(Archive& ar) const {
        
        std::vector<std::pair<std::string,double>> parameter_copy;
        for(auto x:_parameter){
            parameter_copy.push_back(std::pair<std::string,double>(x.first,*(x.second)));
        }
        ar(CEREAL_NVP(parameter_copy),CEREAL_NVP(_map_type));
        
    }

    template <class Archive>
    void load(Archive& ar) {
        std::vector<std::pair<std::string,double>> parameter_copy;
        ar(CEREAL_NVP(parameter_copy),CEREAL_NVP(_map_type));
        for(auto x:parameter_copy){
            (*(_parameter[x.first])) = x.second;
        }
        
    }
    virtual ~QuantumGateBase(){};
    virtual MapType get_map_type() const { return _map_type; }
    virtual UINT get_qubit_count() const = 0;
    virtual const std::vector<QuantumGateBase*>& get_kraus_list() const = 0;
    virtual const std::vector<UINT> get_qubit_index_list() const = 0;
    virtual const std::vector<UINT> get_target_index_list() const = 0;
    virtual const std::vector<UINT> get_control_index_list() const = 0;
    virtual void get_matrix(ComplexMatrix& matrix) const = 0;
    virtual std::vector<double> get_cumulative_distribution() const = 0;
    virtual void optimize_ProbabilisticGate() const {};
    virtual void reset_qubit_index_list(
        const std::vector<UINT>& src, const std::vector<UINT>& dst) = 0;
    virtual QuantumGateBase* copy() const = 0;
    virtual std::string to_string() const = 0;
    virtual void update_quantum_state(QuantumStateBase*) = 0;
    virtual UINT get_parameter_count() const { return (UINT)_parameter.size(); }
    virtual bool has_parameter(std::string name) const {
        return _parameter.count(name);
    }
    virtual double get_parameter(std::string name) const {
        return *(_parameter.at(name));
    }
    virtual void set_parameter(std::string name, double value) {
        (*(_parameter[name])) = value;
    }
    friend DllExport std::ostream& operator<<(
        std::ostream& os, const QuantumGateBase& gate) {
        os << gate.to_string();
        return os;
    }
};

class DllExport QuantumGateBasic : public QuantumGateBase {
public:
    GateMatrixType _matrix_type;
    SpecialFuncType _special_func_type;
    std::vector<UINT> _target_qubit_index;
    std::vector<UINT> _target_qubit_commutation;
    std::vector<UINT> _control_qubit_index;
    std::vector<UINT> _control_qubit_value;
    UINT _gate_property;

    ComplexMatrix _dense_matrix_element;
    ComplexVector _diagonal_matrix_element;
    SparseComplexMatrix _sparse_matrix_element;
    std::vector<UINT> _pauli_id;
    double _rotation_angle;
    // PermutationFunction* permutation_function;
    QuantumGateBasic(){};
    QuantumGateBasic(GateMatrixType matrix_type,
        SpecialFuncType special_func_type, UINT gate_property,
        const std::vector<UINT>& target_qubit_index,
        const std::vector<UINT>& target_qubit_commutation,
        const std::vector<UINT>& control_qubit_index,
        const std::vector<UINT>& control_qubit_value)
        : QuantumGateBase(Basic),
          _matrix_type(matrix_type),
          _special_func_type(special_func_type),
          _gate_property(gate_property) {
        // set target_qubit_index
        if (target_qubit_index.size() == 0) {
            throw std::invalid_argument("target_qubit_index.size() == 0");
        }
        _target_qubit_index = target_qubit_index;

        // set target_qubit_commutation
        if (target_qubit_commutation.size() == 0) {
            _target_qubit_commutation =
                std::vector<UINT>(target_qubit_index.size(), 0);
        } else {
            if (target_qubit_commutation.size() != target_qubit_index.size()) {
                throw std::invalid_argument(
                    "target_qubit_index.size() != "
                    "target_qubit_commutation.size()");
            }
            _target_qubit_commutation = target_qubit_commutation;
        }

        // set control_qubit_index
        _control_qubit_index = control_qubit_index;

        // set control_qubit_value
        if (_control_qubit_index.size() != control_qubit_value.size()) {
            throw std::invalid_argument(
                "control_qubit_index.size() != control_qubit_value.size()");
        }
        for (auto item : _control_qubit_value) {
            if (item >= 2) {
                throw std::invalid_argument(
                    "control_qubit_value contains a value that is not 0 nor 1");
            }
        }
        _control_qubit_value = control_qubit_value;
    };
    QuantumGateBasic& operator=(const QuantumGateBasic& rhs) = delete;

    virtual void _expand_control_qubit(ComplexMatrix&) const {
        if (_control_qubit_index.size() > 0)
            throw std::invalid_argument(
                "Expand control part is not implemented");
    };

    template <class Archive>
    void save(Archive& ar) const {
        // TODO!
        // Documentize SparseMatrix serializer
        ar(CEREAL_NVP(_matrix_type), CEREAL_NVP(_special_func_type),
            CEREAL_NVP(_target_qubit_index),
            CEREAL_NVP(_target_qubit_commutation),
            CEREAL_NVP(_control_qubit_index), CEREAL_NVP(_control_qubit_value),
            CEREAL_NVP(_gate_property), CEREAL_NVP(_dense_matrix_element),
            CEREAL_NVP(
                _diagonal_matrix_element),//CEREAL_NVP(_sparse_matrix_element)
        
        CEREAL_NVP(_pauli_id), CEREAL_NVP(_rotation_angle));
        
    }

    template <class Archive>
    void load(Archive& ar) {
        // TODO!
        // Documentize SparseMatrix serializer
        ar(CEREAL_NVP(_matrix_type), CEREAL_NVP(_special_func_type),
            CEREAL_NVP(_target_qubit_index),
            CEREAL_NVP(_target_qubit_commutation),
            CEREAL_NVP(_control_qubit_index), CEREAL_NVP(_control_qubit_value),
            CEREAL_NVP(_gate_property), CEREAL_NVP(_dense_matrix_element),
            CEREAL_NVP(
                _diagonal_matrix_element) //,CEREAL_NVP(_sparse_matrix_element)
        ,
        CEREAL_NVP(_pauli_id), CEREAL_NVP(_rotation_angle));

    }
    virtual ~QuantumGateBasic(){};
    virtual UINT get_qubit_count() const override {
        return (UINT)(_target_qubit_index.size() + _control_qubit_index.size());
    }
    virtual const std::vector<QuantumGateBase*>& get_kraus_list()
        const override {
        throw std::invalid_argument("Basic gate does not have Kraus list");
    }
    virtual std::vector<double> get_cumulative_distribution() const override {
        return {1.0};
    }
    virtual void reset_qubit_index_list(
        const std::vector<UINT>& src_list, const std::vector<UINT>& dst_list) {
        if (src_list.size() != dst_list.size())
            throw std::invalid_argument("src.size() != dst.size()");
        for (UINT index = 0; index < src_list.size(); ++index) {
            UINT src = src_list[index];
            UINT dst = dst_list[index];
            auto replace_func = [](std::vector<UINT>& vec, UINT src_ind,
                                    UINT dst_ind) -> void {
                for (auto& v : vec)
                    if (v == src_ind) v = dst_ind;
            };
            replace_func(_target_qubit_index, src, dst);
            replace_func(_control_qubit_index, src, dst);
        }
    };
    virtual const std::vector<UINT> get_target_index_list() const override {
        return _target_qubit_index;
    };
    virtual const std::vector<UINT> get_control_index_list() const override {
        return _control_qubit_index;
    };
    virtual const std::vector<UINT> get_qubit_index_list() const override {
        std::vector<UINT> res = _target_qubit_index;
        for (auto val : _control_qubit_index) res.push_back(val);
        return res;
    }

    static QuantumGateBasic* DenseMatrixGate(
        const std::vector<UINT>& target_qubit, const ComplexMatrix& matrix,
        const std::vector<UINT>& target_commute = {}) {
        ITYPE dim = 1ULL << target_qubit.size();
        if ((unsigned)matrix.cols() != dim)
            throw std::invalid_argument("matrix.cols() != dim");
        if ((unsigned)matrix.rows() != dim)
            throw std::invalid_argument("matrix.rows() != dim");
        auto ptr = new QuantumGateBasic(
            DenseMatrix, None, 0, target_qubit, target_commute, {}, {});
        ptr->_dense_matrix_element = matrix;
        return ptr;
    }
    static QuantumGateBasic* DiagonalMatrixGate(
        const std::vector<UINT> target_qubit,
        const ComplexVector& diagonal_vector) {
        ITYPE dim = 1ULL << target_qubit.size();
        if ((unsigned)diagonal_vector.size() != dim)
            throw std::invalid_argument("diagonal_vector.size() != dim");
        auto ptr = new QuantumGateBasic(DiagonalMatrix, None, 0, target_qubit,
            std::vector<UINT>(FLAG_COMMUTE_Z, (UINT)target_qubit.size()), {},
            {});
        ptr->_diagonal_matrix_element = diagonal_vector;
        return ptr;
    }
    static QuantumGateBasic* SparseMatrixGate(
        const std::vector<UINT>& target_qubit,
        const SparseComplexMatrix& sparse_matrix,
        const std::vector<UINT>& target_commute = {}) {
        ITYPE dim = 1ULL << target_qubit.size();
        if ((unsigned)sparse_matrix.cols() != dim)
            throw std::invalid_argument("sparse_matrix.cols() != dim");
        if ((unsigned)sparse_matrix.rows() != dim)
            throw std::invalid_argument("sparse_matrix.rows() != dim");
        auto ptr = new QuantumGateBasic(
            SparseMatrix, None, 0, target_qubit, target_commute, {}, {});
        ptr->_sparse_matrix_element = sparse_matrix;
        return ptr;
    }

    static QuantumGateBasic* PauliMatrixGate(
        const std::vector<UINT>& target_qubit,
        const std::vector<UINT>& pauli_id, const double rotation_angle) {
        if (pauli_id.size() != target_qubit.size())
            throw std::invalid_argument(
                "pauli_id.size() != target_qubit.size()");

        std::vector<UINT> target_commute((UINT)target_qubit.size(), 0);
        for (UINT ind = 0; ind < target_qubit.size(); ++ind) {
            UINT value = pauli_id[ind];
            if (value == PAULI_ID_I)
                target_commute[ind] =
                    FLAG_COMMUTE_X | FLAG_COMMUTE_Y | FLAG_COMMUTE_Z;
            if (value == PAULI_ID_X) target_commute[ind] = FLAG_COMMUTE_X;
            if (value == PAULI_ID_Y) target_commute[ind] = FLAG_COMMUTE_Y;
            if (value == PAULI_ID_Z) target_commute[ind] = FLAG_COMMUTE_Z;
            if (value >= 4)
                throw std::invalid_argument(
                    "pauli_id contains a value >= 4. ID must be any of "
                    "(I,X,Y,Z) = (0,1,2,3).");
        }
        auto ptr = new QuantumGateBasic(
            PauliMatrix, None, 0, target_qubit, target_commute, {}, {});
        ptr->_rotation_angle = rotation_angle;
        ptr->_pauli_id = pauli_id;
        return ptr;
    }

    virtual void set_special_func_type(SpecialFuncType special_func_type) {
        this->_special_func_type = special_func_type;
    }
    virtual void add_control_qubit(UINT control_index, UINT control_value) {
        if (_matrix_type != DenseMatrix) {
            throw std::invalid_argument(
                "Cannot call add_control_qubit to gate other than "
                "DenseMatrixGate");
        }
        if (std::find(_target_qubit_index.begin(), _target_qubit_index.end(),
                control_index) != _target_qubit_index.end())
            throw std::invalid_argument(
                "control_index is already in target_qubit_index");
        if (std::find(_control_qubit_index.begin(), _control_qubit_index.end(),
                control_index) != _control_qubit_index.end())
            throw std::invalid_argument(
                "control_index is already in control_qubit_index");
        if (control_value >= 2)
            throw std::invalid_argument("control_value is not 0 nor 1");

        _control_qubit_index.push_back(control_index);
        _control_qubit_value.push_back(control_value);
    }

    virtual void multiply_scalar(CPPCTYPE value) {
        if (_matrix_type == DenseMatrix)
            _dense_matrix_element *= value;
        else if (_matrix_type == SparseMatrix)
            _sparse_matrix_element *= value;
        else if (_matrix_type == DiagonalMatrix)
            _diagonal_matrix_element *= value;
        else
            throw std::invalid_argument("This gate cannot multiply scalar");
    }

    void _update_state_vector_cpu_special(QuantumStateBase* state) const;

    void _update_state_vector_cpu_general(QuantumStateBase* state) const {
        if (_matrix_type == DenseMatrix) {
            const CTYPE* matrix_ptr = reinterpret_cast<const CTYPE*>(
                this->_dense_matrix_element.data());
            // single qubit dense matrix gate
            if (_target_qubit_index.size() == 1) {
                // no control qubit
                if (_control_qubit_index.size() == 0) {
                    single_qubit_dense_matrix_gate(_target_qubit_index[0],
                        matrix_ptr, state->data_c(), state->dim);
                }
                // single control qubit
                else if (_control_qubit_index.size() == 1) {
                    single_qubit_control_single_qubit_dense_matrix_gate(
                        _control_qubit_index[0], _control_qubit_value[0],
                        _target_qubit_index[0], matrix_ptr, state->data_c(),
                        state->dim);
                }
                // multiple control qubits
                else {
                    multi_qubit_control_single_qubit_dense_matrix_gate(
                        _control_qubit_index.data(),
                        _control_qubit_value.data(),
                        (UINT)(_control_qubit_index.size()),
                        _target_qubit_index[0], matrix_ptr, state->data_c(),
                        state->dim);
                }
            }

            // multi qubit dense matrix gate
            else {
                // no control qubit
                if (_control_qubit_index.size() == 0) {
                    multi_qubit_dense_matrix_gate(_target_qubit_index.data(),
                        (UINT)(_target_qubit_index.size()), matrix_ptr,
                        state->data_c(), state->dim);
                }
                // single control qubit
                else if (_control_qubit_index.size() == 1) {
                    single_qubit_control_multi_qubit_dense_matrix_gate(
                        _control_qubit_index[0], _control_qubit_value[0],
                        _target_qubit_index.data(),
                        (UINT)(_target_qubit_index.size()), matrix_ptr,
                        state->data_c(), state->dim);
                }
                // multiple control qubit
                else {
                    multi_qubit_control_multi_qubit_dense_matrix_gate(
                        _control_qubit_index.data(),
                        _control_qubit_value.data(),
                        (UINT)(_control_qubit_index.size()),
                        _target_qubit_index.data(),
                        (UINT)(_target_qubit_index.size()), matrix_ptr,
                        state->data_c(), state->dim);
                }
            }
        } else if (_matrix_type == DiagonalMatrix) {
            const CTYPE* matrix_ptr = reinterpret_cast<const CTYPE*>(
                this->_diagonal_matrix_element.data());
            if (_target_qubit_index.size() == 1)
                single_qubit_diagonal_matrix_gate(_target_qubit_index[0],
                    matrix_ptr, state->data_c(), state->dim);
            else
                multi_qubit_diagonal_matrix_gate(_target_qubit_index.data(),
                    (UINT)_target_qubit_index.size(), matrix_ptr,
                    state->data_c(), state->dim);
        } else if (_matrix_type == SparseMatrix) {
            multi_qubit_sparse_matrix_gate_eigen(_target_qubit_index.data(),
                (UINT)(_target_qubit_index.size()),
                this->_sparse_matrix_element, state->data_c(), state->dim);
        } else if (_matrix_type == PauliMatrix) {
            if (_target_qubit_index.size() == 1) {
                if (fabs(_rotation_angle) < 1e-16) {
                    single_qubit_Pauli_gate(_target_qubit_index[0],
                        _pauli_id[0], state->data_c(), state->dim);
                } else {
                    // invert
                    single_qubit_Pauli_rotation_gate(_target_qubit_index[0],
                        _pauli_id[0], -_rotation_angle, state->data_c(),
                        state->dim);
                }
            } else {
                if (fabs(_rotation_angle) < 1e-16) {
                    multi_qubit_Pauli_gate_partial_list(
                        _target_qubit_index.data(), _pauli_id.data(),
                        (UINT)_target_qubit_index.size(), state->data_c(),
                        state->dim);
                } else {
                    // invert
                    multi_qubit_Pauli_rotation_gate_partial_list(
                        _target_qubit_index.data(), _pauli_id.data(),
                        (UINT)_target_qubit_index.size(), -_rotation_angle,
                        state->data_c(), state->dim);
                }
            }
        }
    }

    void _update_density_matrix_cpu_general(QuantumStateBase* state) const {
        if (_matrix_type == DenseMatrix) {
            const CTYPE* matrix_ptr = reinterpret_cast<const CTYPE*>(
                this->_dense_matrix_element.data());
            if (_control_qubit_index.size() == 0) {
                if (_target_qubit_index.size() == 1) {
                    dm_single_qubit_dense_matrix_gate(_target_qubit_index[0],
                        matrix_ptr, state->data_c(), state->dim);
                } else {
                    dm_multi_qubit_dense_matrix_gate(_target_qubit_index.data(),
                        (UINT)_target_qubit_index.size(), matrix_ptr,
                        state->data_c(), state->dim);
                }
            } else {
                if (_target_qubit_index.size() == 1) {
                    dm_multi_qubit_control_single_qubit_dense_matrix_gate(
                        _control_qubit_index.data(),
                        _control_qubit_value.data(),
                        (UINT)_control_qubit_index.size(),
                        _target_qubit_index[0], matrix_ptr, state->data_c(),
                        state->dim);
                } else {
                    dm_multi_qubit_control_multi_qubit_dense_matrix_gate(
                        _control_qubit_index.data(),
                        _control_qubit_value.data(),
                        (UINT)_control_qubit_index.size(),
                        _target_qubit_index.data(),
                        (UINT)_target_qubit_index.size(), matrix_ptr,
                        state->data_c(), state->dim);
                }
            }
        } else {
            throw std::invalid_argument(
                "Only DenseMatrix gate type is supported for density matrix");
        }
    }

#ifdef _USE_GPU
    void _update_state_vector_gpu(QuantumStateBase* state) {
        if (_matrix_type == DenseMatrix) {
            const CTYPE* matrix_ptr = reinterpret_cast<const CTYPE*>(
                this->_dense_matrix_element.data());
            // single qubit dense matrix gate
            if (_target_qubit_index.size() == 1) {
                // no control qubit
                if (_control_qubit_index.size() == 0) {
                    single_qubit_dense_matrix_gate_host(_target_qubit_index[0],
                        (const CPPCTYPE*)matrix_ptr, state->data(), state->dim,
                        state->get_cuda_stream(), state->device_number);
                }
                // single control qubit
                else if (_control_qubit_index.size() == 1) {
                    single_qubit_control_single_qubit_dense_matrix_gate_host(
                        _control_qubit_index[0], _control_qubit_value[0],
                        _target_qubit_index[0], (const CPPCTYPE*)matrix_ptr,
                        state->data(), state->dim, state->get_cuda_stream(),
                        state->device_number);
                }
                // multiple control qubits
                else {
                    multi_qubit_control_multi_qubit_dense_matrix_gate_host(
                        _control_qubit_index.data(),
                        _control_qubit_value.data(),
                        (UINT)(_control_qubit_index.size()),
                        _target_qubit_index.data(),
                        (UINT)(_target_qubit_index.size()),
                        (const CPPCTYPE*)matrix_ptr, state->data(), state->dim,
                        state->get_cuda_stream(), state->device_number);
                }
            }

            // multi qubit dense matrix gate
            else {
                // no control qubit
                if (_control_qubit_index.size() == 0) {
                    multi_qubit_dense_matrix_gate_host(
                        _target_qubit_index.data(),
                        (UINT)(_target_qubit_index.size()),
                        (const CPPCTYPE*)matrix_ptr, state->data(), state->dim,
                        state->get_cuda_stream(), state->device_number);
                }
                // single control qubit
                else if (_control_qubit_index.size() == 1) {
                    single_qubit_control_multi_qubit_dense_matrix_gate_host(
                        _control_qubit_index[0], _control_qubit_value[0],
                        _target_qubit_index.data(),
                        (UINT)(_target_qubit_index.size()),
                        (const CPPCTYPE*)matrix_ptr, state->data(), state->dim,
                        state->get_cuda_stream(), state->device_number);
                }
                // multiple control qubit
                else {
                    multi_qubit_control_multi_qubit_dense_matrix_gate_host(
                        _control_qubit_index.data(),
                        _control_qubit_value.data(),
                        (UINT)(_control_qubit_index.size()),
                        _target_qubit_index.data(),
                        (UINT)(_target_qubit_index.size()),
                        (const CPPCTYPE*)matrix_ptr, state->data(), state->dim,
                        state->get_cuda_stream(), state->device_number);
                }
            }
        } else {
            throw std::invalid_argument(
                "Only DenseMatrix gate type is supported for density matrix");
        }
    }
    void _update_density_matrix_gpu(QuantumStateBase* state) {
        throw std::runtime_error(
            "Density matrix simulation is not supported on GPU.");
    }
#endif

    void update_quantum_state(QuantumStateBase* state) override {
        if (state->get_device_type() == DEVICE_CPU) {
            if (state->is_state_vector()) {
                if (_special_func_type == None)
                    this->_update_state_vector_cpu_general(state);
                else
                    this->_update_state_vector_cpu_special(state);
            } else {
                this->_update_density_matrix_cpu_general(state);
            }
        } else if (state->get_device_type() == DEVICE_GPU) {
#ifdef _USE_GPU
            if (state->is_state_vector()) {
                this->_update_state_vector_gpu(state);
            } else {
                this->_update_density_matrix_gpu(state);
            }
#else
            throw std::runtime_error("GPU simulation is disabled.");
#endif
        }
    }

    virtual QuantumGateBase* copy() const override {
        return new QuantumGateBasic(*this);
    };

    virtual void to_dense_matrix() {
        this->_matrix_type = DenseMatrix;
        this->_special_func_type = None;
        this->get_target_matrix(this->_dense_matrix_element);
    }

    virtual void get_target_matrix(ComplexMatrix& matrix) const {
        if (_matrix_type == DenseMatrix) {
            matrix = this->_dense_matrix_element;
        } else if (_matrix_type == SparseMatrix) {
            matrix = this->_sparse_matrix_element.toDense();
        } else if (_matrix_type == DiagonalMatrix) {
            matrix = this->_diagonal_matrix_element.asDiagonal();
        } else if (_matrix_type == PauliMatrix) {
            ComplexMatrix pauli_matrix;
            get_Pauli_matrix(pauli_matrix, this->_pauli_id);
            ITYPE dim = 1ULL << this->_pauli_id.size();
            matrix = cos(this->_rotation_angle / 2) *
                         ComplexMatrix::Identity(dim, dim) -
                     1.i * sin(this->_rotation_angle / 2) * pauli_matrix;
        } else
            throw std::invalid_argument(
                "Cannot obtain gate matrix for this type");
    }
    virtual void get_matrix(ComplexMatrix& matrix) const {
        this->get_target_matrix(matrix);
        this->_expand_control_qubit(matrix);
    }

    virtual std::string to_string() const override {
        std::stringstream ss;
        ss << "*** Gate Info ***" << std::endl;
        ss << "target index" << std::endl;
        for (auto item : _target_qubit_index) ss << item << "," << std::endl;
        ss << "control index" << std::endl;
        for (auto item : _control_qubit_index) ss << item << "," << std::endl;
        ss << "parameter list" << std::endl;
        for (auto item : _parameter) ss << item.first << "," << std::endl;
        switch (_matrix_type) {
            case DenseMatrix:
                ss << "Dense Matrix" << std::endl;
                ss << _dense_matrix_element << std::endl;
                break;
            case SparseMatrix:
                ss << "Sparse Matrix" << std::endl;
                ss << _sparse_matrix_element << std::endl;
                break;
            case DiagonalMatrix:
                ss << "Diagonal Matrix" << std::endl;
                ss << _diagonal_matrix_element << std::endl;
                break;
            case PauliMatrix:
                ss << "Pauli Matrix" << std::endl;
                ss << "pauli string" << std::endl;
                for (auto item : _pauli_id) ss << item << "," << std::endl;
                ss << "rotation angle: " << _rotation_angle << std::endl;
                break;
            case PermutationMatrix:
                ss << "Permutation Matrix" << std::endl;
                break;
            default:
                assert(false && "Unknown gate type");
                break;
        }
        return ss.str();
    }
};

namespace gate {
DllExport QuantumGateBasic* Identity(UINT target_qubit);
DllExport QuantumGateBasic* X(UINT target_qubit);
DllExport QuantumGateBasic* Y(UINT target_qubit);
DllExport QuantumGateBasic* Z(UINT target_qubit);
DllExport QuantumGateBasic* sqrtX(UINT target_qubit);
DllExport QuantumGateBasic* sqrtXdag(UINT target_qubit);
DllExport QuantumGateBasic* sqrtY(UINT target_qubit);
DllExport QuantumGateBasic* sqrtYdag(UINT target_qubit);
DllExport QuantumGateBasic* S(UINT target_qubit);
DllExport QuantumGateBasic* Sdag(UINT target_qubit);
DllExport QuantumGateBasic* T(UINT target_qubit);
DllExport QuantumGateBasic* Tdag(UINT target_qubit);
DllExport QuantumGateBasic* H(UINT target_qubit);
DllExport QuantumGateBasic* HS(UINT target_qubit);
DllExport QuantumGateBasic* P0(UINT target_qubit);
DllExport QuantumGateBasic* P1(UINT target_qubit);
DllExport QuantumGateBasic* RX(UINT target_qubit, double rotation_angle);
DllExport QuantumGateBasic* RY(UINT target_qubit, double rotation_angle);
DllExport QuantumGateBasic* RZ(UINT target_qubit, double rotation_angle);
DllExport QuantumGateBasic* CX(UINT control_qubit, UINT target_qubit);
DllExport QuantumGateBasic* CNOT(UINT control_qubit, UINT target_qubit);
DllExport QuantumGateBasic* CY(UINT control_qubit, UINT target_qubit);
DllExport QuantumGateBasic* CZ(UINT control_qubit, UINT target_qubit);
DllExport QuantumGateBasic* SWAP(UINT target_qubit1, UINT target_qubit2);
DllExport QuantumGateBasic* ISWAP(UINT target_qubit1, UINT target_qubit2);
DllExport QuantumGateBasic* Toffoli(
    UINT control_qubit1, UINT control_qubit2, UINT target_qubit);
DllExport QuantumGateBasic* Fredkin(
    UINT control_qubit, UINT target_qubit1, UINT target_qubit2);
};  // namespace gate

class QuantumGateWrapped : public QuantumGateBase {
private:
    std::vector<QuantumGateBase*> _gate_list;
    std::vector<double> _prob_list;
    std::vector<double> _prob_cum_list;
    std::vector<UINT> _qubit_index_list;
    Random random_state;
    bool _flag_is_unital = false;
    bool _flag_save_log = false;
    std::string _reg_name = "";

    QuantumGateWrapped(MapType map_type) : QuantumGateBase(map_type){};

    void add_probabilistic_map(
        QuantumGateBase* gate, double prob, double eps = 1e-14) {
        if (prob < 0)
            throw std::invalid_argument("negative probability is assigned");
        _gate_list.push_back(gate);
        _prob_list.push_back(prob);
        double next_sum = _prob_cum_list.back() + prob;
        if (next_sum > 1.0 + eps)
            throw std::invalid_argument("sum of probability exceeds 1.0");
        _prob_cum_list.push_back(next_sum);
    }
    void update_qubit_index_list() {
        std::set<UINT> qubit_set;
        for (auto gate : _gate_list) {
            for (UINT index : gate->get_qubit_index_list()) {
                qubit_set.insert(index);
            }
        }
        _qubit_index_list.clear();
        for (auto ite = qubit_set.begin(); ite != qubit_set.end(); ++ite) {
            _qubit_index_list.push_back(*ite);
        }
    }

public:
    template <class Archive>
    void save(Archive& ar) const {
        int size_gate_list = _gate_list.size();
        ar(CEREAL_NVP(size_gate_list));
        
        for(UINT i = 0;i < _gate_list.size();++i){
            /*
            std::unique_ptr<QuantumGateBase> inputs;
            inputs.reset(_gate_list[i] -> copy());
            ar(CEREAL_NVP(inputs));
            */
        }
        ar(CEREAL_NVP(_prob_list),CEREAL_NVP(_prob_cum_list),CEREAL_NVP(_qubit_index_list),CEREAL_NVP(_flag_is_unital),CEREAL_NVP(_flag_save_log),CEREAL_NVP(_reg_name));
        
    }

    template <class Archive>
    void load(Archive& ar) {
        
        int size_gate_list;
        ar(CEREAL_NVP(size_gate_list));
        _gate_list.clear();
        for(int i = 0;i < size_gate_list;++i){
            /*
            std::unique_ptr<QuantumGateBase> outputs;
            ar(CEREAL_NVP(outputs));
            _gate_list.push_back(outputs -> copy());
            */
        }
        ar(CEREAL_NVP(_prob_list),CEREAL_NVP(_prob_cum_list),CEREAL_NVP(_qubit_index_list),CEREAL_NVP(_flag_is_unital),CEREAL_NVP(_flag_save_log),CEREAL_NVP(_reg_name));
        
    }
    QuantumGateWrapped(){};
    virtual ~QuantumGateWrapped() {
        for (auto& gate : _gate_list) {
            delete gate;
        }
    }
    virtual UINT get_qubit_count() const override {
        return (UINT)_qubit_index_list.size();
    }
    virtual const std::vector<UINT> get_qubit_index_list() const override {
        return _qubit_index_list;
    }
    virtual const std::vector<UINT> get_target_index_list() const override {
        return get_qubit_index_list();
    }
    virtual const std::vector<UINT> get_control_index_list() const override {
        return {};
    }
    virtual const std::vector<QuantumGateBase*>& get_kraus_list()
        const override {
        return _gate_list;
    }
    virtual void get_matrix(ComplexMatrix&) const {
        throw std::invalid_argument(
            "Get matrix is not supported in wrapper gate");
    }
    virtual std::vector<double> get_cumulative_distribution() const override {
        return _prob_cum_list;
    }

    static QuantumGateWrapped* ProbabilisticGate(
        std::vector<QuantumGateBase*> gates, const std::vector<double>& prob,
        bool take_ownership = false) {
        auto ptr = new QuantumGateWrapped(Probabilistic);
        ptr->_prob_list.clear();
        ptr->_prob_cum_list.clear();
        ptr->_prob_cum_list.push_back(0.);
        for (UINT index = 0; index < gates.size(); ++index) {
            if (take_ownership)
                ptr->add_probabilistic_map(gates[index], prob[index]);
            else
                ptr->add_probabilistic_map(gates[index]->copy(), prob[index]);
        }
        ptr->update_qubit_index_list();
        return ptr;
    }

    virtual void reset_qubit_index_list(
        const std::vector<UINT>& src, const std::vector<UINT>& dst) {
        for (auto gate : _gate_list) {
            gate->reset_qubit_index_list(src, dst);
        }
        this->update_qubit_index_list();
    }

    virtual QuantumGateBase* copy() const override {
        auto ptr = new QuantumGateWrapped(this->_map_type);
        for (auto gate : _gate_list) {
            ptr->_gate_list.push_back(gate->copy());
        }
        ptr->_prob_list = _prob_list;
        ptr->_prob_cum_list = _prob_cum_list;
        ptr->_qubit_index_list = _qubit_index_list;
        ptr->_flag_is_unital = _flag_is_unital;
        ptr->_flag_save_log = _flag_save_log;
        ptr->_reg_name = _reg_name;
        return ptr;
    }
    virtual std::string to_string() const override {
        return "WrappedGate (TODO)";
    }
    virtual void update_quantum_state(QuantumStateBase* state) override {
        if (_map_type == Probabilistic) {
            if (state->is_state_vector()) {
                double r = random_state.uniform();
                auto ite = std::lower_bound(
                    _prob_cum_list.begin(), _prob_cum_list.end(), r);
                assert(ite != _prob_cum_list.begin());
                size_t gate_index =
                    std::distance(_prob_cum_list.begin(), ite) - 1;

                if (gate_index < _gate_list.size()) {
                    _gate_list[gate_index]->update_quantum_state(state);
                }
            } else {
                auto org_state = state->copy();
                auto temp_state = state->copy();
                for (UINT gate_index = 0; gate_index < _gate_list.size();
                     ++gate_index) {
                    if (gate_index == 0) {
                        _gate_list[gate_index]->update_quantum_state(state);
                        state->multiply_coef(_prob_list[gate_index]);
                    } else if (gate_index + 1 < _gate_list.size()) {
                        temp_state->load(org_state);
                        _gate_list[gate_index]->update_quantum_state(
                            temp_state);
                        temp_state->multiply_coef(_prob_list[gate_index]);
                        state->add_state(temp_state);
                    } else {
                        _gate_list[gate_index]->update_quantum_state(org_state);
                        org_state->multiply_coef(_prob_list[gate_index]);
                        state->add_state(org_state);
                    }
                }
                delete org_state;
                delete temp_state;
            }
        } else {
            throw std::invalid_argument("Not implemented");
        }
    }
};

namespace gate {
DllExport QuantumGateWrapped* DepolarizingNoise(UINT index, double prob);
DllExport QuantumGateWrapped* TwoQubitDepolarizingNoise(
    UINT index1, UINT index2, double prob);
DllExport QuantumGateWrapped* BitFlipNoise(UINT index, double prob);
DllExport QuantumGateWrapped* DephasingNoise(UINT index, double prob);
DllExport QuantumGateWrapped* IndependentXZNoise(UINT index, double prob);
};  // namespace gate

//Cereal Type Registration
CEREAL_REGISTER_POLYMORPHIC_RELATION(QuantumGateBase, QuantumGateBasic);
CEREAL_REGISTER_POLYMORPHIC_RELATION(QuantumGateBase, QuantumGateWrapped);
CEREAL_REGISTER_TYPE(QuantumGateBasic);
CEREAL_REGISTER_TYPE(QuantumGateWrapped);