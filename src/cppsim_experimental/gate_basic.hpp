#pragma once

#include "gate.hpp"

class DllExport QuantumGateBasic : public QuantumGateBase {
private:
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

public:
    QuantumGateBasic() = default;
    QuantumGateBasic& operator=(const QuantumGateBasic& rhs) = delete;

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

    virtual void _expand_control_qubit(ComplexMatrix&) const {
        if (_control_qubit_index.size() > 0)
            throw std::invalid_argument(
                "Expand control part is not implemented");
    };

    template <class Archive>
    void save(Archive& ar) const {
        ar(cereal::base_class<QuantumGateBase>(this));
        // TODO!
        // Documentize SparseMatrix serializer
        ar(CEREAL_NVP(_matrix_type), CEREAL_NVP(_special_func_type),
            CEREAL_NVP(_target_qubit_index),
            CEREAL_NVP(_target_qubit_commutation),
            CEREAL_NVP(_control_qubit_index), CEREAL_NVP(_control_qubit_value),
            CEREAL_NVP(_gate_property), CEREAL_NVP(_dense_matrix_element),
            CEREAL_NVP(_diagonal_matrix_element),
            CEREAL_NVP(_sparse_matrix_element), CEREAL_NVP(_pauli_id),
            CEREAL_NVP(_rotation_angle));
    }

    template <class Archive>
    void load(Archive& ar) {
        ar(cereal::base_class<QuantumGateBase>(this));
        // TODO!
        // Documentize SparseMatrix serializer
        ar(CEREAL_NVP(_matrix_type), CEREAL_NVP(_special_func_type),
            CEREAL_NVP(_target_qubit_index),
            CEREAL_NVP(_target_qubit_commutation),
            CEREAL_NVP(_control_qubit_index), CEREAL_NVP(_control_qubit_value),
            CEREAL_NVP(_gate_property), CEREAL_NVP(_dense_matrix_element),
            CEREAL_NVP(_diagonal_matrix_element),
            CEREAL_NVP(_sparse_matrix_element), CEREAL_NVP(_pauli_id),
            CEREAL_NVP(_rotation_angle));
    }
    virtual std::string dump_as_byte() const override {
        // serialize quantum gate
        std::ostringstream ss;
        {
            cereal::PortableBinaryOutputArchive archive(ss);
            archive(*this);
        }
        return ss.str();
    }
    virtual void load_from_byte(std::string obj) override {
        // deserialize quantum gate
        std::istringstream ss(obj);
        {
            cereal::PortableBinaryInputArchive archive(ss);
            archive(*this);
        }
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
        throw std::invalid_argument("Basic gate does not have distribution");
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

    virtual void _set_special_func_type(SpecialFuncType special_func_type) {
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

    void _update_state_vector_cpu_general(QuantumStateBase* state) const;

    void _update_density_matrix_cpu_general(QuantumStateBase* state) const;

#ifdef _USE_GPU
    void _update_state_vector_gpu(QuantumStateBase* state);
    void _update_density_matrix_gpu(QuantumStateBase* state);
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

// Cereal Type Registration
CEREAL_REGISTER_TYPE(QuantumGateBasic);
