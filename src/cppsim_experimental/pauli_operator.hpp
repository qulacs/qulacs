#pragma once

#include <boost/dynamic_bitset.hpp>
#include <regex>
#include <vector>

#include "state.hpp"
#include "type.hpp"

enum {
    PAULI_ID_I = 0,
    PAULI_ID_X = 1,
    PAULI_ID_Y = 2,
    PAULI_ID_Z = 3,
};

class DllExport MultiQubitPauliOperator {
private:
    std::vector<UINT> _target_index;
    std::vector<UINT> _pauli_id;
    boost::dynamic_bitset<> _z;
    boost::dynamic_bitset<> _x;

    void set_bit(UINT pauli_id, UINT target_index);

public:
    MultiQubitPauliOperator() {};
    explicit MultiQubitPauliOperator(const std::vector<UINT>& target_qubit_index_list,
        const std::vector<UINT>& pauli_id_list)
        : _target_index(target_qubit_index_list), _pauli_id(pauli_id_list) {
        for (ITYPE i = 0; i < pauli_id_list.size(); i++) {
            set_bit(pauli_id_list[i], target_qubit_index_list[i]);
        }
    };

    explicit MultiQubitPauliOperator(std::string pauli_string) {
        std::string pattern = "([IXYZ])\\s*([0-9]+)\\s*";
        std::regex re(pattern);
        std::cmatch result;
        while (std::regex_search(pauli_string.c_str(), result, re)) {
            std::string pauli = result[1].str();
            UINT index = (UINT)std::stoul(result[2].str());
            _target_index.push_back(index);
            UINT pauli_id;
            if (pauli == "I")
                pauli_id = PAULI_ID_I;
            else if (pauli == "X")
                pauli_id = PAULI_ID_X;
            else if (pauli == "Y")
                pauli_id = PAULI_ID_Y;
            else if (pauli == "Z")
                pauli_id = PAULI_ID_Z;
            else
                assert(false && "Error in regex");
            _pauli_id.push_back(pauli_id);
            set_bit(pauli_id, index);
            pauli_string = result.suffix();
        }
        assert(_target_index.size() == _pauli_id.size());
    }

    MultiQubitPauliOperator(
        const boost::dynamic_bitset<>& x, const boost::dynamic_bitset<>& z)
        : _x(x), _z(z) {
        ITYPE index;
#pragma omp parallel for
        for (index = 0; index < this->_x.size(); index++) {
            UINT pauli_id;
            if (!this->_x[index] && !this->_z[index])
                pauli_id = PAULI_ID_I;
            else if (!this->_x[index] && this->_z[index])
                pauli_id = PAULI_ID_Z;
            else if (this->_x[index] && !this->_z[index])
                pauli_id = PAULI_ID_X;
            else if (this->_x[index] && this->_z[index])
                pauli_id = PAULI_ID_Y;
            _target_index.push_back(index);
            _pauli_id.push_back(pauli_id);
        }
    };

    virtual ~MultiQubitPauliOperator() {};

    virtual const std::vector<UINT>& get_pauli_id_list() const;
    virtual const std::vector<UINT>& get_index_list() const;
    virtual const boost::dynamic_bitset<>& get_x_bits() const { return this->_x; }
    virtual const boost::dynamic_bitset<>& get_z_bits() const { return this->_z; }

    virtual void add_single_Pauli(UINT qubit_index, UINT pauli_type);

    virtual CPPCTYPE get_expectation_value(
        const QuantumStateBase* state) const;

    virtual CPPCTYPE get_transition_amplitude(const QuantumStateBase* state_bra,
        const QuantumStateBase* state_ket) const;

    MultiQubitPauliOperator* copy() const {
        const auto pauli = new MultiQubitPauliOperator(_target_index, _pauli_id);
        return pauli;
    }

    bool operator==(const MultiQubitPauliOperator& target) const {
        auto x = this->_x;
        auto z = this->_z;
        auto target_x = target.get_x_bits();
        auto target_z = target.get_x_bits();
        if (target_x.size() != this->_x.size()) {
            size_t max_size = std::max(this->_x.size(), target_x.size());
            x.resize(max_size);
            z.resize(max_size);
            target_x.resize(max_size);
            target_z.resize(max_size);
        }
        return x == target_x && z == target_z;
    }

    MultiQubitPauliOperator operator*(
        const MultiQubitPauliOperator& target) const {
        auto x = this->_x;
        auto z = this->_z;
        auto target_x = target.get_x_bits();
        auto target_z = target.get_x_bits();
        if (target_x.size() != this->_x.size()) {
            size_t max_size = std::max(this->_x.size(), target_x.size());
            x.resize(max_size);
            z.resize(max_size);
            target_x.resize(max_size);
            target_z.resize(max_size);
        }
        MultiQubitPauliOperator res(x ^ target_x, z ^ target_z);
        return res;
    }

    MultiQubitPauliOperator& operator*=(const MultiQubitPauliOperator& target) {
        auto target_x = target.get_x_bits();
        auto target_z = target.get_z_bits();
        size_t max_size = std::max(this->_x.size(), target_x.size());
        if (target_x.size() != this->_x.size()) {
            this->_x.resize(max_size);
            this->_z.resize(max_size);
            target_x.resize(max_size);
            target_z.resize(max_size);
        }
        this->_x ^= target_x;
        this->_z ^= target_z;
        _target_index.clear();
        _pauli_id.clear();
        ITYPE i;
#pragma omp parallel for
        for (i = 0; i < max_size; i++) {
            UINT pauli_id = PAULI_ID_I;
            if (this->_x[i] && !this->_z[i]) {
                pauli_id = PAULI_ID_X;
            }
            else if (this->_x[i] && this->_z[i]) {
                pauli_id = PAULI_ID_Y;
            }
            else if (!this->_x[i] && this->_z[i]) {
                pauli_id = PAULI_ID_Z;
            }
            _target_index.push_back(i);
            _pauli_id.push_back(pauli_id);
        }
        return *this;
    }
};

using PauliOperator = MultiQubitPauliOperator;
