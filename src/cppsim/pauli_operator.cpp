
#include <boost/dynamic_bitset.hpp>

#include "utility.hpp"
#ifdef _USE_GPU
#include <gpusim/stat_ops.h>
#endif

#include <csim/stat_ops.hpp>
#include <csim/stat_ops_dm.hpp>

#include "exception.hpp"
#include "gate_factory.hpp"
#include "pauli_operator.hpp"
#include "state.hpp"

PauliOperator::PauliOperator(std::string strings, CPPCTYPE coef) : _coef(coef) {
    std::string trimmed_string = rtrim(strings);
    if (trimmed_string.length() == 0) return;
    std::stringstream ss(trimmed_string);
    std::string pauli_str;
    UINT index, pauli_type = 0;
    while (!ss.eof()) {
        ss >> pauli_str >> index;
        if (ss.fail()) {
            throw InvalidPauliIdentifierException(
                "Error: "
                "PauliOperator::PauliOperator(std::string, CPPCTYPE):"
                "Detected pauli_str without indices. Maybe mistyped? "
                "Original Pauli string: " +
                strings);
        }
        if (pauli_str == "I" || pauli_str == "i")
            pauli_type = 0;
        else if (pauli_str == "X" || pauli_str == "x")
            pauli_type = 1;
        else if (pauli_str == "Y" || pauli_str == "y")
            pauli_type = 2;
        else if (pauli_str == "Z" || pauli_str == "z")
            pauli_type = 3;
        else {
            throw InvalidPauliIdentifierException(
                "invalid Pauli string is given : " + pauli_str);
        }
        if (pauli_type != 0) this->add_single_Pauli(index, pauli_type);
    }
}

PauliOperator::PauliOperator(const std::vector<UINT>& target_qubit_list,
    std::string Pauli_operator_type_list, CPPCTYPE coef)
    : _coef(coef) {
    UINT term_count = (UINT)(strlen(Pauli_operator_type_list.c_str()));
    assert((UINT)target_qubit_list.size() == term_count);
    UINT pauli_type = 0;
    for (UINT term_index = 0; term_index < term_count; ++term_index) {
        if (Pauli_operator_type_list[term_index] == 'i' ||
            Pauli_operator_type_list[term_index] == 'I') {
            pauli_type = 0;
        } else if (Pauli_operator_type_list[term_index] == 'x' ||
                   Pauli_operator_type_list[term_index] == 'X') {
            pauli_type = 1;
        } else if (Pauli_operator_type_list[term_index] == 'y' ||
                   Pauli_operator_type_list[term_index] == 'Y') {
            pauli_type = 2;
        } else if (Pauli_operator_type_list[term_index] == 'z' ||
                   Pauli_operator_type_list[term_index] == 'Z') {
            pauli_type = 3;
        } else {
            throw InvalidPauliIdentifierException(
                "invalid Pauli string is given : " +
                std::string{Pauli_operator_type_list[term_index]});
        }

        if (pauli_type != 0)
            this->add_single_Pauli(target_qubit_list[term_index], pauli_type);
    }
}

PauliOperator::PauliOperator(const std::vector<UINT>& pauli_list, CPPCTYPE coef)
    : _coef(coef) {
    for (UINT term_index = 0; term_index < pauli_list.size(); ++term_index) {
        if (pauli_list[term_index] != 0)
            this->add_single_Pauli(term_index, pauli_list[term_index]);
    }
}

PauliOperator::PauliOperator(const std::vector<UINT>& target_qubit_index_list,
    const std::vector<UINT>& target_qubit_pauli_list, CPPCTYPE coef)
    : _coef(coef) {
    assert(target_qubit_index_list.size() == target_qubit_pauli_list.size());
    for (UINT term_index = 0; term_index < target_qubit_index_list.size();
         ++term_index) {
        this->add_single_Pauli(target_qubit_index_list[term_index],
            target_qubit_pauli_list[term_index]);
    }
}

PauliOperator::PauliOperator(const boost::dynamic_bitset<>& x,
    const boost::dynamic_bitset<>& z, CPPCTYPE coef) {
    _coef = coef;
    for (UINT i = 0; i < x.size(); i++) {
        UINT pauli_type = 0;
        if (x[i] && !z[i]) {
            pauli_type = 1;
        } else if (x[i] && z[i]) {
            pauli_type = 2;
        } else if (!x[i] && z[i]) {
            pauli_type = 3;
        }
        if (pauli_type != 0) {
            this->add_single_Pauli(i, pauli_type);
        }
    }
}

void PauliOperator::add_single_Pauli(UINT qubit_index, UINT pauli_type) {
    this->_pauli_list.push_back(SinglePauliOperator(qubit_index, pauli_type));
    while (_x.size() <= qubit_index) {
        _x.resize(_x.size() * 2 + 1);
        _z.resize(_z.size() * 2 + 1);
    }
    if (pauli_type == 1) {
        _x.set(qubit_index);
    } else if (pauli_type == 2) {
        _x.set(qubit_index);
        _z.set(qubit_index);
    } else if (pauli_type == 3) {
        _z.set(qubit_index);
    }
}

CPPCTYPE PauliOperator::get_expectation_value(
    const QuantumStateBase* state) const {
    if (state->qubit_count < this->get_qubit_count()) {
        throw InvalidPauliIdentifierException(
            "Error: "
            "PauliOperator::get_expectation_value(QuantumStateBase*):"
            "The number of qubit in PauliOperator is greater than QuantumState."
            "PauliOperator: " +
            std::to_string(this->get_qubit_count()) +
            " QuantumState: " + std::to_string(state->qubit_count));
    }
    if (state->is_state_vector()) {
#ifdef _USE_GPU
        if (state->get_device_name() == "gpu") {
            return _coef *
                   expectation_value_multi_qubit_Pauli_operator_partial_list_host(
                       this->get_index_list().data(),
                       this->get_pauli_id_list().data(),
                       (UINT)this->get_index_list().size(), state->data(),
                       state->dim, state->get_cuda_stream(),
                       state->device_number);
        } else
#endif
#ifdef _USE_MPI
            if (state->outer_qc > 0) {
            CPPCTYPE result =
                _coef *
                expectation_value_multi_qubit_Pauli_operator_partial_list_mpi(
                    this->get_index_list().data(),
                    this->get_pauli_id_list().data(),
                    (UINT)this->get_index_list().size(), state->data_c(),
                    state->dim, state->outer_qc, state->inner_qc);
            return result;
        } else
#endif
        {
            return _coef *
                   expectation_value_multi_qubit_Pauli_operator_partial_list(
                       this->get_index_list().data(),
                       this->get_pauli_id_list().data(),
                       (UINT)this->get_index_list().size(), state->data_c(),
                       state->dim);
        }
    } else {
        return _coef *
               dm_expectation_value_multi_qubit_Pauli_operator_partial_list(
                   this->get_index_list().data(),
                   this->get_pauli_id_list().data(),
                   (UINT)this->get_index_list().size(), state->data_c(),
                   state->dim);
    }
}

CPPCTYPE PauliOperator::get_expectation_value_single_thread(
    const QuantumStateBase* state) const {
    if (state->is_state_vector()) {
#ifdef _USE_GPU
        if (state->get_device_name() == "gpu") {
            // TODO: implement single_thread version of
            // expectation_value_multi_qubit_Pauli_operator_partial_list_host
            return _coef *
                   expectation_value_multi_qubit_Pauli_operator_partial_list_host(
                       this->get_index_list().data(),
                       this->get_pauli_id_list().data(),
                       (UINT)this->get_index_list().size(), state->data(),
                       state->dim, state->get_cuda_stream(),
                       state->device_number);
        }
#endif
#ifdef _USE_MPI
        if (state->outer_qc > 0)
            std::cout
                << "# Warning! This implementation with MPI is not thread-safe."
                << std::endl;
#endif
        return _coef *
               expectation_value_multi_qubit_Pauli_operator_partial_list_single_thread(
                   this->get_index_list().data(),
                   this->get_pauli_id_list().data(),
                   (UINT)this->get_index_list().size(), state->data_c(),
                   state->dim);
    } else {
        // TODO: implement single_thread version of
        // dm_expectation_value_multi_qubit_Pauli_operator_partial_list
        return _coef *
               dm_expectation_value_multi_qubit_Pauli_operator_partial_list(
                   this->get_index_list().data(),
                   this->get_pauli_id_list().data(),
                   (UINT)this->get_index_list().size(), state->data_c(),
                   state->dim);
    }
}

CPPCTYPE PauliOperator::get_transition_amplitude(
    const QuantumStateBase* state_bra,
    const QuantumStateBase* state_ket) const {
    if ((!state_bra->is_state_vector()) || (!state_ket->is_state_vector())) {
        throw NotImplementedException(
            "get_transition_amplitude for density matrix is not implemented");
    }
#ifdef _USE_GPU
    if (state_ket->get_device_name() == "gpu" &&
        state_bra->get_device_name() == "gpu") {
        return _coef *
               (CPPCTYPE)
                   transition_amplitude_multi_qubit_Pauli_operator_partial_list_host(
                       this->get_index_list().data(),
                       this->get_pauli_id_list().data(),
                       (UINT)this->get_index_list().size(), state_bra->data(),
                       state_ket->data(), state_bra->dim,
                       state_ket->get_cuda_stream(), state_ket->device_number);
    } else
#endif
    {
        return _coef *
               (CPPCTYPE)
                   transition_amplitude_multi_qubit_Pauli_operator_partial_list(
                       this->get_index_list().data(),
                       this->get_pauli_id_list().data(),
                       (UINT)this->get_index_list().size(), state_bra->data_c(),
                       state_ket->data_c(), state_bra->dim);
    }
}

PauliOperator* PauliOperator::copy() const {
    auto pauli = new PauliOperator(this->_coef);
    for (auto val : this->_pauli_list) {
        pauli->add_single_Pauli(val.index(), val.pauli_id());
    }
    return pauli;
}

std::string PauliOperator::get_pauli_string() const {
    std::string res = "";
    UINT size = _pauli_list.size();
    UINT target_index, pauli_id;
    if (size == 0) {
        return "";
    }
    for (UINT index = 0; index < size; index++) {
        target_index = _pauli_list[index].index();
        pauli_id = _pauli_list[index].pauli_id();
        if (pauli_id == 0)
            continue;
        else if (pauli_id == 1)
            res += "X";
        else if (pauli_id == 2)
            res += "Y";
        else if (pauli_id == 3)
            res += "Z";
        res += " " + std::to_string(target_index) + " ";
    }
    res.pop_back();
    return res;
}

void PauliOperator::change_coef(CPPCTYPE new_coef) { _coef = new_coef; }

boost::property_tree::ptree PauliOperator::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "PauliOperator");
    std::vector<boost::property_tree::ptree> pauli_list_pt;
    std::transform(_pauli_list.begin(), _pauli_list.end(),
        std::back_inserter(pauli_list_pt),
        [](const SinglePauliOperator& spo) { return spo.to_ptree(); });
    pt.put_child("pauli_list", ptree::to_ptree(pauli_list_pt));
    pt.put_child("coef", ptree::to_ptree(_coef));
    return pt;
}

PauliOperator PauliOperator::operator*(const PauliOperator& target) const {
    CPPCTYPE bits_coef = 1.0;
    CPPCTYPE I = 1.0i;
    auto x = _x;
    auto z = _z;
    auto target_x = target.get_x_bits();
    auto target_z = target.get_z_bits();
    if (target_x.size() != _x.size()) {
        ITYPE max_size = std::max(_x.size(), target_x.size());
        x.resize(max_size);
        z.resize(max_size);
        target_x.resize(max_size);
        target_z.resize(max_size);
    }
    ITYPE i;
    for (i = 0; i < x.size(); i++) {
        if (x[i] && !z[i]) {                    // X
            if (!target_x[i] && target_z[i]) {  // Z
                bits_coef = bits_coef * -I;
            } else if (target_x[i] && target_z[i]) {  // Y
                bits_coef = bits_coef * I;
            }
        } else if (!x[i] && z[i]) {             // Z
            if (target_x[i] && !target_z[i]) {  // X
                bits_coef = bits_coef * I;
            } else if (target_x[i] && target_z[i]) {  // Y
                bits_coef = bits_coef * -I;
            }
        } else if (x[i] && z[i]) {              // Y
            if (target_x[i] && !target_z[i]) {  // X
                bits_coef = bits_coef * -I;
            } else if (!target_x[i] && target_z[i]) {  // Z
                bits_coef = bits_coef * I;
            }
        }
    }
    PauliOperator res(
        x ^ target_x, z ^ target_z, _coef * target.get_coef() * bits_coef);
    return res;
}

PauliOperator PauliOperator::operator*(CPPCTYPE target) const {
    PauliOperator res(_x, _z, _coef * target);
    return res;
}

PauliOperator& PauliOperator::operator*=(const PauliOperator& target) {
    _coef *= target.get_coef();
    CPPCTYPE I = 1.0i;
    auto target_x = target.get_x_bits();
    auto target_z = target.get_z_bits();
    ITYPE max_size = std::max(_x.size(), target_x.size());
    if (target_x.size() != _x.size()) {
        _x.resize(max_size);
        _z.resize(max_size);
        target_x.resize(max_size);
        target_z.resize(max_size);
    }
    ITYPE i;
    for (i = 0; i < _x.size(); i++) {
        if (_x[i] && !_z[i]) {                  // X
            if (!target_x[i] && target_z[i]) {  // Z
                _coef *= -I;
            } else if (target_x[i] && target_z[i]) {  // Y
                _coef *= I;
            }
        } else if (!_x[i] && _z[i]) {           // Z
            if (target_x[i] && !target_z[i]) {  // X
                _coef *= I;
            } else if (target_x[i] && target_z[i]) {  // Y
                _coef *= -I;
            }
        } else if (_x[i] && _z[i]) {            // Y
            if (target_x[i] && !target_z[i]) {  // X
                _coef *= -I;
            } else if (!target_x[i] && target_z[i]) {  // Z
                _coef *= I;
            }
        }
    }
    auto x_bit = _x ^ target_x;
    auto z_bit = _z ^ target_z;
    _x.clear();
    _z.clear();
    _pauli_list.clear();
    _x.resize(max_size);
    _z.resize(max_size);
    for (i = 0; i < x_bit.size(); i++) {
        ITYPE pauli_type = 0;
        if (x_bit[i] && !z_bit[i]) {
            pauli_type = 1;
        } else if (x_bit[i] && z_bit[i]) {
            pauli_type = 2;
        } else if (!x_bit[i] && z_bit[i]) {
            pauli_type = 3;
        }
        if (pauli_type != 0) {
            this->add_single_Pauli(i, pauli_type);
        }
    }
    return *this;
}

PauliOperator& PauliOperator::operator*=(CPPCTYPE target) {
    _coef *= target;
    return *this;
}
