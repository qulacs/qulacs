#include "single_fermion_operator.hpp"

#include <regex>
#include <vector>

#include "state.hpp"
#include "type.hpp"

SingleFermionOperator::SingleFermionOperator(){};

SingleFermionOperator::SingleFermionOperator(
    const std::vector<UINT>& target_index_list,
    const std::vector<UINT>& action_id_list)
    : _target_index(target_index_list), _action_id(action_id_list) {
    assert(_target_index.size() == _action_id.size());
};

SingleFermionOperator::SingleFermionOperator(std::string action_string) {
    std::string pattern = "([0-9]+)(\\^?)\\s*";
    std::regex re(pattern);
    std::cmatch result;
    while (std::regex_search(action_string.c_str(), result, re)) {
        UINT index = (UINT)std::stoul(result[1].str());
        std::string action = result[2].str();
        UINT action_id;
        _target_index.push_back(index);

        if (action == "")
            action_id = ACTION_DESTROY_ID;
        else if (action == "^")
            action_id = ACTION_CREATE_ID;
        else
            assert(false && "Error in regex");
        _action_id.push_back(action_id);

        action_string = result.suffix();
    }
    assert(_target_index.size() == _action_id.size());
};

const std::vector<UINT>& SingleFermionOperator::get_target_index_list() const {
    return _target_index;
}

const std::vector<UINT>& SingleFermionOperator::get_action_id_list() const {
    return _action_id;
}

SingleFermionOperator SingleFermionOperator::operator*(
    const SingleFermionOperator& target) const {
    auto target_index_list = _target_index;
    auto tmp_target_index = target.get_target_index_list();

    auto action_id_list = _action_id;
    auto tmp_action_id = target.get_action_id_list();

    int base_size = target_index_list.size();
    int changed_size = base_size + tmp_target_index.size();

    target_index_list.resize(changed_size);
    action_id_list.resize(changed_size);

    ITYPE i;
#pragma omp parallel for
    for (i = 0; i < target_index_list.size(); i++) {
        int insert_pos = base_size + i;
        target_index_list[insert_pos] = tmp_target_index[i];
        action_id_list[insert_pos] = tmp_action_id[i];
    }

    SingleFermionOperator res(target_index_list, action_id_list);

    return res;
}

SingleFermionOperator& SingleFermionOperator::operator*=(
    const SingleFermionOperator& target) {
    auto target_index_list = target.get_target_index_list();
    auto action_id_list = target.get_action_id_list();

    int base_size = _target_index.size();
    int changed_size = base_size + target_index_list.size();
    _target_index.resize(changed_size);
    _action_id.resize(changed_size);

    ITYPE i;
#pragma omp parallel for
    for (i = 0; i < target_index_list.size(); i++) {
        int insert_pos = base_size + i;
        _target_index[insert_pos] = target_index_list[i];
        _action_id[insert_pos] = action_id_list[i];
    }

    return *this;
}

std::string SingleFermionOperator::to_string() const {
    std::string res;
    for (int i = 0; i < _target_index.size(); i++) {
        if (i > 0) {
            res.push_back(' ');
        }
        res += std::to_string(_target_index[i]);
        if (_action_id[i] == 1) {
            res.push_back('^');
        }
    }
    return res;
}