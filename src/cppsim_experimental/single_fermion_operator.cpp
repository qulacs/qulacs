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