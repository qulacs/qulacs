#pragma once

#include <regex>
#include <vector>

#include "state.hpp"
#include "type.hpp"

enum {
    ACTION_DESTROY_ID = 0,
    ACTION_CREATE_ID = 1,
};

class DllExport SingleFermionOperator {
private:
    std::vector<UINT> _target_index;
    std::vector<UINT> _action_id;

public:
    SingleFermionOperator(const std::vector<UINT>& target_index_list,
        const std::vector<UINT>& action_id_list);

    SingleFermionOperator(std::string action_string);
};