
#include "qubit_info.hpp"

QubitInfo::QubitInfo(UINT index_) : _index(index_) {}
QubitInfo::~QubitInfo() {}
UINT QubitInfo::index() const { return _index; }
void QubitInfo::set_index(int idx) { _index = idx; }

TargetQubitInfo::TargetQubitInfo(void)
    : QubitInfo(invalid_qubit), _commutation_property(0){};

TargetQubitInfo::TargetQubitInfo(UINT index_)
    : QubitInfo(index_), _commutation_property(0){};

TargetQubitInfo::TargetQubitInfo(UINT index_, UINT commutation_property_)
    : QubitInfo(index_), _commutation_property(commutation_property_){};

bool TargetQubitInfo::is_commute_X() const {
    return (_commutation_property & FLAG_X_COMMUTE) != 0;
}

bool TargetQubitInfo::is_commute_Y() const {
    return (_commutation_property & FLAG_Y_COMMUTE) != 0;
}

bool TargetQubitInfo::is_commute_Z() const {
    return (_commutation_property & FLAG_Z_COMMUTE) != 0;
}

bool TargetQubitInfo::is_commute_with(const TargetQubitInfo& info) const {
    if (this->index() != info.index())
        return true;
    else if ((_commutation_property & info._commutation_property) != 0)
        return true;
    else
        return false;
}

bool TargetQubitInfo::is_commute_with(const ControlQubitInfo& info) const {
    if (this->index() != info.index())
        return true;
    else if (this->is_commute_Z())
        return true;
    else
        return false;
}

UINT TargetQubitInfo::get_merged_property(UINT property) const {
    return _commutation_property & property;
}

UINT TargetQubitInfo::get_merged_property(const TargetQubitInfo& target) const {
    return _commutation_property & target._commutation_property;
}

UINT TargetQubitInfo::get_merged_property(
    const ControlQubitInfo& control) const {
    (void)control;
    return _commutation_property & FLAG_Z_COMMUTE;
}

ControlQubitInfo::ControlQubitInfo(void)
    : QubitInfo(invalid_qubit), _control_value(1){};
ControlQubitInfo::ControlQubitInfo(UINT index_)
    : QubitInfo(index_), _control_value(1){};
ControlQubitInfo::ControlQubitInfo(UINT index_, UINT control_value_)
    : QubitInfo(index_), _control_value(control_value_){};

bool ControlQubitInfo::is_commute_with(const TargetQubitInfo& info) const {
    if (this->index() != info.index())
        return true;
    else if (info.is_commute_Z())
        return true;
    else
        return false;
}
bool ControlQubitInfo::is_commute_with(const ControlQubitInfo&) const {
    return true;
}

UINT ControlQubitInfo::control_value() const { return _control_value; }
