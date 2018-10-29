
#ifndef _MSC_VER
extern "C" {
#endif
#include <csim/utility.h>
#ifndef _MSC_VER
}
#endif

#include "gate_merge.hpp"
#include "gate_matrix.hpp"
#include "gate_general.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>

void get_new_qubit_list(const QuantumGateBase* gate_first, const QuantumGateBase* gate_second,
    std::vector<TargetQubitInfo>& new_target_list, std::vector<ControlQubitInfo>& new_control_list);
void get_extended_matrix(const QuantumGateBase* gate, const std::vector<TargetQubitInfo>& new_target_list, const std::vector<ControlQubitInfo>& new_control_list, ComplexMatrix& matrix);


// Create target_gate_set and control_gate_set after merging
// Any qubit index is classified as 9 cases :  (first_target, first_control, not_in_first) * (second_target, second_control, not_in_second)
// Since each target_qubit_list is not sorted and since we cannot sort them without corrupsing matrix correspondense, we cannot use stl set functions.
// Currently, all the indices are classified with a dirty way.
void get_new_qubit_list(const QuantumGateBase* gate_first, const QuantumGateBase* gate_second,
    std::vector<TargetQubitInfo>& new_target_list, std::vector<ControlQubitInfo>& new_control_list) {

    for (auto info1 : gate_first->target_qubit_list) {

        // case 0 : qubit belongs to both target_set -> property is merged, goto new_target_set
        auto ite_target = std::find_if(gate_second->target_qubit_list.begin(), gate_second->target_qubit_list.end(), [&info1](TargetQubitInfo info) {return info.index() == info1.index(); });
        if (ite_target != gate_second->target_qubit_list.end()) {
            new_target_list.push_back(TargetQubitInfo(info1.index(), info1.get_merged_property(*ite_target)));
            continue;
        }

        // case 1 : qubit belongs to first gate and second control -> first property is merged to Z, goto new_target_set
        auto ite_control = std::find_if(gate_second->control_qubit_list.begin(), gate_second->control_qubit_list.end(), [&info1](ControlQubitInfo info) {return info.index() == info1.index(); });
        if (ite_control != gate_second->control_qubit_list.end()) {
            new_target_list.push_back(TargetQubitInfo(info1.index(), info1.get_merged_property(FLAG_Z_COMMUTE)));
            continue;
        }

        // case 2 : qubit belongs to first gate and not in second -> first property is preserved, goto new_target_set
        else {
            new_target_list.push_back(TargetQubitInfo(info1));
        }
    }


    for (auto info1 : gate_first->control_qubit_list) {

        // case 3 : qubit belongs to first control and second target -> second property is merged with Z, goto new_target_set
        auto ite_target = std::find_if(gate_second->target_qubit_list.begin(), gate_second->target_qubit_list.end(), [&info1](TargetQubitInfo info) {return info.index() == info1.index(); });
        if (ite_target != gate_second->target_qubit_list.end()) {
            new_target_list.push_back(TargetQubitInfo(info1.index(), ite_target->get_merged_property(FLAG_Z_COMMUTE)));
            continue;
        }

        // case 4 : qubit belongs to first control and second control ->  if control_value is equal, goto new_control_set. If not, goto new_target_set with Z_COMMUTE
        auto ite_control = std::find_if(gate_second->control_qubit_list.begin(), gate_second->control_qubit_list.end(), [&info1](ControlQubitInfo info) {return info.index() == info1.index(); });
        if (ite_control != gate_second->control_qubit_list.end()) {
            if (info1.control_value() == ite_control->control_value()) {
                new_control_list.push_back(ControlQubitInfo(info1));
            }
            else {
                new_target_list.push_back(TargetQubitInfo(info1.index(), FLAG_Z_COMMUTE));
            }
            continue;
        }

        // case 5 : qubit belongs to first control and not in second -> goto new_target_set with Z_COMMUTE
        else {
            new_target_list.push_back(TargetQubitInfo(info1.index(), FLAG_Z_COMMUTE));
        }
    }


    for (auto info1 : gate_second->target_qubit_list) {
        auto ite_target = std::find_if(gate_first->target_qubit_list.begin(), gate_first->target_qubit_list.end(), [&info1](TargetQubitInfo info) {return info.index() == info1.index(); });
        if (ite_target != gate_first->target_qubit_list.end()) {
            continue;
        }
        auto ite_control = std::find_if(gate_first->control_qubit_list.begin(), gate_first->control_qubit_list.end(), [&info1](ControlQubitInfo info) {return info.index() == info1.index(); });
        if (ite_control != gate_first->control_qubit_list.end()) {
            continue;
        }

        // case 6 : qubit belongs to second target but not in first -> goto new_target_set with second property
        else {
            new_target_list.push_back(TargetQubitInfo(info1));
        }
    }

    for (auto info1 : gate_second->control_qubit_list) {
        auto ite_target = std::find_if(gate_first->target_qubit_list.begin(), gate_first->target_qubit_list.end(), [&info1](TargetQubitInfo info) {return info.index() == info1.index(); });
        if (ite_target != gate_first->target_qubit_list.end()) {
            continue;
        }
        auto ite_control = std::find_if(gate_first->control_qubit_list.begin(), gate_first->control_qubit_list.end(), [&info1](ControlQubitInfo info) {return info.index() == info1.index(); });
        if (ite_control != gate_first->control_qubit_list.end()) {
            continue;
        }

        // case 7 : qubit belongs to second control but not in first -> goto new_target_set with Z_COMMUTE
        else {
            new_target_list.push_back(TargetQubitInfo(info1.index(), FLAG_Z_COMMUTE));
        }
    }

    // case 8 : qubit belongs to nothing -> do nothing
}


// Join new qubit indices to target_list according to a given new_target_list, and set a new matrix to "matrix"
void get_extended_matrix(const QuantumGateBase* gate, const std::vector<TargetQubitInfo>& new_target_list, const std::vector<ControlQubitInfo>&, ComplexMatrix& matrix) {
    // New qubits index may be in either gate_target_index, gate_control_index, or it comes from the other gate.
    // Case 0 : If qubit index is in gate_target_index -> named A = original gate_target_index (Order must not be changed!!!)
    std::vector<UINT> join_from_target = gate->get_target_index_list();

    // Case 1 : If qubit index is in gate_control_index -> named B
    std::vector<UINT> join_from_control;
    ITYPE control_mask = 0;
    for (auto val : new_target_list) {
        auto ite = std::find_if(gate->control_qubit_list.begin(), gate->control_qubit_list.end(), [&val](const ControlQubitInfo& info) { return info.index() == val.index(); });
        if (ite != gate->control_qubit_list.end()) {
            join_from_control.push_back(val.index());

            if ((*ite).control_value() == 1)
                control_mask ^= (1ULL << (join_from_control.size()-1));
        }
    }

    // Case 2 : If qubit index is not in both -> named C
    std::vector<UINT> join_from_other_gate;
    for (auto val : new_target_list) {
        auto ite1 = std::find_if(gate->target_qubit_list.begin(), gate->target_qubit_list.end(), [&val](const TargetQubitInfo& info) { return info.index() == val.index(); });
        auto ite2 = std::find_if(gate->control_qubit_list.begin(), gate->control_qubit_list.end(), [&val](const ControlQubitInfo& info) { return info.index() == val.index(); });
        if (ite1 == gate->target_qubit_list.end() && ite2 == gate->control_qubit_list.end()) {
            join_from_other_gate.push_back(val.index());
        }
    }

    // At first, qubit indices are ordered as (A,C,B)
    std::vector<UINT> unsorted_new_target_index_list = join_from_target;
    unsorted_new_target_index_list.insert(unsorted_new_target_index_list.end(), join_from_other_gate.begin(), join_from_other_gate.end());
    unsorted_new_target_index_list.insert(unsorted_new_target_index_list.end(), join_from_control.begin(), join_from_control.end());

    // *** NOTE ***
    // Order of tensor product is reversed!!!
    // U_0 I_1 = I \tensor U = [[U,0], [0,U]]
    // 0-control-U_0 = |0><0| \tensor U + |1><1| \tensor I = [[U,0],[0,I]]
    // 1-control-U_0 = |0><0| \tensor I + |1><1| \tensor U = [[I,0],[0,U]]

    // *** Algorithm ***
    // The gate matrix corresponding to indices (A,C,B) has 2^|B| blocks of gate matrix with (A,C).
    // The (control_mask)-th block matrix is (A,C), and the others are Identity.
    // The gate matrix with (A,C) has 2^|C| blocks of gate matrix with A, which is equal to the original gate matrix.

    // Thus, the following steps work.
    // 1. Enumerate set B and C. -> Done
    // 2. Generate 2^{|A|+|B|+|C|}-dim identity matrix
    size_t new_matrix_qubit_count = (UINT)new_target_list.size();
    size_t new_matrix_dim = 1ULL << new_matrix_qubit_count;
    matrix = ComplexMatrix::Identity(new_matrix_dim, new_matrix_dim);

    // 3. Decide correct 2^{|A|+|C|}-dim block matrix from control values.
    ITYPE start_block_basis = (1ULL << (join_from_target.size() + join_from_other_gate.size())) * control_mask;

    // 4. Repeat 2^{|C|}-times paste of original gate matrix A .
    ComplexMatrix org_matrix;
    gate->set_matrix(org_matrix);
    size_t org_matrix_dim = 1ULL << gate->target_qubit_list.size();
    ITYPE repeat_count = 1ULL << join_from_other_gate.size();
    for (ITYPE repeat_index = 0; repeat_index < repeat_count; ++repeat_index) {
        size_t paste_start = (size_t)(start_block_basis + repeat_index * org_matrix_dim );
        matrix.block( paste_start, paste_start, org_matrix_dim, org_matrix_dim) = org_matrix;
    }

    // 5. Since the order of (C,B,A) is different from that of the other gate, we sort (C,B,A) after generating matrix.
    // We do nothing if it is already sorted
    if (!std::is_sorted(unsorted_new_target_index_list.begin(), unsorted_new_target_index_list.end())) {

        // generate ascending index of the INDEX_NUMBER of unsorted_target_qubit_index_list.
        std::vector<std::pair<UINT,UINT>> sorted_element_position;
        for (UINT i = 0; i<unsorted_new_target_index_list.size();++i){
            sorted_element_position.push_back(std::make_pair( unsorted_new_target_index_list[i], i));
        }
        std::sort(sorted_element_position.begin(), sorted_element_position.end());
        std::vector<UINT> sorted_index(sorted_element_position.size(), -1);
        for (UINT i = 0; i < sorted_index.size(); ++i) sorted_index[ sorted_element_position[i].second ] = i;

        // If target qubit is not in the sorted position, we swap the element to the element in correct position. If not, go next index.
        // This sort will stop with n-time swap in the worst case, which is smaller than the cost of std::sort.
        // We cannot directly sort target qubit list in order to swap matrix rows and columns with respect to qubit ordering.
        UINT ind1 = 0;
        while (ind1 < sorted_index.size()) {
            if (sorted_index[ind1] != ind1) {
                UINT ind2 = sorted_index[ind1];

                // move to correct position
                std::swap(sorted_index[ind1], sorted_index[ind2]);
                std::swap(unsorted_new_target_index_list[ind1], unsorted_new_target_index_list[ind2]);

                // create masks
                const UINT min_index = std::min(ind1, ind2);
                const UINT max_index = std::max(ind1, ind2);
                const ITYPE min_mask = 1ULL << min_index;
                const ITYPE max_mask = 1ULL << max_index;

                const ITYPE loop_dim = new_matrix_dim >> 2;

                for (ITYPE state_index = 0; state_index < loop_dim; ++state_index) {
                    ITYPE basis_00 = state_index;
                    basis_00 = insert_zero_to_basis_index(basis_00, min_mask, min_index);
                    basis_00 = insert_zero_to_basis_index(basis_00, max_mask, max_index);
                    ITYPE basis_01 = basis_00 ^ min_mask;
                    ITYPE basis_10 = basis_00 ^ max_mask;

                    matrix.col((size_t)basis_01).swap(matrix.col((size_t)basis_10));
                    matrix.row((size_t)basis_01).swap(matrix.row((size_t)basis_10));
                }
            }
            else ind1++;
        }
    }

    //std::cout << "unsorted " << std::endl;
    //for (auto val : unsorted_target_list) std::cout << val << " "; std::cout << std::endl;
    //std::cout << matrix << std::endl;
    //sort_target_qubit(unsorted_new_target_index_list, matrix);
    //std::cout << "sorted " << std::endl;
    //for (auto val : unsorted_target_list) std::cout << val << " "; std::cout << std::endl;
    //std::cout << matrix << std::endl;
    
}

/**
 * This function generate merged quantum gate with two given gates.
 * 
 * Currently, this is lazy two-step implementation. 
 */
namespace gate {
    QuantumGateMatrix* merge(const QuantumGateBase* gate_first, const QuantumGateBase* gate_second) {
        // obtain updated qubit information
        std::vector<TargetQubitInfo> new_target_list;
        std::vector<ControlQubitInfo> new_control_list;
        get_new_qubit_list(gate_first, gate_second, new_target_list, new_control_list);
        std::sort(new_target_list.begin(), new_target_list.end(), [](const TargetQubitInfo& a, const TargetQubitInfo& b) { return a.index() < b.index();  });
        std::sort(new_control_list.begin(), new_control_list.end(), [](const ControlQubitInfo& a, const ControlQubitInfo& b) { return a.index() < b.index();  });

        // extend gate matrix to whole qubit list
        ComplexMatrix matrix_first, matrix_second;
        get_extended_matrix(gate_first, new_target_list, new_control_list, matrix_first);
        get_extended_matrix(gate_second, new_target_list, new_control_list, matrix_second);

        ComplexMatrix orgmat1, orgmat2;
        gate_first->set_matrix(orgmat1);
        gate_second->set_matrix(orgmat2);
        //std::cout << "first gate is extended from \n" << orgmat1 << " \nto\n" << matrix_first << "\n\n";
        //std::cout << "second gate is extended from \n" << orgmat2 << " \nto\n" << matrix_second << "\n\n";

        ComplexMatrix new_matrix = matrix_second * matrix_first;

        // generate new matrix gate
        QuantumGateMatrix* new_gate = new QuantumGateMatrix(new_target_list, &new_matrix, new_control_list);
        new_gate->set_gate_property(gate_first->get_property_value() & gate_second->get_property_value());

        //std::cout << "result matrix is " << new_gate << "\n\n";
        return new_gate;
    }

    DllExport QuantumGateMatrix* merge(std::vector<const QuantumGateBase*> gate_list) {
        QuantumGateMatrix* new_gate = NULL;
        for (auto item : gate_list) {
            if (new_gate == NULL) {
                new_gate = gate::to_matrix_gate(item);
            }
            else {
                auto tmp = merge(new_gate, item);
                delete new_gate;
                new_gate = tmp;
            }
        }
        return new_gate;
    }

    DllExport QuantumGateMatrix* add(std::vector<const QuantumGateBase*> gate_list) {
        QuantumGateMatrix* new_gate = NULL;
        for (auto item : gate_list) {
            if (new_gate == NULL) {
                new_gate = gate::to_matrix_gate(item);
            }
            else {
                auto tmp = add(new_gate, item);
                delete new_gate;
                new_gate = tmp;
            }
        }
        return new_gate;
    }



    // TODO: code is almost common with merge except * or +
    QuantumGateMatrix* add(const QuantumGateBase* gate_first, const QuantumGateBase* gate_second) {
        // obtain updated qubit information
        std::vector<TargetQubitInfo> new_target_list;
        std::vector<ControlQubitInfo> new_control_list;
        get_new_qubit_list(gate_first, gate_second, new_target_list, new_control_list);
        std::sort(new_target_list.begin(), new_target_list.end(), [](const TargetQubitInfo& a, const TargetQubitInfo& b) { return a.index() < b.index();  });
        std::sort(new_control_list.begin(), new_control_list.end(), [](const ControlQubitInfo& a, const ControlQubitInfo& b) { return a.index() < b.index();  });

        // extend gate matrix to whole qubit list
        ComplexMatrix matrix_first, matrix_second;
        get_extended_matrix(gate_first, new_target_list, new_control_list, matrix_first);
        get_extended_matrix(gate_second, new_target_list, new_control_list, matrix_second);

        ComplexMatrix orgmat1, orgmat2;
        gate_first->set_matrix(orgmat1);
        gate_second->set_matrix(orgmat2);
        //std::cout << "first gate is extended from \n" << orgmat1 << " \nto\n" << matrix_first << "\n\n";
        //std::cout << "second gate is extended from \n" << orgmat2 << " \nto\n" << matrix_second << "\n\n";

        ComplexMatrix new_matrix = matrix_second + matrix_first;

        // generate new matrix gate
        QuantumGateMatrix* new_gate = new QuantumGateMatrix(new_target_list, &new_matrix, new_control_list);
        new_gate->set_gate_property(0);

        //std::cout << "result matrix is " << new_gate << "\n\n";
        return new_gate;
    }

    QuantumGateMatrix* to_matrix_gate(const QuantumGateBase* gate) {
        ComplexMatrix mat;
        gate->set_matrix(mat);
        return new QuantumGateMatrix(gate->target_qubit_list, &mat, gate->control_qubit_list);
    }

    QuantumGateBase* Probabilistic(std::vector<double> distribution, std::vector<QuantumGateBase*> gate_list) {
        return new QuantumGate_Probabilistic(distribution, gate_list);
    }

    QuantumGateBase* CPTP(std::vector<QuantumGateBase*> gate_list) {
        return new QuantumGate_CPTP(gate_list);
    }

    QuantumGateBase* Instrument(std::vector<QuantumGateBase*> gate_list, UINT classical_register_address) {
        return new QuantumGate_Instrument(gate_list, classical_register_address);
    }

    QuantumGateBase* Adaptive(QuantumGateBase* gate, std::function<bool(const std::vector<UINT>&)> func) {
        return new QuantumGate_Adaptive(gate, func);
    }
}

