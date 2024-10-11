#include "circuit_optimizer.hpp"

#include <stdio.h>

#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "circuit.hpp"
#include "gate.hpp"
#include "gate_factory.hpp"
#include "gate_matrix.hpp"
#include "gate_merge.hpp"
#include "qubit_table.hpp"

#define LOG \
    if (log_enabled) std::cerr << "[CircOpt] "

struct GateProp {
    const QuantumGateBase* gate;
    UINT index;
};

typedef boost::adjacency_list<boost::vecS,  // OutEdgeList
    boost::vecS,                            // VertexList
    boost::directedS,                       // Directed
    GateProp                                // VertexProperties
    >
    DepGraph;
typedef boost::graph_traits<DepGraph>::vertex_descriptor Vertex;

UINT QuantumCircuitOptimizer::get_rightmost_commute_index(UINT gate_index) {
    UINT cursor = gate_index + 1;
    for (; cursor < circuit->gate_list.size(); ++cursor) {
        if (!circuit->gate_list[gate_index]->is_commute(
                circuit->gate_list[cursor]))
            break;
    }
    return cursor - 1;
}

UINT QuantumCircuitOptimizer::get_leftmost_commute_index(UINT gate_index) {
    // be careful for underflow of unsigned value
    int cursor = (signed)(gate_index - 1);
    for (; cursor >= 0; cursor--) {
        if (!circuit->gate_list[gate_index]->is_commute(
                circuit->gate_list[cursor]))
            break;
    }
    return cursor + 1;
}

UINT QuantumCircuitOptimizer::get_merged_gate_size(
    UINT gate_index1, UINT gate_index2) {
    auto fetch_target_index =
        [](const std::vector<TargetQubitInfo>& info_list) {
            std::vector<UINT> index_list;
            std::transform(info_list.cbegin(), info_list.cend(),
                std::back_inserter(index_list),
                [](const TargetQubitInfo& info) { return info.index(); });
            return index_list;
        };
    auto fetch_control_index =
        [](const std::vector<ControlQubitInfo>& info_list) {
            std::vector<UINT> index_list;
            std::transform(info_list.cbegin(), info_list.cend(),
                std::back_inserter(index_list),
                [](const ControlQubitInfo& info) { return info.index(); });
            return index_list;
        };

    auto target_index_list1 =
        fetch_target_index(circuit->gate_list[gate_index1]->target_qubit_list);
    auto target_index_list2 =
        fetch_target_index(circuit->gate_list[gate_index2]->target_qubit_list);
    auto control_index_list1 = fetch_control_index(
        circuit->gate_list[gate_index1]->control_qubit_list);
    auto control_index_list2 = fetch_control_index(
        circuit->gate_list[gate_index2]->control_qubit_list);

    std::sort(target_index_list1.begin(), target_index_list1.end());
    std::sort(target_index_list2.begin(), target_index_list2.end());
    std::sort(control_index_list1.begin(), control_index_list1.end());
    std::sort(control_index_list2.begin(), control_index_list2.end());

    std::vector<UINT> target_index_merge, control_index_merge, whole_index;
    std::set_union(target_index_list1.begin(), target_index_list1.end(),
        target_index_list2.begin(), target_index_list2.end(),
        std::back_inserter(target_index_merge));
    std::set_union(control_index_list1.begin(), control_index_list1.end(),
        control_index_list2.begin(), control_index_list2.end(),
        std::back_inserter(control_index_merge));
    std::set_union(target_index_merge.begin(), target_index_merge.end(),
        control_index_merge.begin(), control_index_merge.end(),
        std::back_inserter(whole_index));
    return (UINT)(whole_index.size());
}

/////////////////////////////////////

bool QuantumCircuitOptimizer::is_neighboring(
    UINT gate_index1, UINT gate_index2) {
    assert(gate_index1 != gate_index2);
    if (gate_index1 > gate_index2) std::swap(gate_index1, gate_index2);
    UINT ind1_right = this->get_rightmost_commute_index(gate_index1);
    UINT ind2_left = this->get_leftmost_commute_index(gate_index2);
    return ind2_left <= ind1_right + 1;
}

void QuantumCircuitOptimizer::optimize(
    QuantumCircuit* circuit_, UINT max_block_size, UINT swap_level) {
    circuit = circuit_;
    set_qubit_count();

    insert_swap_gates(swap_level);

    bool merged_flag = true;
    while (merged_flag) {
        merged_flag = false;
        for (UINT ind1 = 0; ind1 < circuit->gate_list.size(); ++ind1) {
            for (UINT ind2 = ind1 + 1; ind2 < circuit->gate_list.size();
                 ++ind2) {
                // parametric gate cannot be merged
                if (circuit->gate_list[ind1]->is_parametric() ||
                    circuit->gate_list[ind2]->is_parametric())
                    continue;

                // if merged block size is larger than max_block_size, we cannot
                // merge them
                if (this->get_merged_gate_size(ind1, ind2) > max_block_size)
                    continue;

                // we skip merging that would interfere swap insertion
                if (!can_merge_with_swap_insertion(ind1, ind2, swap_level))
                    continue;

                // if they are separated by not-commutive gate, we cannot merge
                // them
                // TODO: use cache for merging
                if (!this->is_neighboring(ind1, ind2)) continue;

                // generate merged gate
                auto merged_gate = gate::merge(
                    circuit->gate_list[ind1], circuit->gate_list[ind2]);

                // remove merged two gates, and append new one
                UINT ind2_left = this->get_leftmost_commute_index(ind2);
                // insert at max(just after first_applied_gate, just before
                // left-most gate commuting with later_applied_gate ) Insertion
                // point is always later than the first, and always earlier than
                // the second.
                UINT insert_point = std::max(ind2_left, ind1 + 1);

                // Not to change index with removal, process from later ones to
                // earlier ones.
                circuit->remove_gate(ind2);
                circuit->add_gate(merged_gate, insert_point);
                circuit->remove_gate(ind1);

                ind2--;
                merged_flag = true;
            }
        }
    }
}

void QuantumCircuitOptimizer::optimize_light(
    QuantumCircuit* circuit_, UINT swap_level) {
    circuit = circuit_;
    set_qubit_count();

    insert_swap_gates(swap_level);

    UINT qubit_count = circuit->qubit_count;
    std::vector<std::pair<int, std::vector<UINT>>> current_step(
        qubit_count, std::make_pair(-1, std::vector<UINT>()));
    for (UINT ind1 = 0; ind1 < circuit->gate_list.size(); ++ind1) {
        QuantumGateBase* gate = circuit->gate_list[ind1];
        std::vector<UINT> target_qubits;
        std::vector<UINT> parent_qubits;

        for (auto val : gate->get_target_index_list())
            target_qubits.push_back(val);
        for (auto val : gate->get_control_index_list())
            target_qubits.push_back(val);
        std::sort(target_qubits.begin(), target_qubits.end());

        int pos = -1;
        int hit = -1;
        for (UINT target_qubit : target_qubits) {
            if (current_step[target_qubit].first > pos) {
                pos = current_step[target_qubit].first;
                hit = target_qubit;
            }
        }
        if (hit != -1) parent_qubits = current_step[hit].second;
        if (std::includes(parent_qubits.begin(), parent_qubits.end(),
                target_qubits.begin(), target_qubits.end())) {
            // we merge gates only when it does not interfere swap insertion
            if (can_merge_with_swap_insertion(pos, ind1, swap_level)) {
                // parametric gate cannot be merged
                if (circuit->gate_list[pos]->is_parametric() ||
                    gate->is_parametric())
                    continue;
                auto merged_gate = gate::merge(circuit->gate_list[pos], gate);
                circuit->remove_gate(ind1);
                circuit->add_gate(merged_gate, pos + 1);
                circuit->remove_gate(pos);
                ind1--;
            }
        } else {
            for (auto target_qubit : target_qubits) {
                current_step[target_qubit] = make_pair(ind1, target_qubits);
            }
        }
    }
}

QuantumGateMatrix* QuantumCircuitOptimizer::merge_all(
    const QuantumCircuit* circuit_) {
    QuantumGateBase* identity = gate::Identity(0);
    QuantumGateMatrix* current_gate = gate::to_matrix_gate(identity);
    QuantumGateMatrix* next_gate = NULL;
    delete identity;

    for (auto gate : circuit_->gate_list) {
        next_gate = gate::merge(current_gate, gate);
        delete current_gate;
        current_gate = next_gate;
    }
    return current_gate;
}

////////////////////////////////////////////////////////////
// for swap insertion
////////////////////////////////////////////////////////////

static std::string to_string(std::unordered_set<UINT> s) {
    std::vector<UINT> v(s.cbegin(), s.cend());
    std::sort(v.begin(), v.end());

    std::string str;
    for (UINT i : v) {
        str += std::to_string(i);
        str += ",";
    }
    return str;
}

static std::multimap<const QuantumGateBase*, const QuantumGateBase*>
build_parent_gate_map(DepGraph graph) {
    std::multimap<const QuantumGateBase*, const QuantumGateBase*>
        parent_gate_map;

    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        parent_gate_map.emplace(graph[boost::target(e, graph)].gate,
            graph[boost::source(e, graph)].gate);
    }
    return parent_gate_map;
}

static void write_dot(std::ostream& out, DepGraph graph) {
    class label_writer {
    public:
        label_writer(const DepGraph& graph) : graph_(graph) {}
        void operator()(std::ostream& out, const Vertex& v) const {
            auto& gate = graph_[v].gate;
            auto& index = graph_[v].index;

            out << "[label=\"";
            out << index << ": " << gate->get_name() << "(t:{";
            for (auto q_idx : gate->get_target_index_list()) {
                out << q_idx << ",";
            }
            out << "}, c:{";
            for (auto q_idx : gate->get_control_index_list()) {
                out << q_idx << ",";
            }
            out << "})\"]";
        }

    private:
        const DepGraph& graph_;
    };

    struct graph_writer {
        void operator()(std::ostream& out) const {
            out << "node [shape=rectangle]" << std::endl;
        }
    };

    boost::write_graphviz(out, graph, label_writer(graph),
        boost::default_writer(), graph_writer());
}

static bool is_non_communication_gate(const QuantumGateBase* gate) {
    auto gate_name = gate->get_name();
    return (gate->is_diagonal() || gate_name == "Projection-0" ||
            gate_name == "Projection-1" || gate_name == "DiagonalMatrix");
}

static bool is_swap_gate(const QuantumGateBase* gate) {
    auto gate_name = gate->get_name();
    return (gate_name == "SWAP" || gate_name == "FusedSWAP");
}

static std::unordered_set<UINT> get_qubits_needing_communication(
    const QuantumGateBase* gate) {
    std::unordered_set<UINT> qubits;

    if (!is_non_communication_gate(gate)) {
        auto target_indexes = gate->get_target_index_list();
        qubits.insert(target_indexes.cbegin(), target_indexes.cend());
    }

    return qubits;
}

static DepGraph build_dep_graph(QuantumCircuit* circuit) {
    DepGraph graph;

    const UINT num_gates = circuit->gate_list.size();
    const UINT full_commute_prop =
        (FLAG_X_COMMUTE | FLAG_Y_COMMUTE | FLAG_Z_COMMUTE);

    std::vector<UINT> qubit_to_commute_prop(circuit->qubit_count, 0x0);
    std::vector<std::vector<Vertex>> qubit_to_parent_vertexes(
        circuit->qubit_count);
    std::vector<std::vector<Vertex>> qubit_to_child_vertexes(
        circuit->qubit_count);

    for (UINT i = 0; i < num_gates; i++) {
        auto g = circuit->gate_list[i];
        auto v = add_vertex(graph);

        graph[v].gate = g;
        graph[v].index = i;

        auto collect_parent_vertexes = [&](const UINT q,
                                           const UINT commute_prop) {
            std::unordered_set<Vertex> v_set;

            // if Identity gate, there is no dependency
            if (commute_prop == full_commute_prop) {
                return v_set;
            }

            // depend target vertexes for current commute type
            auto& parent_vertexes = qubit_to_parent_vertexes[q];
            // depend source vertexes for current commute type
            auto& child_vertexes = qubit_to_child_vertexes[q];

            // both are the same commute_prop and not zero
            if ((commute_prop & qubit_to_commute_prop[q]) != 0) {
                child_vertexes.push_back(v);
                v_set.insert(parent_vertexes.cbegin(), parent_vertexes.cend());
            } else {
                v_set.insert(child_vertexes.cbegin(), child_vertexes.cend());

                // swap parent_vertexes and child_vertexes
                // to set child_vertexes to parent_vertexes
                std::swap(parent_vertexes, child_vertexes);
                child_vertexes.clear();
                child_vertexes.push_back(v);
                // check if commute_prop has only one or no prop.
                if (commute_prop == FLAG_X_COMMUTE ||
                    commute_prop == FLAG_Y_COMMUTE ||
                    commute_prop == FLAG_Z_COMMUTE || commute_prop == 0) {
                    qubit_to_commute_prop[q] = commute_prop;
                } else {
                    throw std::runtime_error(
                        "Error: QuantumCircuitOptimizer::build_dep_graph() "
                        "unexpected commute_prop");
                }
            }

            return v_set;
        };

        std::unordered_set<Vertex> parent_vertexes;

        for (auto qubit_info : g->target_qubit_list) {
            const UINT commute_prop =
                qubit_info.get_merged_property(full_commute_prop);
            auto deps =
                collect_parent_vertexes(qubit_info.index(), commute_prop);
            parent_vertexes.insert(deps.cbegin(), deps.cend());
        }
        for (auto qubit_info : g->control_qubit_list) {
            const UINT commute_prop = FLAG_Z_COMMUTE;
            auto deps =
                collect_parent_vertexes(qubit_info.index(), commute_prop);
            parent_vertexes.insert(deps.cbegin(), deps.cend());
        }

        for (auto w : parent_vertexes) {
            boost::add_edge(w, v, graph);
        }
    }

    return graph;
}

void QuantumCircuitOptimizer::set_qubit_count(void) {
    // if local_qc is 1, state vector is not distributed
    UINT log_nodes = std::log2(mpisize);
    if (circuit->qubit_count >= log_nodes + 2) {
        local_qc = circuit->qubit_count - log_nodes;
        global_qc = log_nodes;
    } else {
        local_qc = circuit->qubit_count;
        global_qc = 0;
    }
}

bool QuantumCircuitOptimizer::can_merge_with_swap_insertion(
    UINT gate_idx1, UINT gate_idx2, UINT swap_level) {
    // if swap insertion is disabled, can merge
    if (swap_level == 0) {
        return true;
    }

    auto& gate1 = circuit->gate_list[gate_idx1];
    auto& gate2 = circuit->gate_list[gate_idx2];

    auto is_global_qubit = [&](auto idx) { return idx >= local_qc; };

    auto is_swap_or_nocomm_gate_with_global_target = [&](QuantumGateBase* g) {
        auto targets = g->get_target_index_list();
        return (is_swap_gate(g) || is_non_communication_gate(g)) &&
               std::any_of(targets.cbegin(), targets.cend(), is_global_qubit);
    };

    if (is_swap_or_nocomm_gate_with_global_target(gate1) ||
        is_swap_or_nocomm_gate_with_global_target(gate2)) {
        return false;
    }

    auto controls_gate1 = gate1->get_control_index_list();
    auto controls_gate2 = gate2->get_control_index_list();
    std::unordered_set<UINT> global_control_gate1;
    std::unordered_set<UINT> global_control_gate2;

    std::copy_if(controls_gate1.cbegin(), controls_gate1.cend(),
        std::inserter(global_control_gate1, global_control_gate1.begin()),
        is_global_qubit);
    std::copy_if(controls_gate2.cbegin(), controls_gate2.cend(),
        std::inserter(global_control_gate2, global_control_gate2.begin()),
        is_global_qubit);

    return global_control_gate1 == global_control_gate2;
}

bool QuantumCircuitOptimizer::needs_communication(
    const UINT gate_index, const QubitTable& qt, GateReplacer& replacer) {
    auto gate = replacer.get_replaced_gate(circuit->gate_list[gate_index]);
    auto logical_qubits = get_qubits_needing_communication(gate);
    return std::any_of(
        logical_qubits.cbegin(), logical_qubits.cend(), [&](auto& l_q) {
            // true if physical qubit is global qubit
            return qt.l2p[l_q] >= local_qc;
        });
}

UINT QuantumCircuitOptimizer::move_gates_without_communication(
    const UINT gate_idx, const QubitTable& qt,
    const std::multimap<const QuantumGateBase*, const QuantumGateBase*>&
        dep_map,
    std::unordered_set<const QuantumGateBase*>& processed_gates,
    GateReplacer& replacer) {
    const UINT num_gates = circuit->gate_list.size();
    UINT moved_gates = 0;
    for (UINT i = gate_idx; i < num_gates; i++) {
        // no comm & dependency is solved
        if (!needs_communication(i, qt, replacer)) {
            auto range = dep_map.equal_range(circuit->gate_list[i]);

            bool is_dep_solved = true;
            for (auto it = range.first; it != range.second; ++it) {
                if (processed_gates.find(it->second) == processed_gates.end()) {
                    is_dep_solved = false;
                }
            }

            if (is_dep_solved) {
                auto g = circuit->gate_list[i];
                circuit->move_gate(i, gate_idx + moved_gates);
                moved_gates++;

                auto rewritten = qt.rewrite_gate_qubit_indexes(
                    replacer.get_replaced_gate(g));
                replacer.set_replaced_gate(g, rewritten);
                processed_gates.insert(g);
            }
        }
    }
    return moved_gates;
}

std::unordered_set<UINT> QuantumCircuitOptimizer::find_next_local_qubits(
    const UINT start_gate_idx, GateReplacer& replacer) {
    // next local qubit set (logical index)
    std::unordered_set<UINT> next_local_qubits;

    for (UINT gate_idx = start_gate_idx; gate_idx < circuit->gate_list.size();
         gate_idx++) {
        auto gate = replacer.get_replaced_gate(circuit->gate_list[gate_idx]);
        auto addition_qubits = get_qubits_needing_communication(gate);

        for (auto q : next_local_qubits) {
            addition_qubits.erase(q);
        }

        // if (# of qubits after adding the gate)<=(# of local qubits),
        // add gate indexes
        if (next_local_qubits.size() + addition_qubits.size() <= local_qc) {
            next_local_qubits.insert(
                addition_qubits.cbegin(), addition_qubits.cend());
        } else {
            break;
        }
    }

    LOG << "next local qubits: " << to_string(next_local_qubits) << std::endl;

    return next_local_qubits;
}

UINT QuantumCircuitOptimizer::move_matching_qubits_to_local_upper(
    UINT lowest_idx, QubitTable& qt, std::function<bool(UINT)> is_matched,
    UINT gate_insertion_pos) {
    // bool is_matched(UINT logical_qubit)

    UINT num_inserted_gates = 0;
    int j = lowest_idx - 1;
    for (int i = local_qc - 1; i >= (int)lowest_idx; --i) {
        // if the qubit meets condition, skip it
        if (is_matched(qt.p2l[i])) continue;

        // if the qubit does not meet condition, find a matched qubit
        for (; j >= 0; --j) {
            if (is_matched(qt.p2l[j])) break;
        }

        // swap qubits if a matched qubit is found
        if (j >= 0) {
            num_inserted_gates += qt.add_swap_gate(
                circuit, j, i, 1, gate_insertion_pos + num_inserted_gates);
        } else {
            throw std::runtime_error(
                "Error: "
                "QuantumCircuitOptimizer::move_matching_qubits_to_local_upper()"
                " "
                "no enougth matched qubits");
        }
    }
    return num_inserted_gates;
}

UINT QuantumCircuitOptimizer::rearrange_qubits(const UINT gate_idx,
    const std::unordered_set<UINT>& next_local_qubit, QubitTable& qt) {
    UINT num_inserted_gates = 0;

    std::unordered_set<UINT> cur_local_qubits(
        qt.p2l.cbegin(), qt.p2l.cbegin() + local_qc);
    std::unordered_set<UINT> import_qubits, exportable_qubits;

    // e.g.
    // qt.p2l = [0 4 3 1 2 7 6 5]
    // global_qc = 4, local_qc = 4
    //
    //  local  | global
    // -----------------
    // 0 4 3 1 | 2 7 6 5
    //
    // cur_local_qubits = {0, 1, 3 ,4}
    // next_local_qubits = {2, 4, 6}

    // import_qubits = next_local_qubit - cur_local_qubits
    // e.g. {2, 6} = {2, 4, 6} - {0, 1, 3 ,4}
    for (UINT i : next_local_qubit) {
        if (cur_local_qubits.count(i) == 0) {
            import_qubits.insert(i);
        }
    }

    // exportable_qubits = cur_local_qubits - next_local_qubit
    // e.g. {0, 1, 3} = {0, 1, 3 ,4} - {2, 4, 6}
    for (UINT i : cur_local_qubits) {
        if (next_local_qubit.count(i) == 0) {
            exportable_qubits.insert(i);
        }
    }

    LOG << "import qubits: " << to_string(import_qubits) << std::endl;
    LOG << "exportable qubits: " << to_string(exportable_qubits) << std::endl;

    // make index and width for swap
    // e.g. swap_global_idx_list = [2 6]
    //      swap_width_list      = [1 1]
    std::vector<UINT> swap_global_idx_list;
    std::vector<UINT> swap_width_list;
    for (UINT i = local_qc; i < circuit->qubit_count; i++) {
        if (import_qubits.count(qt.p2l[i])) {
            swap_global_idx_list.push_back(i);
            swap_width_list.push_back(1);
        }
    }

    // fuse swaps
    // e.g. swap_global_idx_list = [2 6] -> [2]
    //      swap_width_list      = [1 1] -> [3]
    // TODO: if fast qubit swapping between global qubits is implemented,
    //       global qubits should be rearranged to be contiguous.
    UINT total_swap_width = import_qubits.size();
    int extra_local_qc =
        (int)exportable_qubits.size() - (int)import_qubits.size();

    for (UINT i = 0; i < swap_global_idx_list.size(); i++) {
        for (UINT j = i + 1; j < swap_global_idx_list.size();) {
            int gap = (int)(swap_global_idx_list[j]) -
                      (swap_global_idx_list[i] + swap_width_list[i]);

            if (gap <= extra_local_qc) {
                swap_width_list[i] += swap_width_list[j] + gap;
                extra_local_qc -= gap;
                total_swap_width += gap;
                swap_global_idx_list.erase(swap_global_idx_list.begin() + j);
                swap_width_list.erase(swap_width_list.begin() + j);
            } else {
                j++;
            }
        }
    }

    LOG << "extra_local_qc = " << extra_local_qc << std::endl;

    UINT lowest_swap_idx = local_qc - total_swap_width;

    // gather target local qubits to the upper of local qubits
    // e.g.
    //            local  | global
    //           -----------------
    // before    0 4 3 1 | 2 7 6 5
    //            X
    // after     4 0 3 1 | 2 7 6 5  exportable_qubits={0,1,3}
    //                              are placed at local upper

    num_inserted_gates += move_matching_qubits_to_local_upper(
        lowest_swap_idx, qt,
        [&](UINT q) { return exportable_qubits.count(q) != 0; },
        gate_idx + num_inserted_gates);

    // add SWAP/FusedSWAP gates to exchange global qubits with local qubits
    // e.g.
    //            local  | global
    //           -----------------
    // before    4 0 3 1 | 2 7 6 5
    //                             ) FusedSWAP(1, 4, 3)
    // after     4 2 7 6 | 0 3 1 5   next_local_qubits={2,4,6} are placed local
    //
    UINT swap_local_idx = lowest_swap_idx;
    for (UINT i = 0; i < swap_global_idx_list.size(); i++) {
        UINT swap_global_idx = swap_global_idx_list[i];
        UINT swap_width = swap_width_list[i];

        num_inserted_gates += qt.add_swap_gate(circuit, swap_local_idx,
            swap_global_idx, swap_width, gate_idx + num_inserted_gates);
        swap_local_idx += swap_width;
    }

    LOG << "next: " << qt << std::endl;

    return num_inserted_gates;
}

void QuantumCircuitOptimizer::revert_qubit_order(QubitTable& qt) {
    // reference qubit order
    std::vector<UINT> ref_order(circuit->qubit_count);
    std::iota(ref_order.begin(), ref_order.end(), 0);

    // find the lowest and highest global qubits which needs to be rearranged
    // e.g.
    //          local  | global
    //         -----------------
    // ref     0 1 2 3 | 4 5 6 7
    // qt.p2l  0 3 5 1 | 4 6 7 2
    //                     |   `- diff_global_idx_highest
    //                     `--- diff_global_idx_lowest
    UINT diff_global_idx_lowest = 0;   // dummy value
    UINT diff_global_idx_highest = 0;  // dummy value
    UINT diff_global_qubits_count = 0;

    for (UINT i = local_qc; i < circuit->qubit_count; i++) {
        if (qt.p2l[i] != ref_order[i]) {
            if (diff_global_qubits_count == 0) {
                diff_global_idx_lowest = i;
            }
            diff_global_idx_highest = i;
            diff_global_qubits_count++;
        }
    }

    if (diff_global_qubits_count > 0) {
        // reorder global qubits

        UINT swap_global_idx = diff_global_idx_lowest;
        UINT swap_width = diff_global_idx_highest - diff_global_idx_lowest + 1;
        UINT swap_local_idx = local_qc - swap_width;
        // find the lowest and highest global qubits which needs to be
        // rearranged e.g.
        //          local  | global
        //         -----------------
        // ref     0 1 2 3 | 4 5 6 7
        // qt.p2l  0 3 5 1 | 4 6 7 2
        //           |         `- swap_global_idx   swap_width=3
        //           `- swap_local_idx

        // check if the global qubits that require reordering include qubits
        // corresponding to the reference qubits or not. e.g.
        //          local  | global
        //         -----------------
        // ref     0 1 2 3 | 4[5 6 7]  corr_ref_qubits = {5,6,7}
        // qt.p2l  0 3 5 1 | 4[6 2 7]  global_qubits_to_reorder = {2,6,7}
        //                             => corr_ref_count = 1
        std::unordered_set<UINT> corr_ref_qubits(
            ref_order.cbegin() + swap_global_idx,
            ref_order.cbegin() + swap_global_idx + swap_width);
        UINT corr_ref_count = std::count_if(qt.p2l.cbegin() + swap_global_idx,
            qt.p2l.cbegin() + swap_global_idx + swap_width,
            [&](UINT q) { return corr_ref_qubits.count(q); });

        // check whether we can reorder qubits in a maximum of 2 FusedSWAPs or
        // not
        if (corr_ref_count == 0 ||
            (UINT)(swap_width * 2 - corr_ref_count) <= local_qc) {
            if (corr_ref_count > 0) {
                // if global_qubits_to_reorder contains corr_ref_qubits,
                // rearrange qubits at swap_local_idx ~ (local_qc-1) not to
                // contain corr_ref_qubits e.g.
                //          local  | global
                //         -----------------
                // ref     0 1 2 3 | 4 5 6 7  corr_ref_qubits = {5,6,7}
                // qt.p2l  0[3 5 1]| 4 6 2 7  [3 5 1] has 5 in corr_ref_quits
                //          \ /
                //           X
                //          / \               #
                // qt.p2l  5[3 0 1]| 4 6 2 7  [3 0 1] has no index in
                //                            corr_ref_quits
                move_matching_qubits_to_local_upper(
                    local_qc - swap_width, qt,
                    [&](UINT q) { return corr_ref_qubits.count(q) == 0; },
                    circuit->qubit_count /*end of circuit*/);

                // and then fused-swap between local and global qubits
                // e.g.
                //          local  | global
                //         -----------------
                // ref     0 1 2 3 | 4 5 6 7
                // qt.p2l  5[3 0 1]| 4[6 2 7]
                //            \ \ \   / / /
                //             ===========
                //            / / /   \ \ \    #
                // qt.p2l  5[6 2 7]| 4[3 0 1]
                qt.add_swap_gate(
                    circuit, swap_local_idx, swap_global_idx, swap_width);
            }

            // in local qubits, reorder the qubits to be swapped
            // e.g.
            //          local  | global
            //         -----------------
            // ref     0 1 2 3 | 4 5 6 7
            // qt.p2l  5[6 2 7]| 4 3 0 1
            //          \\/
            //          /\\               #
            // qt.p2l  2[5 6 7]| 4 3 0 1
            for (UINT i = 0; i < swap_width; i++) {
                qt.add_swap_gate(circuit, swap_local_idx + i,
                    qt.l2p[ref_order[swap_global_idx + i]], 1);
            }

            // and then, fused-swap between local and global qubits
            // e.g.
            //          local  | global
            //         -----------------
            // ref     0 1 2 3 | 4 5 6 7
            // qt.p2l  2[5 6 7]| 4[3 0 1]
            //            \ \ \   / / /
            //             ===========
            //            / / /   \ \ \    #
            // qt.p2l  2[3 0 1]| 4[5 6 7]
            qt.add_swap_gate(
                circuit, swap_local_idx, swap_global_idx, swap_width);
        } else {
            // naively reorder qubits one by one
            for (UINT i = local_qc; i < circuit->qubit_count; i++) {
                qt.add_swap_gate(circuit, i, qt.l2p[ref_order[i]], 1);
            }
        }
    }

    // reorder local qubits one by one
    // e.g.
    //          local  | global
    //         -----------------
    // ref     0 1 2 3 | 4 5 6 7
    // qt.p2l  2 3 0 1 | 4 5 6 7
    //          \ X /
    //           X X
    //          / X \             #
    // qt.p2l  0 1 2 3 | 4 5 6 7
    for (UINT i = 0; i < local_qc; i++) {
        qt.add_swap_gate(circuit, i, qt.l2p[ref_order[i]], 1);
    }
}

void QuantumCircuitOptimizer::insert_swap_gates(const UINT level) {
    if (level == 0) {
        return;
    }

    const bool gate_reordering_enabled = (level >= 2);

    if (level > 2) {
        throw std::invalid_argument(
            "Error: "
            "QuantumCircuit::QuantumCircuitOptimizer::insert_swap_gates(level) "
            ": invalid level. "
            "'0' means adding no SWAP/FusedSWAP, "
            "'1' means adding SWAP/FusedSWAP, "
            "'2' means adding SWAP/FusedSWAP with gate reordering");
    }

    if (global_qc == 0 && mpirank == 0) {
        std::cerr
            << "Warning: "
               "QuantumCircuit::QuantumCircuitOptimizer::insert_swap_gates("
               "level) "
               ": insert_swap is no effect for non-distributed state vector"
            << std::endl;
    }

    // warn if circuit has SWAP/FusedSWAP gates
    bool has_swap = std::any_of(
        circuit->gate_list.cbegin(), circuit->gate_list.cend(), is_swap_gate);
    if (has_swap) {
        std::cerr
            << "Warning: "
               "QuantumCircuit::QuantumCircuitOptimizer::insert_swap_gates("
               "level) "
               ": given circuit already has SWAP and/or FusedSWAP gates. "
               "Inserting SWAP gates may increase the amount of communication."
            << std::endl;
    }

    std::chrono::system_clock::time_point t_begin_depanalysis, t_begin_swapadd,
        t_end;

    // dependency analysis
    t_begin_depanalysis = std::chrono::system_clock::now();
    auto graph =
        gate_reordering_enabled ? build_dep_graph(circuit) : DepGraph();
    auto parent_gate_map = build_parent_gate_map(graph);

    if (log_enabled) {
        std::ofstream file("circuit_dep.dot");
        write_dot(file, graph);
    }

    t_begin_swapadd = std::chrono::system_clock::now();
    UINT num_gates = circuit->gate_list.size();
    QubitTable qt(circuit->qubit_count);

    GateReplacer replacer;
    std::unordered_set<const QuantumGateBase*> processed_gates;

    // add SWAP/FusedSWAP gates
    for (UINT gate_idx = 0; gate_idx < num_gates; gate_idx++) {
        LOG << "processing gate #" << gate_idx << std::endl;
        if (needs_communication(gate_idx, qt, replacer)) {
            LOG << "cur: " << qt << std::endl;
            if (gate_reordering_enabled) {
                gate_idx += move_gates_without_communication(
                    gate_idx, qt, parent_gate_map, processed_gates, replacer);
            }
            std::unordered_set<UINT> next_local_qubit =
                find_next_local_qubits(gate_idx, replacer);
            const UINT num_inserted_gates =
                rearrange_qubits(gate_idx, next_local_qubit, qt);
            gate_idx += num_inserted_gates;
            num_gates += num_inserted_gates;
        }
        LOG << "rewrite_gate_qubit_indexes #" << gate_idx << std::endl;
        auto g = circuit->gate_list[gate_idx];
        auto g_rewrited = qt.rewrite_gate_qubit_indexes(g);
        replacer.set_replaced_gate(g, g_rewrited);
        processed_gates.insert(g);
    }

    // replace gates
    for (UINT gate_idx = 0; gate_idx < circuit->gate_list.size(); gate_idx++) {
        auto g = circuit->gate_list[gate_idx];
        auto g_replaced = replacer.get_replaced_gate(g);
        if (g != g_replaced) {
            circuit->remove_gate(gate_idx);
            circuit->add_gate(g_replaced, gate_idx);
        }
    }

    // reorder qubits to the original order
    revert_qubit_order(qt);

    t_end = std::chrono::system_clock::now();
    double t_depanalysis =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            t_begin_swapadd - t_begin_depanalysis)
            .count() /
        1000.0;
    double t_swapadd = std::chrono::duration_cast<std::chrono::milliseconds>(
                           t_end - t_begin_swapadd)
                           .count() /
                       1000.0;

    LOG << "time for dep_analysis [s] " << t_depanalysis << std::endl;
    LOG << "time for swap_add [s] " << t_swapadd << std::endl;
}
