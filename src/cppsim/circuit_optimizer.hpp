
#pragma once

#include <unordered_set>

#include "csim/MPIutil.hpp"
#include "exception.hpp"
#include "type.hpp"

class QuantumCircuit;
class QuantumGateBase;
class QuantumGateMatrix;
class QubitTable;

/**
 * \~japanese-en 量子回路の圧縮を行うクラス
 *
 * 量子回路の圧縮を行う。
 * 与えらえた量子回路を適切なサイズまで圧縮したり、まとめたゲートに変換するなどの処理を行う。
 */
class DllExport QuantumCircuitOptimizer {
private:
    QuantumCircuit* circuit; /**< \~japanese-en 量子回路*/
    UINT local_qc;    /**< \~japanese-en ローカル量子ビットの数*/
    UINT global_qc;   /**< \~japanese-en グローバル量子ビットの数*/
    UINT mpisize;     /**< \~japanese-en MPIプロセスの数*/
    UINT mpirank;     /**< \~japanese-en MPIプロセスのランク*/
    bool log_enabled; /**< \~japanese-en ログ出力が有効かどうか*/
    UINT get_rightmost_commute_index(UINT gate_index);
    UINT get_leftmost_commute_index(UINT gate_index);
    UINT get_merged_gate_size(UINT gate_index1, UINT gate_index2);
    bool is_neighboring(UINT gate_index1, UINT gate_index2);

    ////////////////////////////////////////////////////////////
    // for swap insertion
    ////////////////////////////////////////////////////////////
    struct GateReplacer {
        std::map<QuantumGateBase*, QuantumGateBase*> replace;

        QuantumGateBase* get_replaced_gate(QuantumGateBase* gate) {
            if (!replace.count(gate)) return gate;
            return replace[gate] = get_replaced_gate(replace[gate]);
        }

        void set_replaced_gate(QuantumGateBase* from, QuantumGateBase* to) {
            if (replace.count(from)) {
                throw std::runtime_error(
                    "QuantumCircuitOptimizer::GateReplacer::set_replaced_gate: "
                    "The gate passed as `from` is already replaced to other "
                    "gate.");
            }
            replace[from] = to;
        }
    };
    void set_qubit_count(void);
    bool can_merge_with_swap_insertion(
        UINT gate_idx1, UINT gate_idx2, UINT swap_level);
    bool needs_communication(
        const UINT gate_index, const QubitTable& qt, GateReplacer& replacer);
    UINT move_gates_without_communication(const UINT gate_idx,
        const QubitTable& qt,
        const std::multimap<const QuantumGateBase*, const QuantumGateBase*>&
            dep_map,
        std::unordered_set<const QuantumGateBase*>& processed_gates,
        GateReplacer& replacer);
    std::unordered_set<UINT> find_next_local_qubits(
        const UINT start_gate_idx, GateReplacer& replacer);
    UINT move_matching_qubits_to_local_upper(UINT lowest_idx, QubitTable& qt,
        std::function<bool(UINT)> fn, UINT gate_insertion_pos);
    UINT rearrange_qubits(const UINT gate_idx,
        const std::unordered_set<UINT>& next_local_qubits, QubitTable& qt);
    void revert_qubit_order(QubitTable& qt);
    void insert_swap_gates(const UINT level);

public:
    /**
     * \~japanese-en コンストラクタ
     */
    QuantumCircuitOptimizer(UINT mpi_size = 0) {
#ifdef _USE_MPI
        MPIutil& mpiutil = MPIutil::get_inst();
        if (mpi_size == 0) {
            mpisize = mpiutil.get_size();
        } else {
            mpisize = mpi_size;
        }
        mpirank = mpiutil.get_rank();
#else
        if (mpi_size == 0) {
            mpisize = 1;
        } else {
            mpisize = mpi_size;
        }
        mpirank = 0;
#endif
        if ((mpisize & (mpisize - 1))) {
            throw MPISizeException(
                "Error: "
                "QuantumCircuitOptimizer::QuantumCircuitOptimizer(UINT): "
                "mpi_size must be power of 2");
        }

        log_enabled = false;
        if (const char* tmp = std::getenv("QULACS_OPTIMIZER_LOG")) {
            const UINT tmp_val = strtol(tmp, nullptr, 0);
            log_enabled = (tmp_val > 0);
        }
        // enable logging only on rank 0
        log_enabled = log_enabled && mpirank == 0;
    };

    /**
     * \~japanese-en デストラクタ
     */
    virtual ~QuantumCircuitOptimizer(){};

    /**
     * \~japanese-en 与えられた量子回路のゲートを指定されたブロックまで纏める。
     *
     * 与えられた量子回路において、若い添え字から任意の二つのゲートを選び、二つが他のゲートに影響を与えず合成可能なら合成を行う。
     * これを合成可能なペアがなくなるまで繰り返す。
     * 二つのゲートが合成可能であるとは、二つのゲートそれぞれについて隣接するゲートとの交換を繰り返し、二つのゲートが隣接した位置まで移動できることを指す。
     *
     * @param[in] circuit 量子回路のインスタンス
     * @param[in] max_block_size 合成後に許されるブロックの最大サイズ
     * @param[in] swap_level SWAP挿入による最適化レベル。0: SWAP追加なし、1:
     * SWAP追加、2: SWAP追加とゲート順序変更。
     */
    void optimize(
        QuantumCircuit* circuit, UINT max_block_size = 2, UINT swap_level = 0);

    /**
     * \~japanese-en 与えられた量子回路のゲートを指定されたブロックまで纏める。
     *
     * 与えられた量子回路において、若い添え字から任意の二つのゲートを選び、二つが他のゲートに影響を与えず合成可能なら合成を行う。
     * これを合成可能なペアがなくなるまで繰り返す。
     * 二つのゲートが合成可能であるとは、二つのゲートそれぞれについて隣接するゲートとの交換を繰り返し、二つのゲートが隣接した位置まで移動できることを指す。
     *
     * @param[in] circuit 量子回路のインスタンス
     * @param[in] swap_level SWAP挿入による最適化レベル。0: SWAP追加なし、1:
     * SWAP追加、2: SWAP追加とゲート順序変更。
     */
    void optimize_light(QuantumCircuit* circuit, UINT swap_level = 0);

    /**
     * \~japanese-en 量子回路を纏めて一つの巨大な量子ゲートにする
     *
     * @param[in] circuit 量子回路のインスタンス
     * @return 変換された量子ゲート
     */
    QuantumGateMatrix* merge_all(const QuantumCircuit* circuit);
};
