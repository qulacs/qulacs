
#pragma once

#include <unordered_set>

#include "type.hpp"

class QuantumCircuit;
class QuantumGateBase;

/**
 * \~japanese-en 量子ビットの順序を管理するためのクラス
 */
class QubitTable {
private:
    UINT _qubit_count;
    std::vector<UINT> _p2l_table;
    std::vector<UINT> _l2p_table;
    QubitTable& operator=(const QubitTable&) = delete;

public:
    const std::vector<UINT>&
        p2l; /**< \~japanese-en 論理添え字への変換テーブル*/
    const std::vector<UINT>&
        l2p; /**< \~japanese-en 物理添え字への変換テーブル*/

    /**
     * \~japanese-en コンストラクタ
     *
     * @param[in] qubit_count 量子ビットの数
     */
    explicit QubitTable(UINT qubit_count);

    /**
     * \~japanese-en コンストラクタ
     *
     * @param[in] qt 量子ビットテーブル
     */
    QubitTable(const QubitTable& qt);

    /**
     * \~japanese-en 量子ビットテーブルを出力する
     *
     * @return 受け取ったストリーム
     */
    friend std::ostream& operator<<(std::ostream& os, const QubitTable& qt) {
        os << "qc:" << qt._qubit_count;
        os << ", p2l:[";
        for (UINT i : qt._p2l_table) {
            os << i << ",";
        }
        os << "], l2p[";
        for (UINT i : qt._l2p_table) {
            os << i << ",";
        }
        os << "]";
        return os;
    }

    /**
     * \~japanese-en 量子ゲートの量子ビットの添え字を書き換える
     *
     * @param[in] g 書き換える量子ゲート
     * @return 書き換え後のゲート
     */
    QuantumGateBase* rewrite_gate_qubit_indexes(QuantumGateBase* g) const;

    /**
     * \~japanese-en
     * 量子回路にSWAP/FusedSWAPゲートを追加し、量子ビットテーブルを更新する
     *
     * @param[in,out] circuit ゲートを追加する量子回路
     * @param[in] idx0 交換する量子ビットの物理開始添え字
     * @param[in] idx1 交換する量子ビットの物理開始添え字
     * @param[in] width 交換する量子ビットの幅
     * @return 追加した量子ゲート数
     */
    UINT add_swap_gate(
        QuantumCircuit* circuit, UINT idx0, UINT idx1, UINT width);

    /**
     * \~japanese-en
     * 量子回路にSWAP/FusedSWAPゲートを追加し、量子ビットテーブルを更新する
     *
     * @param[in,out] circuit ゲートを追加する量子回路
     * @param[in] idx0 交換する量子ビットの物理開始添え字
     * @param[in] idx1 交換する量子ビットの物理開始添え字
     * @param[in] width 交換する量子ビットの幅
     * @param[in] gate_pos ゲートを追加する位置
     * @return 追加した量子ゲート数
     */
    UINT add_swap_gate(QuantumCircuit* circuit, UINT idx0, UINT idx1,
        UINT width, UINT gate_pos);
};
