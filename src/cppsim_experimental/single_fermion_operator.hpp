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
    SingleFermionOperator();

    /**
     * \~japanese-en
     * コンストラクタ
     *
     * 作用する軌道の添字とラダー演算子からインスタンスを生成する。
     *
     * @param[in] target_index_list 作用する軌道の添字
     * @param[in] action_id_list
     * ラダー演算子を表す整数(生成が1、消滅が0に対応する)
     * 軌道2に対して生成、1に対して消滅は、([2 1], [1 0])と表現する。
     * @return 新しいインスタンス
     */
    SingleFermionOperator(const std::vector<UINT>& target_index_list,
        const std::vector<UINT>& action_id_list);

    /**
     * \~japanese-en
     * コンストラクタ
     *
     * 作用する軌道の添字とラダー演算子の文字列表記からインスタンスを生成する。
     *
     * @param[in] action_string
     * 軌道の添字と、生成を"^"、消滅を""と表現した文字列。
     * 軌道2に対して生成、1に対して消滅は、"2^ 1"と表現する。
     * @return 新しいインスタンス
     */
    SingleFermionOperator(std::string action_string);

    /**
     * 作用する軌道の添字のリストを取得する
     */
    const std::vector<UINT>& get_target_index_list() const;

    /**
     * 各軌道に作用させるactionのリストを取得する
     * 消滅: 0, 生成: 1に対応している
     */
    const std::vector<UINT>& get_action_id_list() const;

    SingleFermionOperator operator*(const SingleFermionOperator& target) const;

    SingleFermionOperator& operator*=(const SingleFermionOperator& target);

    std::string to_string() const;
};