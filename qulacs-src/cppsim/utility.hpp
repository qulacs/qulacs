/**
 * \~japanese-en ユーティリティの関数やクラス
 * 
 * @file utility.hpp
 */

#pragma once

#include <cstdio>
#include <chrono>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include "type.hpp"

/**
 * \~japanese-en 1になっているビットの数を数える
 * 
 * @param[in] x 1になっているビットの数を数える整数
 * @return 1になっているビットの数
 */
inline static UINT count_population_cpp(ITYPE x)
{
    x = ((x & 0xaaaaaaaaaaaaaaaaUL) >> 1)
        + (x & 0x5555555555555555UL);
    x = ((x & 0xccccccccccccccccUL) >> 2)
        + (x & 0x3333333333333333UL);
    x = ((x & 0xf0f0f0f0f0f0f0f0UL) >> 4)
        + (x & 0x0f0f0f0f0f0f0f0fUL);
    x = ((x & 0xff00ff00ff00ff00UL) >> 8)
        + (x & 0x00ff00ff00ff00ffUL);
    x = ((x & 0xffff0000ffff0000UL) >> 16)
        + (x & 0x0000ffff0000ffffUL);
    x = ((x & 0xffffffff00000000UL) >> 32)
        + (x & 0x00000000ffffffffUL);
    return (UINT)x;
}

/**
 * \~japanese-en <code>pauli_id_list</code>からパウリ行列を<code>matrix</code>生成する。
 * 
 * @param[out] matrix 作成する行列を格納する行列の参照
 * @param[in] pauli_id_list 生成するパウリ演算子のIDのリスト。\f${I,X,Y,Z}\f$が\f${0,1,2,3}\f$に対応する。
 */
void DllExport get_Pauli_matrix(ComplexMatrix& matrix, const std::vector<UINT>& pauli_id_list) ;

/**
 * \~japanese-en 乱数を管理するクラス
 */
class Random{
private:
    std::uniform_real_distribution<double> uniform_dist;
    std::normal_distribution<double> normal_dist;
    std::mt19937_64 mt;
public:
    /**
     * \~japanese-en コンストラクタ
     */
    Random(): uniform_dist(0,1), normal_dist(0,1){
        std::random_device rd;
        mt.seed(rd());
    }

    /**
     * \~japanese-en シードを設定する
     * 
     * @param seed シード値
     */
    void set_seed(uint64_t seed){
        mt.seed(seed);
    }
    /**
     * \~japanese-en \f$[0,1)\f$の一様分布から乱数を生成する
     * 
     * @return 生成された乱数
     */
    double uniform() {return uniform_dist(mt);}

    /**
     * \~japanese-en 期待値0、分散1の正規分から乱数を生成する
     * 
     * @return double 生成された乱数
     */
    double normal(){return normal_dist(mt);}

    /**
     * \~japanese-en 64bit整数の乱数を生成する
     * 
     * @return 生成された乱数
     */
    unsigned long long int64() { return mt(); }

    /**
     * \~japanese-en 32bit整数の乱数を生成する
     * 
     * @return 生成された乱数
     */
    unsigned long int32() { return mt() % ULONG_MAX; }
};

/**
 * \~japanese-en 時間計測用のユーティリティクラス
 * 
 * 一時停止を行うことで、必要な箇所だけの積算時間を計測する。
 */
class Timer{
private:
    std::chrono::system_clock::time_point last;
    long long stock;
    bool is_stop;
public:
    /**
     * \~japanese-en コンストラクタ
     */
    Timer(){
        reset();
        is_stop = false;
    }

    /**
     * \~japanese-en 時間計測をリセットする
     * 
     * 蓄積された時間を0にし、測定の起点となる時間を0にする。
     */
    void reset(){
        stock=0;
        last = std::chrono::system_clock::now();
    }

    /**
     * \~japanese-en 現在の経過時間を取得する
     * 
     * 経過時間を取得する。単位は秒で返される。一時停止を用いて時間を積算している場合は、積算している時間も併せた値が帰る。
     * @return 経過時間　単位は秒
     */
    double elapsed(){
        if (is_stop) return stock*1e-6;
        else {
            auto duration = std::chrono::system_clock::now() - last;
            return (stock + std::chrono::duration_cast<std::chrono::microseconds>(duration).count())*1e-6;
        }
    }

    /**
     * \~japanese-en タイマーを一時停止する
     * 
     * タイマーを一時停止し、現在までの経過時間を蓄積中の時間に加える。
     */
    void temporal_stop(){
        if (!is_stop) {
            auto duration = std::chrono::system_clock::now() - last;
            stock += std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
            is_stop = true;
        }
    }

    /**
     * \~japanese-en タイマーを再開する
     * 
     * タイマーを再開し、新たな時間計測のための起点を設定する。
     */
    void temporal_resume(){
        if (is_stop) {
            last = std::chrono::system_clock::now();
            is_stop = false;
        }
    }
};


/**
 * \~japanese-en 第一引数の文字列を第二引数の文字列で区切ったリストを返します。この関数は第一引数に対して非破壊です。
 * 
 * @param[in] s 分割したい文字列。
 * @param[in] delim 区切り文字列。
 * @return 第一引数に含まれる区切り文字列で区切った文字列。(example: split("aabcaaca", "bc") -> {"aa", "aa", "a"}
 */
DllExport std::vector<std::string> split(const std::string &s, const std::string &delim);

/**
 * \~japanese-en 引数にとった文字列を、PauliOperatorで渡すことができる形に整形します。この関数は第一引数に与えた文字列に破壊的な変更を加えます。
 * 
 * @param[in] ops 演算子と添字が隣り合った文字列。(example: "X12 Y13 Z5" に対して, "X 12 Y 13 Z 5"のように変更を加えます。)
 */
DllExport void chfmt(std::string& ops);


DllExport std::tuple<double, double, std::string> parse_openfermion_line(std::string line);

/**
 * \~japanese-en 配列の中に重複する添え字があるかをチェックする。
 *
 * @param[in] index_list チェックする配列
 * @return 重複がある場合にtrue、ない場合にfalse
 */
bool check_is_unique_index_list(std::vector<UINT> index_list);
