
#pragma once

#include "type.hpp"

//! Flags for bit property: diagonal in X-basis
#define FLAG_X_COMMUTE      ((UINT)(0x01))
//! Flags for bit property: diagonal in Y-basis
#define FLAG_Y_COMMUTE      ((UINT)(0x02))
//! Flags for bit property: diagonal in Z-basis
#define FLAG_Z_COMMUTE      ((UINT)(0x04))

/**
 * \~japanese-en 量子ゲートが対象とする量子ビットの情報を保持するクラス
 */
class DllExport QubitInfo{
protected:
    UINT _index; /**< \~japanese-en 量子ビットの添え字 */
public:
    /**
     * \~japanese-en 量子ビットの添え字を返す
     * 
     * @return この量子ビットの添え字
     */
    UINT index() const { return _index; }

    /**
     * \~japanese-en コンストラクタ
     * 
     * @param index_ 量子ビットの添え字
     */
    QubitInfo(UINT index_): _index(index_) {};
};

class TargetQubitInfo;
class ControlQubitInfo;

/**
 * \~japanese-en コントロール量子ビットの情報を保持するクラス
 */
class DllExport ControlQubitInfo : public QubitInfo {
private:
    UINT _control_value; /**< \~japanese-en コントロール値　この量子ビットの添え字がコントロール値の時だけゲートが作用する。 */
public:
    /**
     * \~japanese-en コントロール値を取得する
     * 
     * @return コントロール値
     */
    UINT control_value() const { return _control_value;  }
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param index_ この量子ビットの添え字
     */
    ControlQubitInfo(UINT index_) : QubitInfo(index_), _control_value(1) {};
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param index_ この量子ビットの添え字
     * @param control_value_ この量子ビットのコントロール値
     */
    ControlQubitInfo(UINT index_, UINT control_value_) : QubitInfo(index_), _control_value(control_value_) {};

    /**
     * \~japanese-en ターゲット量子ビットの情報<code>info</code>と可換かどうかを調べる
     * 
     * @param info 可換かどうかを調べる量子ビットの情報
     * @return true 可換である
     * @return false 可換ではない
     */
    virtual bool is_commute_with(const TargetQubitInfo& info) const;
    /**
     * \~japanese-en コントロール量子ビットの情報<code>info</code>と可換かどうかを調べる
     * 
     * @param info 可換かどうかを調べる量子ビットの情報
     * @return true 可換である
     * @return false 可換ではない
     */
    virtual bool is_commute_with(const ControlQubitInfo& info) const;
};

/**
 * \~japanese-en ターゲット量子ビットの情報を保持するクラス
 */
class DllExport TargetQubitInfo : public QubitInfo {
private:
    UINT _commutation_property; /**< \~japanese-en この量子ビットがパウリ演算子と可換か同課を保持する値 */
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param index_ この量子ビットの添え字
     */
    TargetQubitInfo(UINT index_) : QubitInfo(index_), _commutation_property(0) {};
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param index_ この量子ビットの添え字
     * @param commutation_property_ この量子ビットのパウリとの交換関係
     */
    TargetQubitInfo(UINT index_, UINT commutation_property_) : QubitInfo(index_), _commutation_property(commutation_property_) {};
    /**
     * \~japanese-en Xパウリと可換かを調べる
     * 
     * @return true 可換である
     * @return false 非可換である
     */
    bool is_commute_X() const { return (_commutation_property & FLAG_X_COMMUTE) != 0; }
    /**
     * \~japanese-en Yパウリと可換かを調べる
     * 
     * @return true 可換である
     * @return false 非可換である
     */
    bool is_commute_Y() const { return (_commutation_property & FLAG_Y_COMMUTE) != 0; }
    /**
     * \~japanese-en Zパウリと可換かを調べる
     * 
     * @return true 可換である
     * @return false 非可換である
     */
    bool is_commute_Z() const { return (_commutation_property & FLAG_Z_COMMUTE) != 0; }
    /**
     * \~japanese-en ターゲット量子ビットの情報<code>info</code>と可換かどうかを調べる
     * 
     * @param info 可換かどうかを調べる量子ビットの情報
     * @return true 可換である
     * @return false 可換ではない
     */
    virtual bool is_commute_with(const TargetQubitInfo& info) const;
    /**
     * \~japanese-en コントロール量子ビットの情報<code>info</code>と可換かどうかを調べる
     * 
     * @param info 可換かどうかを調べる量子ビットの情報
     * @return true 可換である
     * @return false 可換ではない
     */
    virtual bool is_commute_with(const ControlQubitInfo& info) const;

    /**
     * \~japanese-en 与えられた<code>property</code>の値のパウリとの交換関係と自身をマージした時、得られるパウリ演算子との可換関係のプロパティ値を返す。
     * 
     * @param property マージするプロパティ値
     * @return マージされたプロパティ値
     */
    virtual UINT get_merged_property(UINT property) const { return _commutation_property & property; }
    /**
     * \~japanese-en 与えられた<code>target</code>の量子ビットの情報のパウリとの交換関係と自身をマージした時、得られるパウリ演算子との可換関係のプロパティ値を返す。
     * 
     * @param target マージする量子ビット情報
     * @return マージされたプロパティ値
     */
    virtual UINT get_merged_property(const TargetQubitInfo& target) const { return _commutation_property & target._commutation_property; }
    /**
     * \~japanese-en 与えられた<code>control</code>の量子ビットの情報のパウリとの交換関係と自身をマージした時、得られるパウリ演算子との可換関係のプロパティ値を返す。
     * 
     * @param control マージする量子ビット情報
     * @return マージされたプロパティ値
     */
    virtual UINT get_merged_property(const ControlQubitInfo& control) const { (void)control; return _commutation_property & FLAG_Z_COMMUTE; }
};
