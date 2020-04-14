
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
/**
 * \~english A class that holds the information of the qubit targeted by the quantum gate
 */
class DllExport QubitInfo{
protected:
    UINT _index; /**< \~japanese-en 量子ビットの添え字 */
    /**< \~english Index of qubit */
public:
    /**
     * \~japanese-en 量子ビットの添え字を返す
     * 
     * @return この量子ビットの添え字
     */
    /**
     * \~english Return index of qubit
     * 
     * @return Index
     */
    UINT index() const { return _index; }

    /**
     * \~japanese-en コンストラクタ
     * 
     * @param index_ 量子ビットの添え字
     */
    /**
     * \~english Construct
     * 
     * @param index_ Index of qubit
     */
    QubitInfo(UINT index_): _index(index_) {};
};

class TargetQubitInfo;
class ControlQubitInfo;

/**
 * \~japanese-en コントロール量子ビットの情報を保持するクラス
 */
/**
 * \~english A class that holds information of control qubit
 */
class DllExport ControlQubitInfo : public QubitInfo {
private:
    UINT _control_value; /**< \~japanese-en コントロール値　この量子ビットの添え字がコントロール値の時だけゲートが作用する。 */
/**< \~english Control value　The gate operates only when the index of this qubit equals control value */

public:
    /**
     * \~japanese-en コントロール値を取得する
     * 
     * @return コントロール値
     */
    /**
     * \~english Obtain control value
     * 
     * @return Control value
     */
    UINT control_value() const { return _control_value;  }
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param index_ この量子ビットの添え字
     */
    /**
     * \~english Constructor
     * 
     * @param index_ Index of this qubit
     */
    ControlQubitInfo(UINT index_) : QubitInfo(index_), _control_value(1) {};
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param index_ この量子ビットの添え字
     * @param control_value_ この量子ビットのコントロール値
     */
    /**
     * \~english Constructor
     * 
     * @param index_ Index of this qubit
     * @param control_value_ Control value of this qubit
     */
    ControlQubitInfo(UINT index_, UINT control_value_) : QubitInfo(index_), _control_value(control_value_) {};

    /**
     * \~japanese-en ターゲット量子ビットの情報<code>info</code>と可換かどうかを調べる
     * 
     * @param info 可換かどうかを調べる量子ビットの情報
     * @return true 可換である
     * @return false 可換ではない
     */
    /**
     * \~english Check if it is commutative with target qubit information <code>info</code>
     * 
     * @param info Information of qubit which needs to be checked whether it is commutative
     * @return true Commutative
     * @return false Not commutative
     */
    virtual bool is_commute_with(const TargetQubitInfo& info) const;
    /**
     * \~japanese-en コントロール量子ビットの情報<code>info</code>と可換かどうかを調べる
     * 
     * @param info 可換かどうかを調べる量子ビットの情報
     * @return true 可換である
     * @return false 可換ではない
     */
    /**
     * \~english Check if it is commutative with control qubit information <code>info</code>
     * 
     * @param info Information of qubit which needs to be checked whether it is commutative
     * @return true Commutative
     * @return false Not commutative
     */
    virtual bool is_commute_with(const ControlQubitInfo& info) const;
};

/**
 * \~japanese-en ターゲット量子ビットの情報を保持するクラス
 */
/**
 * \~english A class that holds information of target qubit
 */
class DllExport TargetQubitInfo : public QubitInfo {
private:
    UINT _commutation_property; /**< \~japanese-en この量子ビットがパウリ演算子と可換かどうかを保持する値 */
     /**< \~english A value that expresses whether this qubit is commutative with the Pauli operator */
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param index_ この量子ビットの添え字
     */
    /**
     * \~english Constructor
     * 
     * @param index_ Index of this qubit
     */
    TargetQubitInfo(UINT index_) : QubitInfo(index_), _commutation_property(0) {};
    /**
     * \~japanese-en コンストラクタ
     * 
     * @param index_ この量子ビットの添え字
     * @param commutation_property_ この量子ビットのパウリとの交換関係
     */
    /**
     * \~english Constructor
     * 
     * @param index_ Index of this qubit
     * @param commutation_property_ Exchange relation between this qubit and Pauli
     */
    TargetQubitInfo(UINT index_, UINT commutation_property_) : QubitInfo(index_), _commutation_property(commutation_property_) {};
    /**
     * \~japanese-en Xパウリと可換かを調べる
     * 
     * @return true 可換である
     * @return false 非可換である
     */
    /**
     * \~english Check if it is commutative with X-Pauli
     * 
     * @return true Commutative
     * @return false Not commutative
     */
    bool is_commute_X() const { return (_commutation_property & FLAG_X_COMMUTE) != 0; }
    /**
     * \~japanese-en Yパウリと可換かを調べる
     * 
     * @return true 可換である
     * @return false 非可換である
     */
    /**
     * \~english Check if it is commutative with Y-Pauli
     * 
     * @return true Commutative
     * @return false Not commutative
     */
    bool is_commute_Y() const { return (_commutation_property & FLAG_Y_COMMUTE) != 0; }
    /**
     * \~japanese-en Zパウリと可換かを調べる
     * 
     * @return true 可換である
     * @return false 非可換である
     */
    /**
     * \~english Check if it is commutative with Z-Pauli
     * 
     * @return true Commutative
     * @return false Not commutative
     */
    bool is_commute_Z() const { return (_commutation_property & FLAG_Z_COMMUTE) != 0; }
    /**
     * \~japanese-en ターゲット量子ビットの情報<code>info</code>と可換かどうかを調べる
     * 
     * @param info 可換かどうかを調べる量子ビットの情報
     * @return true 可換である
     * @return false 可換ではない
     */
    /**
     * \~english Check if it is commutative with target qubit information <code>info</code>
     * 
     * @param info Information of qubit which needs to be checked whether it is commutative
     * @return true Commutative
     * @return false Not commutative
     */
    virtual bool is_commute_with(const TargetQubitInfo& info) const;
    /**
     * \~japanese-en コントロール量子ビットの情報<code>info</code>と可換かどうかを調べる
     * 
     * @param info 可換かどうかを調べる量子ビットの情報
     * @return true 可換である
     * @return false 可換ではない
     */
    /**
     * \~english Check if it is commutative with control qubit information <code>info</code>
     * 
     * @param info Information of qubit which needs to be checked whether it is commutative
     * @return true Commutative
     * @return false Not commutative
     */
    virtual bool is_commute_with(const ControlQubitInfo& info) const;

    /**
     * \~japanese-en 与えられた<code>property</code>の値のパウリとの交換関係と自身をマージした時、得られるパウリ演算子との可換関係のプロパティ値を返す。
     * 
     * @param property マージするプロパティ値
     * @return マージされたプロパティ値
     */
    /**
     * \~english Returns the property value of the commutation relation with the Pauli operator obtained when merging itself with the commutation relation of Pauli with the given <code>property</code> value.
     * 
     * @param property Property value to merge
     * @return Property value merged
     */
    virtual UINT get_merged_property(UINT property) const { return _commutation_property & property; }
    /**
     * \~japanese-en 与えられた<code>target</code>の量子ビットの情報のパウリとの交換関係と自身をマージした時、得られるパウリ演算子との可換関係のプロパティ値を返す。
     * 
     * @param target マージする量子ビット情報
     * @return マージされたプロパティ値
     */
    /**
     * \~english Returns the property value of the commutation relation with the Pauli operator obtained when merging itself with the commutation relation of Pauli of qubit information with the given <code>target</code>.
     * 
     * @param property Property value to merge
     * @return Property value merged
     */
    virtual UINT get_merged_property(const TargetQubitInfo& target) const { return _commutation_property & target._commutation_property; }
    /**
     * \~japanese-en 与えられた<code>control</code>の量子ビットの情報のパウリとの交換関係と自身をマージした時、得られるパウリ演算子との可換関係のプロパティ値を返す。
     * 
     * @param control マージする量子ビット情報
     * @return マージされたプロパティ値
     */
    /**
     * \~english Returns the property value of the commutation relation with the Pauli operator obtained when merging itself with the commutation relation of Pauli of qubit information with the given <code>control</code>.
     * 
     * @param property Property value to merge
     * @return Property value merged
     */
    virtual UINT get_merged_property(const ControlQubitInfo& control) const { (void)control; return _commutation_property & FLAG_Z_COMMUTE; }
};
