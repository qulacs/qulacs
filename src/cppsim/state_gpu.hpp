#pragma once

#include "exception.hpp"
#include "state.hpp"

#ifdef _USE_GPU

#include <gpusim/memory_ops.h>
#include <gpusim/stat_ops.h>
#include <gpusim/update_ops_cuda.h>
#include <gpusim/util_func.h>

class QuantumStateGpu : public QuantumStateBase {
private:
    void* _state_vector;  // void* is assumed as GTYPE*
    Random random;

public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param qubit_count_ 量子ビット数
     */
    explicit QuantumStateGpu(UINT qubit_count_);

    QuantumStateGpu(UINT qubit_count_, UINT device_number_);

    /**
     * \~japanese-en デストラクタ
     */
    virtual ~QuantumStateGpu();

    /**
     * \~japanese-en 量子状態を計算基底の0状態に初期化する
     */
    virtual void set_zero_state() override;

    /**
     * \~japanese-en 量子状態を計算基底の0状態に初期化する
     * TODO: implement this
     */
    virtual void set_zero_norm_state() override;

    /**
     * \~japanese-en 量子状態を<code>comp_basis</code>の基底状態に初期化する
     *
     * @param comp_basis 初期化する基底を表す整数
     */
    virtual void set_computational_basis(ITYPE comp_basis) override;

    /**
     * \~japanese-en 量子状態をHaar
     * randomにサンプリングされた量子状態に初期化する
     */
    virtual void set_Haar_random_state() override;

    /**
     * \~japanese-en 量子状態をシードを用いてHaar
     * randomにサンプリングされた量子状態に初期化する
     */
    virtual void set_Haar_random_state(UINT seed) override;

    /**
     * \~japanese-en
     * <code>target_qubit_index</code>の添え字の量子ビットを測定した時、0が観測される確率を計算する。
     *
     * 量子状態は変更しない。
     * @param target_qubit_index
     * @return double
     */
    virtual double get_zero_probability(UINT target_qubit_index) const override;

    /**
     * \~japanese-en 複数の量子ビットを測定した時の周辺確率を計算する
     *
     * @param measured_values
     * 量子ビット数と同じ長さの0,1,2の配列。0,1はその値が観測され、2は測定をしないことを表す。
     * @return 計算された周辺確率
     */
    virtual double get_marginal_probability(
        std::vector<UINT> measured_values) const override;

    /**
     * \~japanese-en
     * 計算基底で測定した時得られる確率分布のエントロピーを計算する。
     *
     * @return エントロピー
     */
    virtual double get_entropy() const override;

    /**
     * \~japanese-en 量子状態のノルムを計算する
     *
     * 量子状態のノルムは非ユニタリなゲートを作用した時に小さくなる。
     * @return ノルム
     */
    virtual double get_squared_norm() const override;

    /**
     * \~japanese-en 量子状態のノルムを計算する
     *
     * 量子状態のノルムは非ユニタリなゲートを作用した時に小さくなる。
     * TODO: implement this as a single thread version.
     * @return ノルム
     */
    virtual double get_squared_norm_single_thread() const override;

    /**
     * \~japanese-en 量子状態を正規化する
     *
     * @param norm 自身のノルム
     */
    virtual void normalize(double squared_norm) override;

    /**
     * \~japanese-en 量子状態を正規化する
     *
     * @param norm 自身のノルム
     */
    virtual void normalize_single_thread(double squared_norm) override;

    /**
     * \~japanese-en バッファとして同じサイズの量子状態を作成する。
     *
     * @return 生成された量子状態
     */
    virtual QuantumStateBase* allocate_buffer() const override;

    /**
     * \~japanese-en 自身の状態のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual QuantumStateGpu* copy() const override;

    /**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
    virtual void load(const QuantumStateBase* _state) override;

    /**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
    virtual void load(const std::vector<CPPCTYPE>& _state) override;

    /**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
    virtual void load(const CPPCTYPE* _state) override;

    /**
     * \~japanese-en
     * 量子状態が配置されているメモリを保持するデバイス名を取得する。
     */
    virtual const std::string get_device_name() const override;

    /**
     * \~japanese-en
     * 量子状態をC++の<code>std::complex\<double\></code>の配列として取得する
     *
     * @return 複素ベクトルのポインタ
     */
    virtual CPPCTYPE* data_cpp() const override;

    /**
     * \~japanese-en
     * 量子状態をC++の<code>std::complex\<double\></code>の配列として取得する
     *
     * @return 複素ベクトルのポインタ
     */
    virtual CTYPE* data_c() const override;

    /**
     * \~japanese-en 量子状態をcsimのComplex型の配列として取得する
     *
     * @return 複素ベクトルのポインタ
     */
    virtual void* data() const override;

    virtual CTYPE* duplicate_data_c() const override;

    virtual CPPCTYPE* duplicate_data_cpp() const override;

    /**
     * \~japanese-en 量子状態を足しこむ
     */
    virtual void add_state(const QuantumStateBase* state) override;

    /**
     * \~japanese-en 量子状態を足しこむ (とりあえずの実装なので遅い)
     */
    virtual void add_state_with_coef(
        CPPCTYPE coef, const QuantumStateBase* state) override;

    /**
     * \~japanese-en 量子状態を足しこむ (とりあえずの実装なので遅い)
     */
    virtual void add_state_with_coef_single_thread(
        CPPCTYPE coef, const QuantumStateBase* state) override;

    /**
     * \~japanese-en 複素数をかける
     */
    virtual void multiply_coef(CPPCTYPE coef) override;

    virtual void multiply_elementwise_function(
        const std::function<CPPCTYPE(ITYPE)>& func) override;

    /**
     * \~japanese-en 量子状態を測定した際の計算基底のサンプリングを行う
     *
     * @param[in] sampling_count サンプリングを行う回数
     * @return サンプルされた値のリスト
     */
    virtual std::vector<ITYPE> sampling(UINT sampling_count) override;

    virtual std::vector<ITYPE> sampling(
        UINT sampling_count, UINT random_seed) override;

    virtual std::string to_string() const override;

    virtual boost::property_tree::ptree to_ptree() const override;
};

namespace state {
/**
 * \~japanese-en 量子状態間の内積を計算する
 *
 * @param[in] state_bra 内積のブラ側の量子状態
 * @param[in] state_ket 内積のケット側の量子状態
 * @return 内積の値
 */
CPPCTYPE DllExport inner_product(
    const QuantumStateGpu* state_bra, const QuantumStateGpu* state_ket);
}  // namespace state

#endif  // _USE_GPU
