
#pragma once

#include <csim/memory_ops_dm.hpp>
#include <csim/stat_ops_dm.hpp>
#include <csim/update_ops_dm.hpp>

#include "state.hpp"

class DensityMatrixCpu : public QuantumStateBase {
private:
    CPPCTYPE* _density_matrix;
    Random random;

public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param qubit_count_ 量子ビット数
     */
    explicit DensityMatrixCpu(UINT qubit_count_);

    /**
     * \~japanese-en デストラクタ
     */
    virtual ~DensityMatrixCpu();

    /**
     * \~japanese-en 量子状態を計算基底の0状態に初期化する
     */
    virtual void set_zero_state() override;

    /**
     * \~japanese-en ノルム0の状態 (すべての要素が0の行列にする)
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
    virtual DensityMatrixCpu* allocate_buffer() const override;

    /**
     * \~japanese-en 自身の状態のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual DensityMatrixCpu* copy() const override;

    /**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
    virtual void load(const QuantumStateBase* _state) override;

    /**
     * \~japanese-en <code>state</code>の量子状態を自身へコピーする。
     */
    virtual void load(const std::vector<CPPCTYPE>& _state) override;

    virtual void load(const Eigen::VectorXcd& _state);

    virtual void load(const ComplexMatrix& _state);

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
     * \~japanese-en 量子状態のポインタをvoid*型として返す
     */
    virtual void* data() const override;

    /**
     * \~japanese-en
     * 量子状態をC++の<code>std::complex\<double\></code>の配列として取得する
     *
     * @return 複素ベクトルのポインタ
     */
    virtual CPPCTYPE* data_cpp() const override;

    /**
     * \~japanese-en 量子状態をcsimのComplex型の配列として取得する
     *
     * @return 複素ベクトルのポインタ
     */
    virtual CTYPE* data_c() const override;

    virtual CTYPE* duplicate_data_c() const override;

    virtual CPPCTYPE* duplicate_data_cpp() const override;

    /**
     * \~japanese-en 量子状態を足しこむ
     */
    virtual void add_state(const QuantumStateBase* state) override;

    /**
     * \~japanese-en 量子状態を足しこむ
     */
    virtual void add_state_with_coef(
        CPPCTYPE coef, const QuantumStateBase* state) override;

    /**
     * \~japanese-en 量子状態を足しこむ
     * TODO: implement this in single_thread
     */
    virtual void add_state_with_coef_single_thread(
        CPPCTYPE coef, const QuantumStateBase* state) override;

    /**
     * \~japanese-en 複素数をかける
     */
    virtual void multiply_coef(CPPCTYPE coef) override;

    [[noreturn]] virtual void multiply_elementwise_function(
        const std::function<CPPCTYPE(ITYPE)>&) override;

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

using DensityMatrix = DensityMatrixCpu;

namespace state {
DllExport DensityMatrixCpu* tensor_product(
    const DensityMatrixCpu* state_bra, const DensityMatrixCpu* state_ket);
DllExport DensityMatrixCpu* permutate_qubit(
    const DensityMatrixCpu* state, std::vector<UINT> qubit_order);
DllExport DensityMatrixCpu* partial_trace(
    const QuantumStateCpu* state, std::vector<UINT> target);
DllExport DensityMatrixCpu* partial_trace(
    const DensityMatrixCpu* state, std::vector<UINT> target_traceout);
// create a mixed state such that the proportion of state1 is prob1, the
// proportion of state2 is prob2
DllExport DensityMatrixCpu* make_mixture(CPPCTYPE prob1,
    const QuantumStateBase* state1, CPPCTYPE prob2,
    const QuantumStateBase* state2);
DllExport QuantumStateBase* from_ptree(const boost::property_tree::ptree& pt);
}  // namespace state
