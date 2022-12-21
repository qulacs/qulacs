#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "circuit.hpp"
#include "gate_factory.hpp"
#include "gate_merge.hpp"
#include "state.hpp"

/**
 * \~japanese-en 回路にDepolarizingNoiseを加えてサンプリングするクラス
 */

class DllExport NoiseSimulator {
private:
    Random random;
    QuantumCircuit* circuit;
    QuantumStateBase* initial_state;

    /**
     * \~japanese-en
     *
     * サンプリングのリクエストに関する構造体
     */
    struct SamplingRequest {
        /**
         * \~japanese-en
         * 1つのゲート内で複数のゲートのうちどれかが選ばれる時、どのゲートを選んでサンプリングすべきかを示す値のvector。
         */
        std::vector<UINT> gate_pos;
        /**
         * \~japanese-en
         *
         * サンプリング回数。
         */
        UINT num_of_sampling;
        SamplingRequest(
            std::vector<UINT> init_gate_pos, UINT init_num_of_sampling)
            : gate_pos(init_gate_pos), num_of_sampling(init_num_of_sampling) {}
    };

    void apply_gates(const std::vector<UINT>& chosen_gate,
        QuantumState* sampling_state, const int StartPos);

    /**
     * \~japanese-en
     *
     * サンプリングの回数だけを入力して、実際にどうゲートを適用してサンプリングするかの計画であるSamplingRequestのvectorを生成する関数。
     * @param[in] sample_count 行うサンプリングの回数
     */
    std::vector<SamplingRequest> generate_sampling_request(
        const UINT sample_count);

    /**
     * \~japanese-en
     *
     * ノイズゲートの場合、ノイズあり/なしで複数個のゲートのうちどれか一つが選ばれる。
     * どちらを選ぶかを決めて適用するゲートの番号を返す関数。
     * @param[in] gate 入力ゲート
     */
    UINT randomly_select_which_gate_pos_to_apply(QuantumGateBase* gate);

    /**
     * \~japanese-en
     *
     * SamplingRequestの計画通りにシミュレーションを行い、結果をpairの配列で返す。
     * @param[in] sampling_request_vector SamplingRequestのvector
     */
    std::vector<std::pair<QuantumState*, UINT>> simulate(
        std::vector<SamplingRequest> sampling_request_vector);

public:
    /**
     * \~japanese-en 複数回の実行結果をまとめた構造体
     */
    struct Result {
    public:
        std::vector<std::pair<QuantumState*, UINT>> result;

        Result(const std::vector<std::pair<QuantumState*, UINT>>& result_);
        ~Result();
        std::vector<ITYPE> sampling() const;
    };

    /**
     * \~japanese-en
     * コンストラクタ。
     *
     * NoiseSimulatorを作成する。
     * @param[in] init_circuit  シミュレータに使用する量子回路。
     * @param[in] init_state
     * 最初の状態。指定されなかった場合は|00...0>で初期化される。
     * @return NoiseSimulatorのインスタンス
     */
    explicit NoiseSimulator(const QuantumCircuit* init_circuit,
        const QuantumState* init_state = NULL);
    /**
     * \~japanese-en
     * デストラクタ。このとき、NoiseSimulatorが保持しているcircuitとinitial_stateは解放される。
     */
    virtual ~NoiseSimulator();

    /**
     * \~japanese-en
     *
     * サンプリングを行い、結果を配列で返す。
     * @param[in] sample_count 行うsamplingの回数
     */
    virtual std::vector<ITYPE> execute(const UINT sample_count);

    /**
     * \~japanese-en
     *
     * 実際にサンプリングまではせずにノイズがランダムにかかった量子状態の列を返す。
     * @param[in] execution_count 実行回数
     * @return
     * 量子状態の列。Resultクラスに入れられる。
     */
    virtual Result* execute_and_get_result(const UINT execution_count);
};
