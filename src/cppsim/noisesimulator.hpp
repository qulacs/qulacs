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

    struct SamplingRequest {
        std::vector<UINT> gate_pos;
        UINT numOfSampling;
        SamplingRequest(
            std::vector<UINT> init_gate_pos, UINT init_num_of_sampling)
            : gate_pos(init_gate_pos), numOfSampling(init_num_of_sampling) {}
    };

    void apply_gates(const std::vector<UINT>& chosen_gate,
        QuantumState* sampling_state, const int StartPos);

    std::vector<SamplingRequest> create_sampling_request(
        const UINT sample_count);

    UINT randomly_select_which_gate_pos_to_apply(QuantumGateBase* gate);

    std::vector<ITYPE> execute_sampling(
        std::vector<SamplingRequest> sampling_request_vector);

public:
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
};
