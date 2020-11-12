#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <mpi.h>
#include "cppsim/state.hpp"
#include "cppsim/gate_factory.hpp"
#include "cppsim/gate_merge.hpp"
#include "cppsim/circuit.hpp"

/**
 * \~japanese-en 回路にDepolarizingNoiseを加えてサンプリングするクラス
 */

class DllExport NoiseSimulatorMPI{
	private:
		QuantumCircuit *circuit;
		QuantumStateBase *initial_state;
		std::vector<std::pair<UINT,UINT>> noise_info;

		void evaluate_gates(const std::vector<UINT> chosen_gate, QuantumState *sampling_state, const int StartPos);
	public: 

		/**
		 * \~japanese-en
		 * コンストラクタ。
		 *
		 * NoiseSimulatorを作成する。
		 * @param[in] init_circuit  シミュレータに使用する量子回路。
		 * @param[in] Noise_itr ノイズを乗せ**ない**ゲートの先頭からの番号(0-indexed)のvector<UINT>。指定されなかった場合はすべてのゲートにノイズを乗せるものとする。
		 * @param[in] init_state 最初の状態。指定されなかった場合は0で初期化される。
		 * @return NoiseSimulatorのインスタンス
		 */
		NoiseSimulatorMPI(const QuantumCircuit *init_circuit,const QuantumState *init_state = NULL,const std::vector<UINT> *Noise_itr = NULL);
		/**
		 * \~japanese-en
		 * デストラクタ。このとき、NoiseSimulatorが保持しているcircuitとinitial_stateは解放される。
		 */
		virtual ~NoiseSimulatorMPI();

		/**
		 * \~japanese-en
		 * 
		 * サンプリングを行い、結果を配列で返す。
		 * @param[in] prob ノイズが乗る確率
		 * @param[in] sample_count 行うsamplingの回数
		 */
		virtual std::vector<UINT> execute(const UINT sample_count,const double prob);
};