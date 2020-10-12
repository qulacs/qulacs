#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "state.hpp"
#include "gate_factory.hpp"
#include "gate_merge.hpp"
#include "circuit.hpp"

/**
 * \~japanese-en 回路にDepolarizingNoiseを加えてサンプリングするクラス
 */

class DllExport NoiseSimulator{
	private:
		QuantumCircuit *circuit;
		QuantumStateBase *initial_state;
		
	public: 

		/**
		 * \~japanese-en
		 * コンストラクタ。
		 *
		 * NoiseSimulatorを作成する。
		 * @param[in] init_circuit  
		 * @param[in] prob ノイズが乗る確率
		 * @param[in] init_state 最初の状態。指定されなかった場合は0で初期化される。
		 * @return NoiseSimulatorのインスタンス
		 */
		NoiseSimulator(const QuantumCircuit *init_circuit,const double prob,const QuantumState *init_state = NULL);

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
		virtual std::vector<UINT> execute(const UINT sample_count);
};