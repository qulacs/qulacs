/**
 * \~japanese-en 大体必要だと思われる ファイルを一括インクルードするファイル
 *
 * cppsimのほぼ全てとvqcsimのparametric_circuit関連をカバー。
 *
 * @file include_some.hpp
 */

#pragma once

#include <cppsim/circuit.hpp>
#include <cppsim/circuit_builder.hpp>
#include <cppsim/circuit_optimizer.hpp>
#include <cppsim/exception.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_general.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_matrix_diagonal.hpp>
#include <cppsim/gate_matrix_sparse.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_named.hpp>
#include <cppsim/gate_named_one.hpp>
#include <cppsim/gate_named_pauli.hpp>
#include <cppsim/gate_named_two.hpp>
#include <cppsim/gate_noisy_evolution.hpp>
#include <cppsim/gate_reflect.hpp>
#include <cppsim/gate_reversible.hpp>
#include <cppsim/general_quantum_operator.hpp>
#include <cppsim/noisesimulator.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/qubit_info.hpp>
#include <cppsim/simulator.hpp>
#include <cppsim/state.hpp>
#include <cppsim/state_dm.hpp>
#include <cppsim/state_gpu.hpp>
#include <cppsim/type.hpp>
#include <cppsim/utility.hpp>
#include <csim/constant.hpp>
#include <vqcsim/parametric_circuit.hpp>
#include <vqcsim/parametric_circuit_builder.hpp>
#include <vqcsim/parametric_gate.hpp>
#include <vqcsim/parametric_gate_factory.hpp>