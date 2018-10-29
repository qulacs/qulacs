#pragma once

#include "type.hpp"

class QuantumCircuit;
class ParametricQuantumCircuit;

/**
 * \~japanese-en 与えられた量子ビットの数に応じて量子回路を生成するアンザッツのクラス
 */
class QuantumCircuitBuilder {
private:
public:
    /**
     * \~japanese-en 量子回路を生成する
     * 
     * @param[in] qubit_count 量子ビット数
     * @return 生成された量子回路
     */
    virtual QuantumCircuit* create_circuit(UINT qubit_count) const = 0;
};

