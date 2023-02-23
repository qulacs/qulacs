#pragma once

#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/state.hpp>
class QuantumGate_SingleParameter;

class DllExport ParametricQuantumCircuit : public QuantumCircuit {
private:
    std::vector<QuantumGate_SingleParameter*> _parametric_gate_list;
    std::vector<UINT> _parametric_gate_position;

public:
    ParametricQuantumCircuit(UINT qubit_count);

    ParametricQuantumCircuit* copy() const;

    virtual void add_parametric_gate(QuantumGate_SingleParameter* gate);
    virtual void add_parametric_gate(
        QuantumGate_SingleParameter* gate, UINT index);
    virtual void add_parametric_gate_copy(QuantumGate_SingleParameter* gate);
    virtual void add_parametric_gate_copy(
        QuantumGate_SingleParameter* gate, UINT index);
    virtual UINT get_parameter_count() const;
    virtual double get_parameter(UINT index) const;
    virtual void set_parameter(UINT index, double value);

    virtual UINT get_parametric_gate_position(UINT index) const;
    virtual void add_gate(QuantumGateBase* gate) override;
    virtual void add_gate(QuantumGateBase* gate, UINT index) override;
    virtual void add_gate_copy(const QuantumGateBase* gate) override;
    virtual void add_gate_copy(
        const QuantumGateBase* gate, UINT index) override;
    virtual void remove_gate(UINT index) override;
    /**
     *  \~japanese-en 量子回路をマージする。
     *
     * 引数で与えた量子回路のゲートを後ろに追加していく。
     * マージされた側の量子回路に変更を加えてもマージした側の量子回路には変更は加わらないことに注意する。
     * パラメータゲートに対応するため、ParametricQuantumCircuit にも
     * merge_circuit() を追加する circuit1.add_circuit(circuit2)
     * circuit2.add_gate(gate) # これをしても、circuit1にgateは追加されない
     *
     * @param[in] circuit マージする量子回路
     */
    virtual void merge_circuit(const ParametricQuantumCircuit* circuit);
    virtual std::string to_string() const override;
    friend DllExport std::ostream& operator<<(
        std::ostream& os, const ParametricQuantumCircuit&);
    friend DllExport std::ostream& operator<<(
        std::ostream& os, const ParametricQuantumCircuit* circuit);

    virtual void add_parametric_RX_gate(
        UINT target_index, double initial_angle);
    virtual void add_parametric_RY_gate(
        UINT target_index, double initial_angle);
    virtual void add_parametric_RZ_gate(
        UINT target_index, double initial_angle);
    virtual void add_parametric_multi_Pauli_rotation_gate(
        std::vector<UINT> target, std::vector<UINT> pauli_id,
        double initial_angle);
    virtual std::vector<double> backprop(GeneralQuantumOperator* obs);
    virtual std::vector<double> backprop_inner_product(QuantumState* bistate);

    /**
     * \~japanese-en ptreeに変換
     *
     * @return ptree
     */
    virtual boost::property_tree::ptree to_ptree() const override;
};

namespace circuit {
/**
 * \~japanese-en ptreeからParametricQuantumCircuitを構築する
 */
ParametricQuantumCircuit* parametric_circuit_from_ptree(
    const boost::property_tree::ptree& pt);
}  // namespace circuit