#pragma once

#include <cppsim/circuit.hpp>

class QuantumGate_SingleParameter;

class DllExport ParametricQuantumCircuit : public QuantumCircuit {
private:
    std::vector<QuantumGate_SingleParameter*> _parametric_gate_list;
    std::vector<UINT> _parametric_gate_position;
public:
    ParametricQuantumCircuit(UINT qubit_count);
    ParametricQuantumCircuit(std::string qasm_path, std::string qasm_loader_script_path = "qasmloader.py");

	ParametricQuantumCircuit* copy() const;
	
	virtual void add_parametric_gate(QuantumGate_SingleParameter* gate);
    virtual void add_parametric_gate(QuantumGate_SingleParameter* gate, UINT index);
    virtual UINT get_parameter_count() const;
    virtual double get_parameter(UINT index) const;
    virtual void set_parameter(UINT index, double value);

    virtual UINT get_parametric_gate_position(UINT index) const;
    virtual void add_gate(QuantumGateBase* gate) override;
    virtual void add_gate(QuantumGateBase* gate, UINT index) override;
	virtual void add_gate_copy(const QuantumGateBase* gate) override;
	virtual void add_gate_copy(const QuantumGateBase* gate, UINT index) override;
	virtual void remove_gate(UINT index) override;

    virtual std::string to_string() const;
    friend DllExport std::ostream& operator<<(std::ostream& os, const ParametricQuantumCircuit&);
    friend DllExport std::ostream& operator<<(std::ostream& os, const ParametricQuantumCircuit* circuit);

    virtual void add_parametric_RX_gate(UINT target_index, double initial_angle);
    virtual void add_parametric_RY_gate(UINT target_index, double initial_angle);
    virtual void add_parametric_RZ_gate(UINT target_index, double initial_angle);
    virtual void add_parametric_multi_Pauli_rotation_gate(std::vector<UINT> target, std::vector<UINT> pauli_id, double initial_angle);

};





