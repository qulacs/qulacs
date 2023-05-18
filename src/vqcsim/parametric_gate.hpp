#pragma once

#include <cppsim/exception.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/state.hpp>
#include <cppsim/utility.hpp>
#include <csim/update_ops.hpp>
#include <csim/update_ops_dm.hpp>

#ifdef _USE_GPU
#include <gpusim/update_ops_cuda.h>
#endif

class QuantumGate_SingleParameter : public QuantumGateBase {
protected:
    double _angle;
    UINT _parameter_type;

public:
    QuantumGate_SingleParameter(double angle);
    virtual void set_parameter_value(double value);
    virtual double get_parameter_value() const;
    virtual QuantumGate_SingleParameter* copy() const override = 0;
};

class QuantumGate_SingleParameterOneQubitRotation
    : public QuantumGate_SingleParameter {
protected:
    using UpdateFunc = void (*)(UINT, double, CTYPE*, ITYPE);
    using UpdateFuncGpu = void (*)(UINT, double, void*, ITYPE, void*, UINT);
    UpdateFunc _update_func = nullptr;
    UpdateFunc _update_func_dm = nullptr;
    UpdateFuncGpu _update_func_gpu = nullptr;

    QuantumGate_SingleParameterOneQubitRotation(double angle);

public:
    virtual void update_quantum_state(QuantumStateBase* state) override;
};

class ClsParametricRXGate : public QuantumGate_SingleParameterOneQubitRotation {
public:
    ClsParametricRXGate(UINT target_qubit_index, double angle);

    virtual void set_matrix(ComplexMatrix& matrix) const override;
    virtual ClsParametricRXGate* copy() const override;
    virtual boost::property_tree::ptree to_ptree() const override;
    virtual ClsParametricRXGate* get_inverse() const override;
};

class ClsParametricRYGate : public QuantumGate_SingleParameterOneQubitRotation {
public:
    ClsParametricRYGate(UINT target_qubit_index, double angle);
    virtual void set_matrix(ComplexMatrix& matrix) const override;
    virtual ClsParametricRYGate* copy() const override;
    virtual boost::property_tree::ptree to_ptree() const override;
    virtual ClsParametricRYGate* get_inverse() const override;
};

class ClsParametricRZGate : public QuantumGate_SingleParameterOneQubitRotation {
public:
    ClsParametricRZGate(UINT target_qubit_index, double angle);
    virtual void set_matrix(ComplexMatrix& matrix) const override;
    virtual ClsParametricRZGate* copy() const override;
    virtual boost::property_tree::ptree to_ptree() const override;
    virtual ClsParametricRZGate* get_inverse() const override;
};

class ClsParametricPauliRotationGate : public QuantumGate_SingleParameter {
protected:
    PauliOperator* _pauli;

public:
    ClsParametricPauliRotationGate(double angle, PauliOperator* pauli);
    virtual ~ClsParametricPauliRotationGate();
    virtual void update_quantum_state(QuantumStateBase* state) override;
    virtual ClsParametricPauliRotationGate* copy() const override;
    virtual void set_matrix(ComplexMatrix& matrix) const override;

    virtual PauliOperator* get_pauli() const;
    virtual boost::property_tree::ptree to_ptree() const override;
    virtual ClsParametricPauliRotationGate* get_inverse() const override;
};
