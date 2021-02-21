#pragma once

#include <map>
#include <set>
#include <stdexcept>
#include <string>

#ifndef _MSC_VER
extern "C" {
#endif
#include <csim/constant.h>
#include <csim/update_ops.h>
#include <csim/update_ops_dm.h>
#ifndef _MSC_VER
}
#endif

#include <csim/update_ops_cpp.hpp>

#ifdef _USE_GPU
#include <gpusim/update_ops_cuda.h>
#endif

#include <cereal/access.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/complex.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include <cppsim_experimental/observable.hpp>
#include <cppsim_experimental/state.hpp>
#include <cppsim_experimental/type.hpp>

enum MapType {
    Basic,
    Sequence,
    Probabilistic,
    CPTP,
    Instrument,
};

enum GateMatrixType {
    DenseMatrix,
    DiagonalMatrix,
    SparseMatrix,
    PauliMatrix,
    PermutationMatrix
};

enum SpecialFuncType {
    None,
    GateI,
    GateX,
    GateY,
    GateZ,
    GateSqrtX,
    GateSqrtY,
    GateSqrtXdag,
    GateSqrtYdag,
    GateRX,
    GateRY,
    GateRZ,
    GateH,
    GateS,
    GateSdag,
    GateT,
    GateTdag,
    GateP0,
    GateP1,
    GateCX,
    GateCY,
    GateCZ,
    GateSWAP,
};

#define FLAG_CLIFFORD 0x01
#define FLAG_GAUSSIAN 0x02
#define FLAG_PARAMETRIC 0x04
#define FLAG_NOISE 0x8

#define FLAG_COMMUTE_X 0x01
#define FLAG_COMMUTE_Y 0x02
#define FLAG_COMMUTE_Z 0x04

class DllExport QuantumGateBase {
protected:
    MapType _map_type;
    std::map<std::string, double*> _parameter;
    explicit QuantumGateBase(MapType map_type) : _map_type(map_type) {};

public:
    QuantumGateBase() = default;
    virtual ~QuantumGateBase() {};

    // qubit function
    virtual UINT get_qubit_count() const = 0;
    virtual const std::vector<UINT> get_qubit_index_list() const = 0;
    virtual const std::vector<UINT> get_target_index_list() const = 0;
    virtual const std::vector<UINT> get_control_index_list() const = 0;
    virtual void reset_qubit_index_list(
        const std::vector<UINT>& src, const std::vector<UINT>& dst) = 0;

    // map function
    virtual void update_quantum_state(QuantumStateBase*) = 0;
    virtual MapType get_map_type() const { return _map_type; }
    virtual void get_matrix(ComplexMatrix& matrix) const = 0;
    virtual const std::vector<QuantumGateBase*>& get_kraus_list() const = 0;
    virtual std::vector<double> get_cumulative_distribution() const = 0;
    virtual void optimize_ProbabilisticGate() const {};

    // copy / serialize
    virtual QuantumGateBase* copy() const = 0;
    template <class Archive>
    void save(Archive& ar) const {
        std::vector<std::pair<std::string, double>> parameter_copy;
        for (auto x : _parameter) {
            parameter_copy.push_back(
                std::pair<std::string, double>(x.first, *(x.second)));
        }
        ar(CEREAL_NVP(parameter_copy), CEREAL_NVP(_map_type));
    }
    template <class Archive>
    void load(Archive& ar) {
        std::vector<std::pair<std::string, double>> parameter_copy;
        ar(CEREAL_NVP(parameter_copy), CEREAL_NVP(_map_type));
        for (auto x : parameter_copy) {
            (*(_parameter[x.first])) = x.second;
        }
    }
    virtual std::string dump_as_byte() const = 0;
    virtual void load_from_byte(std::string obj) = 0;


    virtual std::string to_string() const = 0;
    friend DllExport std::ostream& operator<<(
        std::ostream& os, const QuantumGateBase& gate) {
        os << gate.to_string();
        return os;
    }

    // parameters
    virtual UINT get_parameter_count() const {
        return (UINT)_parameter.size();
    }
    virtual bool has_parameter(std::string name) const {
        return _parameter.count(name);
    }
    virtual double get_parameter(std::string name) const {
        return *(_parameter.at(name));
    }
    virtual void set_parameter(std::string name, double value) {
        (*(_parameter[name])) = value;
    }
};
