#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "utility.hpp"

#include <cctype>

void get_Pauli_matrix(
    ComplexMatrix& matrix, const std::vector<UINT>& pauli_id_list) {
    ITYPE matrix_dim = 1ULL << pauli_id_list.size();
    matrix = ComplexMatrix::Zero(matrix_dim, matrix_dim);

    ITYPE flip_mask = 0;
    ITYPE phase_mask = 0;
    UINT rot90_count = 0;
    for (UINT ind = 0; ind < pauli_id_list.size(); ++ind) {
        UINT pauli_id = pauli_id_list[ind];
        if (pauli_id == 1) {
            flip_mask ^= 1ULL << ind;
        } else if (pauli_id == 2) {
            flip_mask ^= 1ULL << ind;
            phase_mask ^= 1ULL << ind;
            rot90_count++;
        } else if (pauli_id == 3) {
            phase_mask ^= 1ULL << ind;
        }
    }
    std::vector<CPPCTYPE> rot = {1, -1.i, -1, 1.i};
    for (ITYPE index = 0; index < matrix_dim; ++index) {
        double sign = 1. - 2. * (count_population_cpp(index & phase_mask) % 2);
        matrix(index, index ^ flip_mask) = rot[rot90_count % 4] * sign;
    }
}

std::vector<std::string> split(const std::string& s, const std::string& delim) {
    std::vector<std::string> elements;

    std::string item;

    for (char ch : s) {
        if (delim.find(ch) != std::string::npos) {
            if (!item.empty()) elements.push_back(item);
            item.clear();
        } else {
            item += ch;
        }
    }

    if (!item.empty()) elements.push_back(item);

    return elements;
}

void chfmt(std::string& ops) {
    for (UINT i = 0; i < ops.size(); ++i) {
        if (ops[i] == 'X' || ops[i] == 'Y' || ops[i] == 'Z' || ops[i] == 'I') {
            ops.insert(++i, " ");
        }
    }
}

std::tuple<double, double, std::string> parse_openfermion_line(
    std::string line) {
    double coef_real, coef_imag;

    char buf[256];
    UINT matches;

    if (line[0] == '(') {
        matches = std::sscanf(
            line.c_str(), "(%lf+%lfj) [%[^]]]", &coef_real, &coef_imag, buf);
        if (matches < 2) {
            matches = std::sscanf(line.c_str(), "(%lf-%lfj) [%[^]]]",
                &coef_real, &coef_imag, buf);
            coef_imag = -coef_imag;
        }
        if (matches < 3) {
            std::strcpy(buf, "I0");
        }
    } else {
        char symbol_j[1];
        matches = std::sscanf(
            line.c_str(), "%lf%[j] [%[^]]]", &coef_imag, symbol_j, buf);
        coef_real = 0.;
        if (matches < 3) {
            std::strcpy(buf, "I0");
        }
        if (symbol_j[0] != 'j') {
            matches = std::sscanf(line.c_str(), "%lf [%[^]]]", &coef_real, buf);
            coef_imag = 0.;
            if (matches < 2) {
                std::strcpy(buf, "I0");
            }
        }
        if (matches == 0) {
            return std::make_tuple(
                (double)NULL, (double)NULL, (std::string)NULL);
        }
    }

    std::string str_buf(buf, std::strlen(buf));
    chfmt(str_buf);

    return std::make_tuple(coef_real, coef_imag, str_buf);
}

bool check_is_unique_index_list(const std::vector<UINT>& index_list) {
    std::vector<UINT> index_list_sorted(index_list.begin(), index_list.end());
    sort(index_list_sorted.begin(), index_list_sorted.end());
    bool is_unique = true;
    for (UINT i = 0; i + 1 < index_list_sorted.size(); ++i) {
        is_unique &= (index_list_sorted[i] != index_list_sorted[i + 1]);
        if (!is_unique) break;
    }
    return is_unique;
}

std::string& rtrim(std::string& str) {
    auto it = std::find_if(str.rbegin(), str.rend(),
        [](unsigned char c) { return !std::isspace(c); });
    str.erase(it.base(), str.end());
    return str;
}

namespace ptree {
boost::property_tree::ptree to_ptree(const CPPCTYPE& cnum) {
    boost::property_tree::ptree ptree;
    ptree.put("real", cnum.real());
    ptree.put("imag", cnum.imag());
    return ptree;
}
boost::property_tree::ptree to_ptree(const std::vector<UINT>& uarray) {
    boost::property_tree::ptree ptree;
    for (const UINT& unum : uarray) {
        boost::property_tree::ptree child;
        child.put("", unum);
        ptree.push_back(std::make_pair("", child));
    }
    return ptree;
}
boost::property_tree::ptree to_ptree(const std::vector<CPPCTYPE>& carray) {
    boost::property_tree::ptree ptree;
    for (const CPPCTYPE& cnum : carray) {
        ptree.push_back(std::make_pair("", to_ptree(cnum)));
    }
    return ptree;
}
boost::property_tree::ptree to_ptree(
    const std::vector<boost::property_tree::ptree>& pt_array) {
    boost::property_tree::ptree ptree;
    for (const boost::property_tree::ptree& pt : pt_array) {
        ptree.push_back(std::make_pair("", pt));
    }
    return ptree;
}
boost::property_tree::ptree to_ptree(
    const std::vector<TargetQubitInfo>& target_qubit_list) {
    boost::property_tree::ptree pt;
    for (const TargetQubitInfo& info : target_qubit_list) {
        boost::property_tree::ptree child;
        child.put("", info.index());
        pt.push_back(std::make_pair("", child));
    }
    return pt;
}
boost::property_tree::ptree to_ptree(
    const std::vector<ControlQubitInfo>& control_qubit_list) {
    boost::property_tree::ptree pt;
    for (const ControlQubitInfo& info : control_qubit_list) {
        boost::property_tree::ptree child;
        child.put("index", info.index());
        child.put("value", info.control_value());
        pt.push_back(std::make_pair("", child));
    }
    return pt;
}
boost::property_tree::ptree to_ptree(const ComplexVector& vector) {
    boost::property_tree::ptree ptree;
    UINT sz = vector.size();
    ptree.put("size", sz);
    ptree.put_child("data",
        to_ptree(std::vector<CPPCTYPE>(vector.data(), vector.data() + sz)));
    return ptree;
}
boost::property_tree::ptree to_ptree(const ComplexMatrix& matrix) {
    boost::property_tree::ptree ptree;
    UINT r = matrix.rows(), c = matrix.cols();
    ptree.put("rows", r);
    ptree.put("cols", c);
    ptree.put_child("data", to_ptree(std::vector<CPPCTYPE>(
                                matrix.data(), matrix.data() + matrix.size())));
    return ptree;
}
boost::property_tree::ptree to_ptree(const SparseComplexMatrix& matrix) {
    boost::property_tree::ptree ptree;
    UINT r = matrix.rows(), c = matrix.cols();
    ptree.put("rows", r);
    ptree.put("cols", c);
    std::vector<boost::property_tree::ptree> data;
    for (UINT k = 0; k < matrix.outerSize(); ++k) {
        for (SparseComplexMatrix::InnerIterator it(matrix, k); it; ++it) {
            boost::property_tree::ptree child;
            child.put("row", it.row());
            child.put("col", it.col());
            child.put_child("value", to_ptree(it.value()));
            data.push_back(child);
        }
    }
    ptree.put_child("data", to_ptree(data));
    return ptree;
}
CPPCTYPE complex_from_ptree(const boost::property_tree::ptree& pt) {
    double real = pt.get<double>("real");
    double imag = pt.get<double>("imag");
    return CPPCTYPE(real, imag);
}
std::vector<UINT> uint_array_from_ptree(const boost::property_tree::ptree& pt) {
    std::vector<UINT> uarray;
    for (const boost::property_tree::ptree::value_type& unum_pair : pt) {
        uarray.push_back(unum_pair.second.get<UINT>(""));
    }
    return uarray;
}
std::vector<CPPCTYPE> complex_array_from_ptree(
    const boost::property_tree::ptree& pt) {
    std::vector<CPPCTYPE> carray;
    for (const boost::property_tree::ptree::value_type& cnum_pair : pt) {
        carray.push_back(complex_from_ptree(cnum_pair.second));
    }
    return carray;
}
std::vector<boost::property_tree::ptree> ptree_array_from_ptree(
    const boost::property_tree::ptree& pt) {
    std::vector<boost::property_tree::ptree> pt_array;
    for (const boost::property_tree::ptree::value_type& pt_pair : pt) {
        pt_array.push_back(pt_pair.second);
    }
    return pt_array;
}
std::vector<TargetQubitInfo> target_qubit_list_from_ptree(
    const boost::property_tree::ptree& pt) {
    std::vector<TargetQubitInfo> target_qubit_list;
    for (const boost::property_tree::ptree::value_type& target_qubit_pair :
        pt) {
        target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_pair.second.get<UINT>("")));
    }
    return target_qubit_list;
}
std::vector<ControlQubitInfo> control_qubit_list_from_ptree(
    const boost::property_tree::ptree& pt) {
    std::vector<ControlQubitInfo> control_qubit_list;
    for (const boost::property_tree::ptree::value_type& control_qubit_pair :
        pt) {
        boost::property_tree::ptree child = control_qubit_pair.second;
        control_qubit_list.push_back(ControlQubitInfo(
            child.get<UINT>("index"), child.get<UINT>("value")));
    }
    return control_qubit_list;
}
ComplexVector complex_vector_from_ptree(const boost::property_tree::ptree& pt) {
    UINT sz = pt.get<UINT>("size");
    ComplexVector vector(sz);
    std::vector<CPPCTYPE> data = complex_array_from_ptree(pt.get_child("data"));
    for (UINT i = 0; i < sz; i++) {
        vector(i) = data[i];
    }
    return vector;
}
ComplexMatrix complex_matrix_from_ptree(const boost::property_tree::ptree& pt) {
    UINT r = pt.get<UINT>("rows"), c = pt.get<UINT>("cols");
    ComplexMatrix matrix(r, c);
    std::vector<CPPCTYPE> data = complex_array_from_ptree(pt.get_child("data"));
    for (UINT i = 0; i < r; i++) {
        for (UINT j = 0; j < c; j++) {
            matrix(i, j) = data[i * c + j];
        }
    }
    return matrix;
}
SparseComplexMatrix sparse_complex_matrix_from_ptree(
    const boost::property_tree::ptree& pt) {
    UINT r = pt.get<UINT>("rows"), c = pt.get<UINT>("cols");
    SparseComplexMatrix matrix(r, c);
    std::vector<Eigen::Triplet<CPPCTYPE>> triplets;
    std::vector<boost::property_tree::ptree> data =
        ptree_array_from_ptree(pt.get_child("data"));
    for (const boost::property_tree::ptree& child : data) {
        UINT row = child.get<UINT>("row");
        UINT col = child.get<UINT>("col");
        CPPCTYPE value = complex_from_ptree(child.get_child("value"));
        triplets.emplace_back(row, col, value);
    }
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    return matrix;
}
std::string to_json(const boost::property_tree::ptree& pt) {
    std::stringstream ss;
    boost::property_tree::json_parser::write_json(ss, pt);
    return ss.str();
}
boost::property_tree::ptree from_json(const std::string& json) {
    std::stringstream ss(json);
    boost::property_tree::ptree pt;
    boost::property_tree::json_parser::read_json(ss, pt);
    return pt;
}
}  // namespace ptree