#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "utility.hpp"

#include <cctype>

#include "exception.hpp"

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

namespace json {
std::string trim(const std::string& str) {
    UINT st = 0;
    while (st < str.size() && std::isspace(str[st])) st++;
    if (st == str.size()) return "";
    UINT ed = str.size();
    while (ed > 0 && std::isspace(str[ed - 1])) ed--;
    return str.substr(st, ed - st);
}
std::vector<std::string> split_by_comma(const std::string& json) {
    std::vector<char> stk;
    stk.push_back('_');  // dummy
    std::vector<std::string> res;
    std::string tmp = "";
    for (UINT i = 0; i < json.size(); i++) {
        if (json[i] == ',' && stk.back() == '_') {
            std::string trimmed = trim(tmp);
            if (trimmed == "") {
                throw InvalidJSONFormatException(
                    "Empty component of array is found");
            }
            res.push_back(trimmed);
            tmp = "";
            continue;
        }
        if (stk.back() == '\\') {
            stk.pop_back();
        } else if (stk.back() == '\"') {
            if (json[i] == '\\') {
                stk.push_back('\\');
            } else if (json[i] == '\"') {
                stk.pop_back();
            }
        } else {
            if (json[i] == '[') {
                stk.push_back('[');
            } else if (json[i] == '{') {
                stk.push_back('{');
            } else if (json[i] == '\"') {
                stk.push_back('\"');
            } else if (json[i] == ']') {
                if (stk.back() != '[') {
                    throw InvalidJSONFormatException(
                        "beggining bracket \'[\' is expected");
                }
                stk.pop_back();
            } else if (json[i] == '}') {
                if (stk.back() != '{') {
                    throw InvalidJSONFormatException(
                        "beggining brace \'{\' is expected");
                }
                stk.pop_back();
            }
        }
        tmp += json[i];
    }
    tmp = trim(tmp);
    if (tmp != "") res.push_back(tmp);
    return res;
}
std::string escape_string(const std::string& str) {
    std::stringstream ss;
    for (UINT i = 0; i < str.size(); i++) {
        if (str[i] == '\"')
            ss << "\\\"";
        else if (str[i] == '\\')
            ss << "\\\\";
        else if (str[i] == '/')
            ss << "\\/";
        else if (str[i] == '\b')
            ss << "\\b";
        else if (str[i] == '\f')
            ss << "\\f";
        else if (str[i] == '\n')
            ss << "\\n";
        else if (str[i] == '\r')
            ss << "\\r";
        else if (str[i] == '\t')
            ss << "\\t";
        else
            ss << str[i];
    }
    return ss.str();
}
std::string unescape_string(const std::string& str) {
    std::stringstream ss;
    for (UINT i = 0; i < str.size(); i++) {
        if (str.substr(i, 2) == "\\\"") {
            ss << '\"';
            i++;
        } else if (str.substr(i, 2) == "\\\\") {
            ss << '\\';
            i++;
        } else if (str.substr(i, 2) == "\\/") {
            ss << '/';
            i++;
        } else if (str.substr(i, 2) == "\\b") {
            ss << '\b';
            i++;
        } else if (str.substr(i, 2) == "\\f") {
            ss << '\f';
            i++;
        } else if (str.substr(i, 2) == "\\n") {
            ss << '\n';
            i++;
        } else if (str.substr(i, 2) == "\\r") {
            ss << '\r';
            i++;
        } else if (str.substr(i, 2) == "\\t") {
            ss << '\t';
            i++;
        } else {
            if (str[i] == '\"') {
                throw InvalidJSONFormatException(
                    "Double quatation \'\"\' is expected to be escaped");
            }
            if (str[i] == '\\') {
                throw InvalidJSONFormatException(
                    "Backslash \'\\\' is expected to be escaped");
            }
            if (str[i] == '/') {
                throw InvalidJSONFormatException(
                    "Slash \'/\' is expected to be escaped");
            }
            ss << str[i];
        }
    }
    return ss.str();
}

std::string to_json(std::string x) { return "\"" + escape_string(x) + "\""; }
std::string to_json(const char* x) { return "\"" + escape_string(x) + "\""; }
std::string to_json(UINT x) { return std::to_string(x); }
std::string to_json(ITYPE x) { return std::to_string(x); }
std::string to_json(bool x) { return x ? "true" : "false"; }
std::string to_json(double x) {
    std::stringstream ss;
    ss << std::setprecision(20) << x;
    return ss.str();
}
std::string to_json(const std::vector<std::string>& x) {
    std::stringstream ss;
    ss << "[";
    for (UINT i = 0; i < x.size(); i++) {
        if (i) ss << ",";
        ss << x[i];
    }
    ss << "]";
    return ss.str();
}
std::string to_json(const std::map<std::string, std::string>& attributes) {
    std::stringstream ss;
    bool first_attribute = true;
    ss << "{";
    for (const auto& attribute : attributes) {
        if (!first_attribute) ss << ",";
        first_attribute = false;
        ss << to_json(attribute.first) << ":" << attribute.second;
    }
    ss << "}";
    return ss.str();
}
std::string to_json(CPPCTYPE x) {
    std::map<std::string, std::string> attributes;
    attributes["real"] = to_json(x.real());
    attributes["imag"] = to_json(x.imag());
    return to_json(attributes);
}
std::string to_json(const std::vector<UINT>& x) {
    std::vector<std::string> json_strings;
    std::transform(x.begin(), x.end(), std::back_inserter(json_strings),
        [](UINT x) { return to_json(x); });
    return to_json(json_strings);
}
std::string to_json(const std::vector<CPPCTYPE>& x) {
    std::vector<std::string> json_strings;
    std::transform(x.begin(), x.end(), std::back_inserter(json_strings),
        [](CPPCTYPE x) { return to_json(x); });
    return to_json(json_strings);
}

ITYPE uint_from_json(const std::string& json) {
    std::string trimmed = trim(json);
    ITYPE res = 0;
    for (char c : trimmed) {
        if (!isdigit(c)) {
            throw InvalidJSONFormatException(
                "unsigned integer is expected, but given compnent contains "
                "non-digit character");
        }
        res *= 10;
        res += c - '0';
    }
    return res;
}
bool bool_from_json(const std::string& json) {
    std::string trimmed = trim(json);
    if (trimmed == "true") return true;
    if (trimmed == "false") return false;
    throw InvalidJSONFormatException(
        "boolean is expected, but given component is neither \'true\' not "
        "\'false\'");
}
double real_from_json(const std::string& json) {
    std::string trimmed = trim(json);
    UINT point = trimmed.find('.');
    bool minus = trimmed[0] == '-';
    if (point == trimmed.size()) {
        if (minus) {
            return -(double)uint_from_json(trimmed.substr(1));
        } else {
            return uint_from_json(trimmed);
        }
    }
    double res = uint_from_json(trimmed.substr(minus, point - minus));
    double unit = 1.;
    for (UINT i = point + 1; i < trimmed.size(); i++) {
        if (!isdigit(trimmed[i])) {
            throw InvalidJSONFormatException(
                "real number is expected, but given component contains "
                "non-digit character except one point \'.\'");
        }
        unit *= .1;
        res += unit * (trimmed[i] - '0');
    }
    return minus ? -res : res;
}
std::string string_from_json(const std::string& json) {
    std::string trimmed = trim(json);
    if (trimmed.size() < 2 || trimmed.front() != '\"' ||
        trimmed.back() != '\"') {
        throw InvalidJSONFormatException(
            "string is expected, but given component is not surrounded by "
            "double quatation \'\"\"\'");
    }
    return unescape_string(trimmed.substr(1, trimmed.size() - 2));
}
std::vector<std::string> array_from_json(const std::string& json) {
    std::string trimmed = trim(json);
    if (trimmed.size() < 2 || trimmed.front() != '[' || trimmed.back() != ']') {
        throw InvalidJSONFormatException(
            "array is expected, but given component is not surrounded by "
            "bracket \'[]\'");
    }
    return split_by_comma(trimmed.substr(1, trimmed.size() - 2));
}
std::map<std::string, std::string> object_from_json(const std::string& json) {
    std::string trimmed = trim(json);
    if (trimmed.size() < 2 || trimmed.front() != '{' || trimmed.back() != '}') {
        throw InvalidJSONFormatException(
            "object is expected, but given component is not surrounded by "
            "brace \'{}\'");
    }
    std::vector<std::string> components =
        split_by_comma(trimmed.substr(1, trimmed.size() - 2));
    std::map<std::string, std::string> res;
    for (auto& component : components) {
        std::vector<char> stk;
        stk.push_back('_');  // dummy
        int idx = 0;
        for (; idx < component.size(); idx++) {
            if (stk.back() == '\\')
                stk.pop_back();
            else if (stk.back() == '\"' && component[idx] == '\\')
                stk.push_back('\\');
            else if (stk.back() == '\"' && component[idx] == '\"')
                stk.pop_back();
            else if (stk.back() == '_' && component[idx] == ':')
                break;
        }
        if (idx == component.size()) {
            throw InvalidJSONFormatException(
                "object is expected, but colon \':\' is not found in "
                "one of comma-separated components");
        }
        res[string_from_json(component.substr(0, idx))] =
            trim(component.substr(idx + 1));
    }
    return res;
}
CPPCTYPE complex_from_json(const std::string& json) {
    auto attributes = object_from_json(json);
    auto real = attributes.find("real");
    auto imag = attributes.find("imag");
    if (real == attributes.end()) {
        throw InvalidJSONFormatException(
            "complex is expected, but an attribute named \'real\' is not "
            "found");
    }
    if (imag == attributes.end()) {
        throw InvalidJSONFormatException(
            "complex is expected, but an attribute named \'imag\' is not "
            "found");
    }
    return CPPCTYPE(real_from_json(real->second), real_from_json(imag->second));
}
std::vector<ITYPE> uint_array_from_json(const std::string& json) {
    std::vector<std::string> jsons = array_from_json(json);
    std::vector<ITYPE> uints;
    std::transform(jsons.begin(), jsons.end(), std::back_inserter(uints),
        [](std::string json) { return uint_from_json(json); });
    return uints;
}
std::vector<CPPCTYPE> complex_array_from_json(const std::string& json) {
    std::vector<std::string> jsons = array_from_json(json);
    std::vector<CPPCTYPE> complexs;
    std::transform(jsons.begin(), jsons.end(), std::back_inserter(complexs),
        [](std::string json) { return complex_from_json(json); });
    return complexs;
}
}  // namespace json