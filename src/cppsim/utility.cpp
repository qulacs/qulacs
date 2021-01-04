#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "utility.hpp"

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

Observable* generate_random_observable(
    const UINT qubit_count, const UINT operator_count) {
    Observable* observable = new Observable(qubit_count);
    Random random;
    for (auto operator_index = 0; operator_index < operator_count;
         operator_index++) {
        auto target_qubit_index_list = std::vector<UINT>(qubit_count, 0);
        auto target_qubit_pauli_list = std::vector<UINT>(qubit_count, 0);
        for (auto qubit_index = 0; qubit_index < qubit_count; qubit_index++) {
            UINT pauli_id = random.int32() % 4;
            target_qubit_index_list[qubit_index] = qubit_index;
            target_qubit_pauli_list[qubit_index] = pauli_id;
        }
        CPPCTYPE coef = random.uniform();
        auto pauli_operator = PauliOperator(
            target_qubit_index_list, target_qubit_pauli_list, coef);
        observable->add_operator(&pauli_operator);
    }
    return observable;
}

ComplexMatrix convert_observable_to_matrix(const Observable* observable) {
    const auto dim = observable->get_state_dim();
    const auto qubit_count = observable->get_qubit_count();
    ComplexMatrix observable_matrix = ComplexMatrix::Zero(dim, dim);
    for (UINT term_index = 0; term_index < observable->get_term_count();
         ++term_index) {
        const auto pauli_operator = observable->get_term(term_index);
        auto coef = pauli_operator->get_coef();
        auto target_index_list = pauli_operator->get_index_list();
        auto pauli_id_list = pauli_operator->get_pauli_id_list();

        std::vector<UINT> whole_pauli_id_list(qubit_count, 0);
        for (UINT i = 0; i < target_index_list.size(); ++i) {
            whole_pauli_id_list[target_index_list[i]] = pauli_id_list[i];
        }

        ComplexMatrix pauli_matrix;
        get_Pauli_matrix(pauli_matrix, whole_pauli_id_list);
        observable_matrix += coef * pauli_matrix;
    }
    return observable_matrix;
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
    char symbol_j[1];
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

bool check_is_unique_index_list(std::vector<UINT> index_list) {
    sort(index_list.begin(), index_list.end());
    bool flag = true;
    for (UINT i = 0; i + 1 < index_list.size(); ++i) {
        flag = flag & (index_list[i] != index_list[i + 1]);
        if (!flag) break;
    }
    return flag;
}
