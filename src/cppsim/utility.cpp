
#include "utility.hpp"

void get_Pauli_matrix(ComplexMatrix& matrix, const std::vector<UINT>& pauli_id_list) {
    ITYPE matrix_dim = 1ULL << pauli_id_list.size();
    matrix = ComplexMatrix::Zero(matrix_dim, matrix_dim);

    ITYPE flip_mask = 0;
    ITYPE phase_mask = 0;
    UINT rot90_count = 0;
    for (UINT ind = 0; ind < pauli_id_list.size(); ++ind) {
        UINT pauli_id = pauli_id_list[ind];
        if (pauli_id == 1) {
            flip_mask ^= 1ULL << ind;
        }
        else if (pauli_id == 2) {
            flip_mask ^= 1ULL << ind;
            phase_mask ^= 1ULL << ind;
            rot90_count++;
        }
        else if (pauli_id == 3) {
            phase_mask ^= 1ULL << ind;
        }
    }
    std::vector<CPPCTYPE> rot = { 1, -1.i, -1, 1.i };
    for (ITYPE index = 0; index < matrix_dim; ++index) {
        double sign = 1. - 2. * (count_population_cpp(index&phase_mask) % 2);
        matrix(index, index^flip_mask) = rot[rot90_count % 4] * sign;
    }
}

std::vector<std::string> split(const std::string &s, const std::string &delim){
    std::vector<std::string> elements;

    std::string item;

    for (char ch: s){
        if (delim.find(ch) != std::string::npos){
            if (!item.empty()) elements.push_back(item);
            item.clear();
        } else {
            item += ch;
        }
    }

    if (!item.empty()) elements.push_back(item);

    return elements;
}

void chfmt(std::string& ops){
    for (UINT i = 0; i < ops.size(); ++i){
        if (ops[i] == 'X' || ops[i] == 'Y' || ops[i] == 'Z' || ops[i] == 'I'){
            ops.insert(++i, " ");
        }
    }
}

std::tuple<double, double, std::string> parse_openfermion_line(std::string line){
    double coef_real, coef_imag;

    char buf[256];
    char symbol_j[1];
    UINT matches;

    if(line[0]=='('){
        matches = std::sscanf(line.c_str(), "(%lf+%lfj) [%[^]]]", &coef_real, &coef_imag, buf);
        if (matches < 2){
            matches = std::sscanf(line.c_str(), "(%lf-%lfj) [%[^]]]", &coef_real, &coef_imag, buf);
            coef_imag = -coef_imag;
        }
        if (matches < 3){
            std::strcpy(buf, "I0");
        }
    }else{
        matches = std::sscanf(line.c_str(), "%lf%[j] [%[^]]]", &coef_imag, symbol_j, buf);
        coef_real = 0.;
        if (matches < 3){
            std::strcpy(buf, "I0");
        }
        if (symbol_j[0] != 'j'){
            matches = std::sscanf(line.c_str(), "%lf [%[^]]]", &coef_real, buf);
            coef_imag = 0.;
            if (matches < 2){
                std::strcpy(buf, "I0");
            }
        }
        if (matches == 0){
            return std::make_tuple((double)NULL, (double)NULL, (std::string)NULL);
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
