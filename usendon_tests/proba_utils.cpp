#include "usendon_utils.hpp"
#include <vector>
#include <iostream>

int main()
{
    std::vector<ITYPE> result = {3,0,0,0,3,0,0,3,3,0};
    int num_qubits = 3;

    json counts = convert_to_counts(result, num_qubits);
    std::cout << counts.dump(4) << std::endl;

    return 0;
}
