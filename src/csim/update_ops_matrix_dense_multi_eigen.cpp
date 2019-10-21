
#ifndef _MSC_VER
#include "update_ops_cpp.hpp"
extern "C"{
#include "utility.h"
}
#else
#include "update_ops_cpp.hpp"
#include "utility.h"
#endif
#include <Eigen/Core>
#include <functional>

void multi_qubit_dense_matrix_gate_eigen(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim) {

    // matrix dim, mask, buffer
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    const ITYPE* matrix_mask_list = create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);
    Eigen::Map<const Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, Eigen::Aligned> eigen_matrix((std::complex<double>*)matrix,matrix_dim,matrix_dim);
    Eigen::VectorXcd buffer(matrix_dim);
    std::complex<double>* eigen_state = reinterpret_cast<std::complex<double>*>(state);

    // insert index
    const UINT* sorted_insert_index_list = create_sorted_ui_list(target_qubit_index_list, target_qubit_index_count);

    // loop variables
    const ITYPE loop_dim = dim >> target_qubit_index_count;

    ITYPE state_index;
    for(state_index = 0 ; state_index < loop_dim ; ++state_index ){
        // create base index
        ITYPE basis_0 = state_index;
        for(UINT cursor=0; cursor < target_qubit_index_count ; cursor++){
            UINT insert_index = sorted_insert_index_list[cursor];
            basis_0 = insert_zero_to_basis_index(basis_0, 1ULL << insert_index , insert_index );
        }

        // fetch vector
        for(ITYPE y = 0 ; y < matrix_dim ; ++y){
            buffer[y] = eigen_state[basis_0 ^ matrix_mask_list[y]];
        }

        buffer = eigen_matrix * buffer;

        // set result
        for(ITYPE y = 0 ; y < matrix_dim ; ++y){
            eigen_state[basis_0 ^ matrix_mask_list[y]] = buffer[y];
        }
    }
    free((UINT*)sorted_insert_index_list);
    free((ITYPE*)matrix_mask_list);
}

void multi_qubit_dense_matrix_gate_eigen(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>& eigen_matrix, CTYPE* state, ITYPE dim){
    // matrix dim, mask, buffer
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    const ITYPE* matrix_mask_list = create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);
    Eigen::VectorXcd buffer(matrix_dim);
    std::complex<double>* eigen_state = reinterpret_cast<std::complex<double>*>(state);

    // insert index
    const UINT* sorted_insert_index_list = create_sorted_ui_list(target_qubit_index_list, target_qubit_index_count);

    // loop variables
    const ITYPE loop_dim = dim >> target_qubit_index_count;

    ITYPE state_index;
    for(state_index = 0 ; state_index < loop_dim ; ++state_index ){
        // create base index
        ITYPE basis_0 = state_index;
        for(UINT cursor=0; cursor < target_qubit_index_count ; cursor++){
            UINT insert_index = sorted_insert_index_list[cursor];
            basis_0 = insert_zero_to_basis_index(basis_0, 1ULL << insert_index , insert_index );
        }

        // fetch vector
        for(ITYPE y = 0 ; y < matrix_dim ; ++y){
            buffer[y] = eigen_state[basis_0 ^ matrix_mask_list[y]];
        }

        buffer = eigen_matrix * buffer;

        // set result
        for(ITYPE y = 0 ; y < matrix_dim ; ++y){
            eigen_state[basis_0 ^ matrix_mask_list[y]] = buffer[y];
        }
    }
    free((UINT*)sorted_insert_index_list);
    free((ITYPE*)matrix_mask_list);
}

void multi_qubit_dense_matrix_gate_eigen(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const Eigen::MatrixXcd& eigen_matrix, CTYPE* state, ITYPE dim) {

    // matrix dim, mask, buffer
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    const ITYPE* matrix_mask_list = create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);
    std::complex<double>* cppstate = reinterpret_cast<std::complex<double>*>(state);
    Eigen::VectorXcd buffer(matrix_dim);

    // insert index
    const UINT* sorted_insert_index_list = create_sorted_ui_list(target_qubit_index_list, target_qubit_index_count);

    // loop variables
    const ITYPE loop_dim = dim >> target_qubit_index_count;

    ITYPE state_index;
    for(state_index = 0 ; state_index < loop_dim ; ++state_index ){
        // create base index
        ITYPE basis_0 = state_index;
        for(UINT cursor=0; cursor < target_qubit_index_count ; cursor++){
            UINT insert_index = sorted_insert_index_list[cursor];
            basis_0 = insert_zero_to_basis_index(basis_0, 1ULL << insert_index , insert_index );
        }

        // fetch vector
        for(ITYPE y = 0 ; y < matrix_dim ; ++y){
            buffer[y] = cppstate[basis_0 ^ matrix_mask_list[y]];
        }

        buffer = eigen_matrix * buffer;

        // set result
        for(ITYPE y = 0 ; y < matrix_dim ; ++y){
            cppstate[basis_0 ^ matrix_mask_list[y]] = buffer[y];
        }
    }
    free((UINT*)sorted_insert_index_list);
    free((ITYPE*)matrix_mask_list);
}

void multi_qubit_sparse_matrix_gate_eigen(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const Eigen::SparseMatrix<std::complex<double>>& eigen_matrix, CTYPE* state, ITYPE dim) {
	// matrix dim, mask, buffer
	const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
	const ITYPE* matrix_mask_list = create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);
	Eigen::VectorXcd buffer(matrix_dim);
	std::complex<double>* eigen_state = reinterpret_cast<std::complex<double>*>(state);

	// insert index
	const UINT* sorted_insert_index_list = create_sorted_ui_list(target_qubit_index_list, target_qubit_index_count);

	// loop variables
	const ITYPE loop_dim = dim >> target_qubit_index_count;

	ITYPE state_index;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// create base index
		ITYPE basis_0 = state_index;
		for (UINT cursor = 0; cursor < target_qubit_index_count; cursor++) {
			UINT insert_index = sorted_insert_index_list[cursor];
			basis_0 = insert_zero_to_basis_index(basis_0, 1ULL << insert_index, insert_index);
		}

		// fetch vector
		for (ITYPE y = 0; y < matrix_dim; ++y) {
			buffer[y] = eigen_state[basis_0 ^ matrix_mask_list[y]];
		}

		buffer = eigen_matrix * buffer;

		// set result
		for (ITYPE y = 0; y < matrix_dim; ++y) {
			eigen_state[basis_0 ^ matrix_mask_list[y]] = buffer[y];
		}
	}
	free((UINT*)sorted_insert_index_list);
	free((ITYPE*)matrix_mask_list);
}
