#pragma once

#include "type.h"
#include "constant.h"

/**
 * Insert 0 to qubit_index-th bit of basis_index. basis_mask must be 1ULL << qubit_index.
 */
inline static ITYPE insert_zero_to_basis_index(ITYPE basis_index, ITYPE basis_mask, UINT qubit_index){
    ITYPE temp_basis = (basis_index >> qubit_index) << (qubit_index+1);
    return temp_basis + basis_index % basis_mask;
}

/**
 * Swap amplitude of state[basis_index_0] and state[basis_index_1]
 */
inline static void swap_amplitude(CTYPE* state, ITYPE basis_index_0, ITYPE basis_index_1){
    CTYPE temp = state[basis_index_0];
    state[basis_index_0] = state[basis_index_1];
    state[basis_index_1] = temp;
}
/**
 * min for ITYPE
 */
inline static ITYPE get_min_ll(ITYPE index_0, ITYPE index_1){
    return (index_0<index_1) ? index_0 : index_1;
}
/**
 * max for ITYPE
 */
inline static ITYPE get_max_ll(ITYPE index_0, ITYPE index_1){
    return (index_0>index_1) ? index_0 : index_1;
}
/**
 * min for UINT
 */
inline static UINT get_min_ui(UINT index_0, UINT index_1){
    return (index_0<index_1) ? index_0 : index_1;
}
/**
 * max for UINT
 */
inline static UINT get_max_ui(UINT index_0, UINT index_1){
    return (index_0>index_1) ? index_0 : index_1;
}


/**
 * Count population (number of bit with 1) in 64bit unsigned integer.
 * 
 * See http://developer.cybozu.co.jp/takesako/2006/11/binary_hacks.html for why it works
 */ 
inline static UINT count_population(ITYPE x) 
{ 
    x = ((x & 0xaaaaaaaaaaaaaaaaUL) >> 1) 
      +  (x & 0x5555555555555555UL); 
    x = ((x & 0xccccccccccccccccUL) >> 2) 
      +  (x & 0x3333333333333333UL); 
    x = ((x & 0xf0f0f0f0f0f0f0f0UL) >> 4) 
      +  (x & 0x0f0f0f0f0f0f0f0fUL); 
    x = ((x & 0xff00ff00ff00ff00UL) >> 8) 
      +  (x & 0x00ff00ff00ff00ffUL); 
    x = ((x & 0xffff0000ffff0000UL) >> 16) 
      +  (x & 0x0000ffff0000ffffUL); 
    x = ((x & 0xffffffff00000000UL) >> 32) 
      +  (x & 0x00000000ffffffffUL); 
    return (UINT)x; 
} 

void sort_ui(UINT* array, size_t array_size);
UINT* create_sorted_ui_list(const UINT* array, size_t size);
UINT* create_sorted_ui_list_value(const UINT* array, size_t size, UINT value);
UINT* create_sorted_ui_list_list(const UINT* array1, size_t size1, const UINT* array2, size_t size2);

/**
 * Generate mask for all the combination of bit-shifting in qubit_index_list
 * 
 * For example, when qubit_index_list = [0,3], the result is [0000, 0001, 1000, 1001].
 * When two-qubit gate on 0- and 3-qubit, state[0010] is a linear combination of amplitude about [0010, 0011, 1010, 1011], 
 * which can be generated with bit-shifts with the above results.
 */
ITYPE* create_matrix_mask_list(const UINT* qubit_index_list, UINT qubit_index_count);
ITYPE create_control_mask(const UINT* qubit_index_list, const UINT* value_list, UINT qubit_index_count);

/**
 * Calculate bit-flip mask, phase-flip mask, global_phase, and pivot-qubit.
 * 
 * Two masks are generated from Pauli operator. 
 * Bit-flip mask is a bit-array of which the i-th bit has 1 if X or Y occurs on the i-th qubit.
 * Phase-flip mask is a bit-array of which the i-th bit has 1 if Y or Z occurs on the i-th qubit.
 * 
 * Suppose P is Pauli operator, |x> is computational basis, P|x> = alpha |y>, and P|0> = alpha_0 |y_0>.
 * We see 
 * y = x ^ bit_mask.
 * alpha/alpha_0 is (-1)**|x & phase_mask|, where || means the number of 1 in bit array, up to global phase.
 * alpha_0 is i**(the number of Pauli-Y in P).
 * 
 * global_phase_90rot_count is the number of Pauli-Y in P.
 * pivot_qubit_index is the last qubit which is flipped by Pauli operator, which is required in the next computation.
 */
void get_Pauli_masks_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, 
    ITYPE* bit_flip_mask, ITYPE* phase_flip_mask, UINT* global_phase_90rot_count, UINT* pivot_qubit_index);
void get_Pauli_masks_whole_list(const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, 
    ITYPE* bit_flip_mask, ITYPE* phase_flip_mask, UINT* global_phase_90rot_count, UINT* pivot_qubit_index);
