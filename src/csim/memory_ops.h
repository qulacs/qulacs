
/**
 * @file state.h
 * @brief Definition and basic functions for state vector
 */

#pragma once

#include "type.h"

/**
 * allocate quantum state in memory
 * 
 * allocate quantum state in memory
 * @param[in] dim dimension, i.e. size of vector
 * @return pointer to allocated vector
 */
DllExport CTYPE* allocate_quantum_state(ITYPE dim);

/**
 * intiialize quantum state to zero state
 * 
 * intiialize quantum state to zero state
 * @param[out] psi pointer of quantum state
 * @param[in] dim dimension
 */
DllExport void initialize_quantum_state(CTYPE *state, ITYPE dim);

/**
 * release allocated quantum state
 * 
 * release allocated quantum state
 * @param[in] psi quantum state
 */
DllExport void release_quantum_state(CTYPE* state);

DllExport void initialize_Haar_random_state(CTYPE *state, ITYPE dim);

DllExport void initialize_Haar_random_state_with_seed(CTYPE *state, ITYPE dim, UINT seed);
