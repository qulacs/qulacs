
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
DllExport CTYPE* dm_allocate_quantum_state(ITYPE dim);

/**
 * intiialize quantum state to zero state
 * 
 * intiialize quantum state to zero state
 * @param[out] state pointer of quantum state
 * @param[in] dim dimension
 */
DllExport void dm_initialize_quantum_state(CTYPE *state, ITYPE dim);

/**
 * release allocated quantum state
 * 
 * release allocated quantum state
 * @param[in] state quantum state
 */
DllExport void dm_release_quantum_state(CTYPE* state);

/**
 * initialize density matrix from pure state
 *
 * initialize density matrix from pure state
 * @param[out] state pointer of quantum state
 * @param[out] pure_state pointer of quantum state
 * @param[in] dim dimension
 */
DllExport void dm_initialize_with_pure_state(CTYPE *state, const CTYPE *pure_state, ITYPE dim);
