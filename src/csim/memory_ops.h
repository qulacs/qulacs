
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
 * release allocated quantum state
 * 
 * release allocated quantum state
 * @param[in] psi quantum state
 */
DllExport void release_quantum_state(CTYPE* state);
