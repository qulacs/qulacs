
/**
 * @file state.h
 * @brief Definition and basic functions for state vector
 */

#pragma once

#include "type.h"
 
 /**
 * intiialize quantum state to zero state
 *
 * intiialize quantum state to zero state
 * @param[out] psi pointer of quantum state
 * @param[in] dim dimension
 */
DllExport void initialize_quantum_state(CTYPE *state, ITYPE dim);

DllExport void initialize_Haar_random_state(CTYPE *state, ITYPE dim);

DllExport void initialize_Haar_random_state_with_seed(CTYPE *state, ITYPE dim, UINT seed);
