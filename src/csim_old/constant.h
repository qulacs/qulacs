/**
 * @file constant.h
 * @brief definitions of constant values
 */

#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include "type.h"

//! PI value
#ifndef PI
#ifdef M_PI
#define PI M_PI
#else
#define PI      3.141592653589793
#endif
#endif

//! square root of 2
#define SQRT2 1.414213562373095

//! cosine pi/8
#define COSPI8 0.923879532511287

//! sine pi/8
#define SINPI8 0.382683432365090

//! list of Pauli matrix I,X,Y,Z
extern const CTYPE PAULI_MATRIX[4][4];
//! S-gate
extern const CTYPE S_GATE_MATRIX[4];
//! Sdag-gate
extern const CTYPE S_DAG_GATE_MATRIX[4];
//! T-gate
extern const CTYPE T_GATE_MATRIX[4];
//! Tdag-gate
extern const CTYPE T_DAG_GATE_MATRIX[4];
//! Hadamard gate
extern const CTYPE HADAMARD_MATRIX[4];
//! square root of X gate
extern const CTYPE SQRT_X_GATE_MATRIX[4];
//! square root of Y gate
extern const CTYPE SQRT_Y_GATE_MATRIX[4];
//! square root dagger of X gate
extern const CTYPE SQRT_X_DAG_GATE_MATRIX[4];
//! square root dagger of Y gate
extern const CTYPE SQRT_Y_DAG_GATE_MATRIX[4];
//! Projection to 0
extern const CTYPE PROJ_0_MATRIX[4];
//! Projection to 1
extern const CTYPE PROJ_1_MATRIX[4];
//! complex values for exp(j * i*pi/4 )
extern const CTYPE PHASE_90ROT[4];
//! complex values for exp(-j * i*pi/4 )
extern const CTYPE PHASE_M90ROT[4];

