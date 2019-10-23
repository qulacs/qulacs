#include "constant.h"

// elementary set of gates
const CTYPE PAULI_MATRIX[4][4] = {
    {
        1, 0, 
        0, 1
    },
    {
        0, 1, 
        1, 0
    },
    {
        0, -1.i, 
        1.i, 0
    },
    {
        1, 0, 
        0, -1
    }
};

const CTYPE S_GATE_MATRIX[4] = 
{
    1, 0, 
    0, 1.i
};

const CTYPE S_DAG_GATE_MATRIX[4] = 
{
    1, 0, 
    0, -1.i
};

const CTYPE T_GATE_MATRIX[4] =
{
    COSPI8-1.i*SINPI8,  0.,
    0.,                 COSPI8+1.i*SINPI8
};

const CTYPE T_DAG_GATE_MATRIX[4] =
{
    COSPI8+1.i*SINPI8,  0.,
     0.,                COSPI8-1.i*SINPI8
};

const CTYPE HADAMARD_MATRIX[4] =
{
    1./SQRT2, 1./SQRT2,
    1./SQRT2, -1./SQRT2
};

const CTYPE SQRT_X_GATE_MATRIX[4] = 
{
    0.5+0.5i, 0.5-0.5i,
    0.5-0.5i, 0.5+0.5i
};

const CTYPE SQRT_Y_GATE_MATRIX[4] =
{
    0.5+0.5i, -0.5-0.5i,
    0.5+0.5i, 0.5+0.5i
};

const CTYPE SQRT_X_DAG_GATE_MATRIX[4] = 
{
    0.5-0.5i, 0.5+0.5i,
    0.5+0.5i, 0.5-0.5i
};

const CTYPE SQRT_Y_DAG_GATE_MATRIX[4] =
{
    0.5-0.5i, 0.5-0.5i,
    -0.5+0.5i, 0.5-0.5i
};

const CTYPE PROJ_0_MATRIX[4] = 
{
    1, 0,
    0, 0
};

const CTYPE PROJ_1_MATRIX[4] =
{
    0, 0,
    0, 1
};

const CTYPE PHASE_90ROT[4] = {1., 1.i, -1, -1.i};
const CTYPE PHASE_M90ROT[4] = { 1., -1.i, -1, 1.i };


