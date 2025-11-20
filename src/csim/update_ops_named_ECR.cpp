
/////////////////////////////////////////// esto púxeno para que me usara a
/// función de unroll pero cando quira deixar de facer probas teño que eliminalo

// #undef _USE_SIMD
// #undef _USE_SVE

//////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////

#include <cstring>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// meu

#include <bitset>
#include <complex>
#include <iostream>
#include <vector>
using namespace std::complex_literals;
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////////////////

#include "MPIutil.hpp"
#include "constant.hpp"
#include "update_ops.hpp"
#include "utility.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

void ECR_gate(UINT target_qubit_index_0, UINT target_qubit_index_1,
    CTYPE* state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
    ECR_gate_parallel_simd(
        target_qubit_index_0, target_qubit_index_1, state, dim);
#elif defined(_USE_SVE)
    ECR_gate_parallel_sve(
        target_qubit_index_0, target_qubit_index_1, state, dim);
#else
    ECR_gate_parallel_unroll(
        target_qubit_index_0, target_qubit_index_1, state, dim);
#endif

#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void ECR_gate_parallel_unroll(UINT target_qubit_index_0,
    UINT target_qubit_index_1, CTYPE* state, ITYPE dim) {

    std::cout << "target_qubit_index_0 = " << target_qubit_index_0 << std::endl;
    std::cout << "target_qubit_index_1 = " << target_qubit_index_1 << std::endl;

    std::cout << "dim = " << dim << std::endl;

    const ITYPE loop_dim = dim / 4;

    std::cout << "loop_dim = " << loop_dim << std::endl;

    const ITYPE mask_0 = 1ULL << target_qubit_index_0;

    std::cout << "mask_0 = " << mask_0 << std::endl;

    const ITYPE mask_1 = 1ULL << target_qubit_index_1;

    std::cout << "mask_1 = " << mask_1 << std::endl;

    const ITYPE mask = mask_0 + mask_1;

    std::cout << "mask = " << mask << std::endl;

    const UINT min_qubit_index =
        get_min_ui(target_qubit_index_0, target_qubit_index_1);

    std::cout << "min_qubit_index = " << min_qubit_index << std::endl;

    const UINT max_qubit_index =
        get_max_ui(target_qubit_index_0, target_qubit_index_1);

    std::cout << "max_qubit_index = " << max_qubit_index << std::endl;

    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;

    std::cout << "min_qubit_mask = " << min_qubit_mask << std::endl;

    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);

    std::cout << "max_qubit_mask = " << max_qubit_mask << std::endl;

    const ITYPE low_mask = min_qubit_mask - 1;

    std::cout << "low_mask = " << low_mask << std::endl;

    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;

    std::cout << "mid_mask = " << mid_mask << std::endl;

    const ITYPE high_mask = ~(max_qubit_mask - 1);

    std::cout << "high_mask = " << high_mask << std::endl;

    std::cout << "Despois de high mask" << std::endl;

    const double sqrt2inv = 1. / sqrt(2.);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (ITYPE state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_index_00 = (state_index & low_mask) +
                               ((state_index & mid_mask) << 1) +
                               ((state_index & high_mask) << 2);
        ITYPE basis_index_01 = basis_index_00 + mask_0;
        ITYPE basis_index_10 = basis_index_00 + mask_1;
        ITYPE basis_index_11 = basis_index_00 + mask;
        std::cout <<  "basis_index_00 = " << basis_index_00 <<  " basis_index_01 = " << basis_index_01 <<  " basis_index_10 = " << basis_index_10 <<  " basis_index_11 = " << basis_index_11 << std::endl;

        CTYPE v00 = state[basis_index_00];
        CTYPE v01 = state[basis_index_01];
        CTYPE v10 = state[basis_index_10];
        CTYPE v11 = state[basis_index_11];

        CTYPE new_v00 = sqrt2inv * (v01 + 1.i * v11);
        CTYPE new_v01 = sqrt2inv * (v00 - 1.i * v10);
        CTYPE new_v10 = sqrt2inv * (v11 + 1.i * v01);
        CTYPE new_v11 = sqrt2inv * (v10 - 1.i * v00);

        // Actualizamos el estado
        state[basis_index_00] = new_v00;
        state[basis_index_01] = new_v01;
        state[basis_index_10] = new_v10;
        state[basis_index_11] = new_v11;
    }
}

#ifdef _USE_SIMD
void ECR_gate_parallel_simd(UINT target_qubit_index_0,
    UINT target_qubit_index_1, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE mask_0 = 1ULL << target_qubit_index_0;
    const ITYPE mask_1 = 1ULL << target_qubit_index_1;
    const ITYPE mask = mask_0 + mask_1;

    const UINT min_qubit_index =
        get_min_ui(target_qubit_index_0, target_qubit_index_1);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index_0, target_qubit_index_1);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    const double sqrt2inv = 1. / sqrt(2.);

    // std::cout << "\nEstou aplicando SIMD\n";

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (ITYPE state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_index_00 = (state_index & low_mask) +
                               ((state_index & mid_mask) << 1) +
                               ((state_index & high_mask) << 2);
        ITYPE basis_index_01 = basis_index_00 + mask_0;
        ITYPE basis_index_10 = basis_index_00 + mask_1;
        ITYPE basis_index_11 = basis_index_00 + mask;

        CTYPE v00 = state[basis_index_00];
        CTYPE v01 = state[basis_index_01];
        CTYPE v10 = state[basis_index_10];
        CTYPE v11 = state[basis_index_11];

        CTYPE new_v00 = sqrt2inv * (v01 + 1.i * v11);
        CTYPE new_v01 = sqrt2inv * (v00 - 1.i * v10);
        CTYPE new_v10 = sqrt2inv * (v11 + 1.i * v01);
        CTYPE new_v11 = sqrt2inv * (v10 - 1.i * v00);

        // Actualizamos el estado
        state[basis_index_00] = new_v00;
        state[basis_index_01] = new_v01;
        state[basis_index_10] = new_v10;
        state[basis_index_11] = new_v11;
    }
    // std::cout << "\nUsa OPENMP\n";
}
#endif

////////////////////////////////////////////////////// estou cambiando a función
/// que vai despois disto

// Este _USE_SVE non podo probalo co procesador deste ordenador. Vou comentar a
// función que definin por se estivera ben pero non o sei.

/*
#ifdef _USE_SVE
void ECR_gate_parallel_sve(UINT target_qubit_index_0,
    UINT target_qubit_index_1, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE mask_0 = 1ULL << target_qubit_index_0;
    const ITYPE mask_1 = 1ULL << target_qubit_index_1;
    const ITYPE mask = mask_0 + mask_1;

    const UINT min_qubit_index =
        get_min_ui(target_qubit_index_0, target_qubit_index_1);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index_0, target_qubit_index_1);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    ITYPE state_index = 0;

    // # of complex128 numbers in an SVE register
    ITYPE VL = svcntd() / 2;


    if ((dim > VL) && (min_qubit_mask >= VL)) {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += VL) {

            ITYPE basis_index_00 = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2);
            ITYPE basis_index_01 = basis_index_00 + mask_0;
            ITYPE basis_index_10 = basis_index_00 + mask_1;
            ITYPE basis_index_11 = basis_index_00 + mask;

            // Cargar vectores
            svfloat64_t v00 = svld1(svptrue_b64(),
(double*)&state[basis_index_00]); svfloat64_t v01 = svld1(svptrue_b64(),
(double*)&state[basis_index_01]); svfloat64_t v10 = svld1(svptrue_b64(),
(double*)&state[basis_index_10]); svfloat64_t v11 = svld1(svptrue_b64(),
(double*)&state[basis_index_11]);

            const double sqrt2inv = 1.0 / sqrt(2.0);
            svfloat64_t v_sqrt2inv = svdup_f64(sqrt2inv);

            // Separar real e imaginario (intercalados)
            svfloat64_t v00r = svuzp1_f64(v00, v00);
            svfloat64_t v00i = svuzp2_f64(v00, v00);
            svfloat64_t v01r = svuzp1_f64(v01, v01);
            svfloat64_t v01i = svuzp2_f64(v01, v01);
            svfloat64_t v10r = svuzp1_f64(v10, v10);
            svfloat64_t v10i = svuzp2_f64(v10, v10);
            svfloat64_t v11r = svuzp1_f64(v11, v11);
            svfloat64_t v11i = svuzp2_f64(v11, v11);

            // --- new_v00 = sqrt2inv * (v01 + i*v11)
            svfloat64_t new_v00r = svadd_f64_z(svptrue_b64(), v01r,
svneg_f64_z(svptrue_b64(), v11i)); // real = v01r - v11i svfloat64_t new_v00i =
svadd_f64_z(svptrue_b64(), v01i, v11r);                            // imag =
v01i + v11r

            // --- new_v01 = sqrt2inv * (v00 - i*v10)
            svfloat64_t new_v01r = svadd_f64_z(svptrue_b64(), v00r, v10i); //
real = v00r + v10i svfloat64_t new_v01i = svsub_f64_z(svptrue_b64(), v00i,
v10r);                            // imag = v00i - v10r

            // --- new_v10 = sqrt2inv * (v11 + i*v01)
            svfloat64_t new_v10r = svadd_f64_z(svptrue_b64(), v11r,
svneg_f64_z(svptrue_b64(), v01i));// real = v11r - v01i svfloat64_t new_v10i =
svadd_f64_z(svptrue_b64(), v11i, v01r);                            // imag =
v11i + v01r

            // --- new_v11 = sqrt2inv * (v10 - i*v00)
            svfloat64_t new_v11r = svadd_f64_z(svptrue_b64(), v10r, v00i); //
real = v10r + v00i svfloat64_t new_v11i = svsub_f64_z(svptrue_b64(), v10i,
v00r);                            // imag = v10i - v00r

            // Multiplicación escalar por sqrt2inv
            new_v00r = svmul_f64_z(svptrue_b64(), v_sqrt2inv, new_v00r);
            new_v00i = svmul_f64_z(svptrue_b64(), v_sqrt2inv, new_v00i);
            new_v01r = svmul_f64_z(svptrue_b64(), v_sqrt2inv, new_v01r);
            new_v01i = svmul_f64_z(svptrue_b64(), v_sqrt2inv, new_v01i);
            new_v10r = svmul_f64_z(svptrue_b64(), v_sqrt2inv, new_v10r);
            new_v10i = svmul_f64_z(svptrue_b64(), v_sqrt2inv, new_v10i);
            new_v11r = svmul_f64_z(svptrue_b64(), v_sqrt2inv, new_v11r);
            new_v11i = svmul_f64_z(svptrue_b64(), v_sqrt2inv, new_v11i);

            // Recombinar real/imag en formato intercalado
            svfloat64_t new_v00 = svzip1_f64(new_v00r, new_v00i);
            svfloat64_t new_v01 = svzip1_f64(new_v01r, new_v01i);
            svfloat64_t new_v10 = svzip1_f64(new_v10r, new_v10i);
            svfloat64_t new_v11 = svzip1_f64(new_v11r, new_v11i);

            // Guardar resultados
            svst1(svptrue_b64(), (double*)&state[basis_index_00], new_v00);
            svst1(svptrue_b64(), (double*)&state[basis_index_01], new_v01);
            svst1(svptrue_b64(), (double*)&state[basis_index_10], new_v10);
            svst1(svptrue_b64(), (double*)&state[basis_index_11], new_v11);

        }
    } else {  // if ((dim > VL) && (min_qubit_mask >= VL))
#pragma omp parallel for
        for (ITYPE state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_00 = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2);
            ITYPE basis_index_01 = basis_index_00 + mask_0;
            ITYPE basis_index_10 = basis_index_00 + mask_1;
            ITYPE basis_index_11 = basis_index_00 + mask;

            CTYPE v00 = state[basis_index_00];
            CTYPE v01 = state[basis_index_01];
            CTYPE v10 = state[basis_index_10];
            CTYPE v11 = state[basis_index_11];

            CTYPE new_v00 = sqrt2inv*(v01 + 1.i*v11);
            CTYPE new_v01 = sqrt2inv*(v00 - 1.i*v10);
            CTYPE new_v10 = sqrt2inv*(v11 + 1.i*v01);
            CTYPE new_v11 = sqrt2inv*(v10 - 1.i*v00);

            // Actualizamos el estado
            state[basis_index_00] = new_v00;
            state[basis_index_01] = new_v01;
            state[basis_index_10] = new_v10;
            state[basis_index_11] = new_v11;
        }
    }
}
#endif
*/


#include <complex>
#include <cstdio>
#include <type_traits>



#ifdef _USE_MPI
#include <bitset>
#include <iostream>

void ECR_gate_mpi(UINT target_qubit_index_0, UINT target_qubit_index_1,
    CTYPE* state, ITYPE dim, UINT inner_qc) {
    std::cout << " Entrando en ECR_gate_mpi por primeira vez ===" << std::endl;

    UINT left_qubit, right_qubit;
    if (target_qubit_index_0 > target_qubit_index_1) {
        left_qubit = target_qubit_index_0;
        right_qubit = target_qubit_index_1;
    } else {
        left_qubit = target_qubit_index_1;
        right_qubit = target_qubit_index_0;
    }

    if (left_qubit < inner_qc) {
        std::cout << "Caso 1: ambos qubits internos (no MPI)" << std::endl;
        ECR_gate(target_qubit_index_0, target_qubit_index_1, state, dim);
    } else if (right_qubit < inner_qc) {  // one target is outer
        std::cout << "Caso 2: un qubit interno e outro externo" << std::endl;
        MPIutil& m = MPIutil::get_inst();
        const UINT rank = m.get_rank();
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE* t = m.get_workarea(&dim_work, &num_work);
        const ITYPE tgt_rank_bit = 1 << (left_qubit - inner_qc);
        const ITYPE rtgt_blk_dim = 1 << right_qubit;
        const int pair_rank = rank ^ tgt_rank_bit;
        std::cout << "1extrank " << rank << " pair_rank = " << pair_rank
                  << std::endl;

        CTYPE* si = state;

        for (UINT i = 0; i < (UINT)num_work; ++i) {
            m.m_DC_sendrecv(si, t, dim_work, pair_rank);

            _ECR_gate_mpi(t, si, dim_work, rtgt_blk_dim);

            si += dim_work;
        }


    } else {  // both targets are outer
        std::cout << "Caso 3: aplicando a ECR entre dous qubits externos" << std::endl;
        // Aquí os dous qubits son externos ao proceso. Precísase facer dúas comunicacións.
        MPIutil& m = MPIutil::get_inst();
        const UINT rank = m.get_rank();

        int world_size_int = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_int); // MPI_Comm_size obtén o número total 
        // de procesos que están participando no comunicador dado. MPI_COMM_WORLD é o comunicador
        // global que inclúe todos os procesos
        const int world_size = world_size_int; // world_size_int contén a cantidade total de procesos
        // do programa MPI.

        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        const UINT tgt0_rank_bit = 1 << (left_qubit - inner_qc);  // left_qubit é o de maior índice. No caso de
                              // aplicar a porta ECR(2,3), left_qubit = 3.
        const UINT tgt1_rank_bit = 1 << (right_qubit - inner_qc);  // right_qubit é o de menor índice. No caso de
                              // aplicar a porta ECR(2,3), left_qubit = 2.
        const UINT tgt_rank_bit = tgt0_rank_bit + tgt1_rank_bit;  // súmanse as dúas máscaras de bits e polo tanto
                            // obtense unha máscara que ten todo ceros excepto
                            // dous uns nas posicións de left_qubit e
                            // right_qubit

        // cada un dos procesos comunica tanto co que ten o left_qubit invertido
        // (pair_rank1) como co que ten tanto o left_qubit como o right_qubit
        // invertidos.

        const int pair_rank = rank ^ tgt_rank_bit;  // left_qubit e right_qubit invertidos
        const int pair_rank1 = rank ^ tgt1_rank_bit;  // left_qubit invertido


        // Si necesitas consultar la librería por las dimensiones disponibles,
        // hazlo una vez. Si no hace falta, elimina la llamada.
        // Esto sirve para conocer dim_work/num_work si la librería los calcula.

        CTYPE* tmp = m.get_workarea(
            &dim_work, &num_work); 
        (void)tmp; // silencia o warning de variable non usada porque tmp só se usa para
        // obter os valores de dim_work e num_work             


        // reserva unha vez dous buffers independentes e inicializados a cero.
        std::vector<CTYPE> t1_buf(dim_work);
        std::vector<CTYPE> t2_buf(dim_work);

        CTYPE* t1 = t1_buf.data();
        CTYPE* t2 = t2_buf.data();

        CTYPE* si = state; 

        // envíanse os elementos de si do rank ao pair_rank e recíbense deste
        // último o mesmo número de elementos de si en t1.
        for (UINT i = 0; i < (UINT)num_work; ++i) {
            CTYPE* si_i = state + i * dim_work; 

            m.m_DC_sendrecv(si_i, t1, dim_work, pair_rank);

        }

        // envíanse os elementos de si do rank ao pair_rank1 e recíbense deste
        // último o mesmo número de elementos de si en t2.
        for (UINT j = 0; j < (UINT)num_work; ++j) {
            CTYPE* si_j = state + j * dim_work;  

            m.m_DC_sendrecv(si_j, t2, dim_work, pair_rank1);

        }

        const ITYPE rtgt_blk_dim = 1 << right_qubit;

        ITYPE num_proc_bloque = rtgt_blk_dim/dim_work; // con isto calculo o número
        // de procesos por bloque (os bloques son os rtgt_blk_dim e definen o que 
        // tarda en cambiar a fórmula que permite aplicar a ECR gate).

        // agrúpanse os ranks consecutivos en bloques de tamaño num_proc_bloque
        // e asigna bloque 0 -> listaA, bloque 1 -> listaB, bloque 2 -> listaA, ...
        auto split_ranks_alternate = [&](int world_sz, int group_size)
            -> std::pair<std::vector<int>, std::vector<int>> {
            std::vector<int> listA, listB;
            if (group_size <= 0) return {listA, listB};
            int nblocks = (world_sz + group_size - 1) / group_size; // calcula o
            // número de bloques necesarios para cubrir world_sz elementos usando
            // división enteira con redondeo cara arriba. world_size é o número de 
            // procesos totais e group_size = num_proc_bloque é o número de procesos
            // por bloque.
            for (int b = 0; b < nblocks; ++b) {
                int start = b * group_size; // índice do primeiro rank do bloque
                int end = std::min(world_sz, start + group_size); // índice do 
                // final do bloque. Úsase min para non pasarse de world_sz no
                // último bloque que pode ser máis curto.
                // os bucles van ir ata end-1, end xa pertence ao seguinte bloque.
                if ((b % 2) == 0) { // se b é par rechea a listaA. b é o índice do bloque.
                    // Se nos referimos aos procesos, a fórmula que se aplica non depende
                    // de se o proceso é par ou impar, pero sempre se aplica a primeira 
                    // fórmula ao primeiro bloque, a segunda ao segundo e despois vólvese ao
                    // primeiro, de xeito que sempre se aplica a primeira fórmula nos bloques
                    // pares e a segunda nos impares.
                    for (int r = start; r < end; ++r) listA.push_back(r);
                } else { // se b é impar, rechea a listaB. 
                    for (int r = start; r < end; ++r) listB.push_back(r);
                }
            }
            return {listA, listB};
        };


        // world_size é a cantidade total de procesos, num_proc_bloque é a cantidade de procesos
        // por bloque de tamaño rtgt_blk_dim
        auto lists = split_ranks_alternate(world_size, static_cast<int>(num_proc_bloque));
        const std::vector<int>& listA = lists.first;
        const std::vector<int>& listB = lists.second;

        bool inA = std::find(listA.begin(), listA.end(), (int)rank) != listA.end(); // devolve un 
        // booleano indicando se o rank correspondente está ou non na listaA. 0 para False, 1 para True.

        
        // aplícase a función de _ECR_gate para dous qubits externos.
        for (UINT k = 0; k < (UINT)num_work; ++k) {
            _ECR_gate_mpi_externos(t1, t2, si, dim_work, rtgt_blk_dim, inA, num_proc_bloque);

            si += dim_work;
        }
    }
}

// función _ECR_gate que se usa para un qubit interno e outro externo
void _ECR_gate_mpi(CTYPE* t, CTYPE* si, ITYPE dim, ITYPE rtgt_blk_dim) {
    const double sqrt2inv = 1. / sqrt(2.);
    ITYPE state_index = 0;
    const ITYPE amplitude_block_size = rtgt_blk_dim << 1; // 2*rtgt_blk_dim. Define a 
    // amplitude do bloque no que primeiro se aplicará unha fórmula e despois a outra. 
    // A continuación pasarase de volta á primeira fórmula pero iso xa formará parte
    // do seguinte bloque.

#pragma omp parallel for
    for (state_index = 0; state_index < dim;
        state_index += amplitude_block_size) {
        for (ITYPE offset = 0; offset < rtgt_blk_dim; ++offset) {
            // Cada bloque no que se aplican as dúas fórmulas ten tamaño amplitude_block_size, 
            // pero o tamaño dos bloques nos que se aplican cada unha delas é rtgt_blk_dim.
            // rtgt_blk_dim = amplitude_block_size/2.
            const ITYPE idx0 = state_index + offset; 
            const ITYPE idx1 = idx0 + rtgt_blk_dim; 
            const std::complex<double> si0 = si[idx0];

            si[idx0] = (si[idx1] + t[idx1] * 1i) * sqrt2inv;

            si[idx1] = (si0 - t[idx0] * 1i) * sqrt2inv;
        }
    }
}

// función _ECR_gate para dous qubits externos
void _ECR_gate_mpi_externos(
    CTYPE* t1, CTYPE* t2, CTYPE* si, ITYPE dim, ITYPE rtgt_blk_dim, bool inA, ITYPE num_proc_bloque) {
    const double sqrt2inv = 1. / sqrt(2.);
    ITYPE state_index = 0;
    const ITYPE amplitude_block_size = rtgt_blk_dim << 1;

    
    #pragma omp parallel for
        for (state_index = 0; state_index < dim;
            state_index += amplitude_block_size) {

            const ITYPE fin = (num_proc_bloque == 1) ? rtgt_blk_dim : dim;
            // se só hai un proceso por bloque podo percorrer todos os valores nos que
            // se aplica unha mesma fórmula chegando ata o índice rtgt_blk_dim-1 sen problema.
            
            // se hai varios procesos por bloque non podo percorrer todos os valores nos que
            // se aplica unha mesma fórmula chegando ata o índice rtgt_blk_dim-1. Cada proceso chegará
            // ata o índice dim_work-1 e teño que ir percorrendo os índices de todos os procesos ata
            // aplicar a fórmula nos rtgt_blk_dim elementos que preciso pero en cada proceso reiníciase
            // o índice a cero.

            for (ITYPE offset = 0; offset < fin; ++offset) {
                const ITYPE idx0 = state_index + offset;

                if (inA) {
                    si[idx0] = (t2[idx0] + t1[idx0] * 1i) * sqrt2inv;
                } else {
                    si[idx0] = (t2[idx0] - t1[idx0] * 1i) * sqrt2inv;
                }
            }
        }
}



#endif
