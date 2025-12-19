
#include "arm_sve.h"

/* void mul_by_i_sve() {
    svbool_t pg = svwhilelt_b64(0, 2);
    svfloat64_t vec = svdup_f64(3.14);
    svfloat64_t b_hi = svld1(pg, ptr);
    svbool_t pg(true);
    svfloat64_t sign = svdup_f64(-1.0);    
    svfloat64_t test = svmul_f64_m(pg, b_hi, sign);    
} */

void mul_by_i_sve(double* ptr, double* out) {
    // Máscara activa para todos los elementos del vector
    svbool_t pg = svptrue_b64();
    // Cargar elementos desde memoria
    svfloat64_t b_hi = svld1(pg, ptr);
    // Vector de signos (-1.0) para multiplicar por i
    svfloat64_t sign = svdup_f64(-1.0);
    // Multiplicar por la unidad imaginaria: -b_hi
    svfloat64_t result = svmul_f64_m(pg, b_hi, sign);
    // Guardar de vuelta en memoria
    svst1(pg, out, result);
}


int main() {

    double input[8]  = {1,2,3,4,5,6,7,8};
    double output[8] = {0};

    mul_by_i_sve(input, output);
        


    return 0;
}