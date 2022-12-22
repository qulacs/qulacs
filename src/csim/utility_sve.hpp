#pragma once

inline static ITYPE getVecLength(void) { return svcntd(); }
inline static SV_PRED Svptrue(void) { return svptrue_b64(); }
inline static SV_FTYPE SvdupF(double val) { return svdup_f64(val); }
inline static SV_ITYPE SvdupI(UINT val) { return svdup_u64(val); }
inline static SV_ITYPE SvindexI(UINT base, UINT step) {
    return svindex_u64(base, step);
}
