#ifndef _UTIL_EXPORT_H_
#define _UTIL_EXPORT_H_

#if defined(__MINGW32__) || defined(_MSC_VER)
#define DllExport __declspec(dllexport)
#else
#define DllExport __attribute__((visibility ("default")))
#endif

#endif // #ifndef _UTIL_COMMON_CU_H_