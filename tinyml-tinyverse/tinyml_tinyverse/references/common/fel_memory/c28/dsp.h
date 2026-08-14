#ifndef _DSP_H_
#define _DSP_H_
//#############################################################################
//
//! \file   dsp.h
//!
//! \brief  DSP Library Definitions and Types
//! \author Vishal Coelho
//! \date   Jan 11, 2016
//
//  Group:          C2000
//  Target Device:  C28x+FPU32/FPU64
//
//#############################################################################
//
//
// $Copyright: Copyright (C) 2026 Texas Instruments Incorporated -
//             http://www.ti.com/ ALL RIGHTS RESERVED $
//#############################################################################

#ifndef _RTS_IN_USE_
#define _RTS_IN_USE_
//This conditional prevents the following data types from conflicting with the
//FPUfastRTS library

//*****************************************************************************
// the includes
//*****************************************************************************
#include <stdint.h>

//!
//! \defgroup DSP_HEADER DSP Library Definitions and Types
//!
//! \addtogroup DSP_HEADER 
// @{

#ifdef __cplusplus
extern "C" {
#endif

//*****************************************************************************
//defines
//*****************************************************************************
//! DSP Library Version
#define LIBRARY_VERSION     "2.05.00.00"


//*****************************************************************************
// typedefs
//*****************************************************************************
// c2000 types for IEEE754 (same as defined in <device>/inc/hw_types.h)
#ifndef C2000_IEEE754_TYPES
#define C2000_IEEE754_TYPES
#ifdef __TI_EABI__
typedef float         float32_t;
typedef double        float64_t;
#else // TI COFF
typedef float         float32_t;
typedef long double   float64_t;
#endif // __TI_EABI__
#endif // C2000_IEEE754_TYPES

#if defined(__TMS320C28XX_FPU64__)
//! \brief 64-bit Double Precision Float
//! The union of a double precision value, an unsigned long long and a signed
//! long long allows for manipulation of the hex representation of the floating
//! point value as well as signed and unsigned arithmetic to determine error 
//! metrics.
//! This data type is only defined if the compiler option --float_support is 
//! set to fpu64
//! 
typedef union
{
    uint64_t    ui64; //!< Unsigned long long representation
    int64_t     i64;  //!< Signed long long represntaion 
    float64_t   f64;  //!< Double precision (64-bit) representation
}float64u_t;
#endif //defined(__TMS320C28XX_FPU64__)

#if (defined(__TMS320C28XX_FPU32__) || defined(__TMS320C28XX_FPU64__))
//! \brief 32-bit Single Precision Float
//! The union of a single precision value, an unsigned long and a signed
//! long allows for manipulation of the hex representation of the floating
//! point value as well as signed and unsigned arithmetic to determine error 
//! metrics.
//! This data type is only defined if the compiler option --float_support is 
//! set to fpu32
//!
typedef union
{
    uint32_t    ui32; //!< Unsigned long representation
    int32_t     i32;  //!< Signed long represntaion 
    float       f32;  //!< Single precision (32-bit) representation
}float32u_t;
#endif //defined(__TMS320C28XX_FPU32__)

#if !(defined(__TMS320C28XX_FPU32__) || defined(__TMS320C28XX_FPU64__))
#error "--float_support=fpu64 option must be set to build the FPU64        \
    version of the library. --float_support=fpu32 must be set to build the \
    FPU32 version of the DSP library"
#endif // (defined(__TMS320C28XX_FPU32__) || defined(__TMS320C28XX_FPU64__))

//! \brief Function pointer with a void pointer argument returning nothing
//!
typedef void (*v_pfn_v)(void *);

#endif // _RTS_IN_USE_

// @} //addtogroup

#ifdef __cplusplus
}
#endif // extern "C"

#endif  // end of  _DSP_H_

// End of File
