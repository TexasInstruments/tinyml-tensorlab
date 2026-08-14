//#############################################################################
//
//
// $Copyright:
//#############################################################################
#ifndef _CFFT_F32_H_
#define _CFFT_F32_H_
//#############################################################################
//
// FILE:   cfft_f32.h
//
// TITLE:  CFFT structures and prototypes
//
//
//#############################################################################


//*****************************************************************************
// includes
//*****************************************************************************
#include <stdint.h>
#include "math.h"
//!
//! \defgroup DSP_CFFT_F32 Complex Fast Fourier Transforms

//!
//! \addtogroup DSP_CFFT_F32
// @{
    
#ifdef __cplusplus
extern "C" {
#endif

//*****************************************************************************
// defines
//*****************************************************************************


//*****************************************************************************
// typedefs
//*****************************************************************************
typedef float float32_t;

typedef struct {
	float32_t Re; 
	float32_t Im;
	} complex_T;

//! \brief 32-bit Single Precision Float
//! The union of a single precision value, an unsigned long and a signed
//! long allows for manipulation of the hex representation of the floating
//! point value as well as signed and unsigned arithmetic to determine error 
//! metrics.
//!
typedef union
{
    uint32_t    ui32; //!< Unsigned long representation
    int32_t     i32;  //!< Signed long represntaion 
    float32_t   f32;  //!< Single precision (32-bit) representation
} float32u_t;

//! \brief Complex FFT structure
//!
typedef struct {
    float32_t   *InPtr;         //!< Pointer to the input buffer
    float32_t   *OutPtr;        //!< Pointer to the output buffer
    float32_t   *CoefPtr;       //!< Pointer to the twiddle factors
    float32_t   *CurrentInPtr;  //!< Points to input buffer at each FFT stage
    float32_t   *CurrentOutPtr; //!< Points to output buffer at each FFT stage
    uint16_t Stages;        //!< Number of FFT stages
    uint16_t FFTSize;       //!< Size of the FFT (number of complex data points)
}CFFT_F32_STRUCT;

//! \brief Handle to the CFFT_F32 structure
//!
typedef CFFT_F32_STRUCT* CFFT_F32_STRUCT_Handle;

//*****************************************************************************
// globals
//*****************************************************************************


//*****************************************************************************
// 'inline' function
//*****************************************************************************

//! \brief Set the input buffer pointer
//! \param[in] fh  handle to the 'CFFT_F32_STRUCT' object
//! \param[in] pi  pointer to the input buffer
//!
static inline void CFFT_f32_setInputPtr(CFFT_F32_STRUCT_Handle fh,
        const float32_t *pi)
{
    fh->InPtr = (float32_t *)pi;
}

//! \brief Get the input buffer pointer
//! \param[in] fh  handle to the 'CFFT_F32_STRUCT' object
//! \return pi  pointer to the input buffer
//!
static inline float32_t * CFFT_f32_getInputPtr(CFFT_F32_STRUCT_Handle fh)
{
    return(fh->InPtr);
}

//! \brief Set the output buffer pointer
//! \param[in] fh  handle to the 'CFFT_F32_STRUCT' object
//! \param[in] po  pointer to the output buffer
//!
static inline void CFFT_f32_setOutputPtr(CFFT_F32_STRUCT_Handle fh,
        const float32_t *po)
{
    fh->OutPtr = (float32_t *)po;
}

//! \brief Get the output buffer pointer
//! \param[in] fh  handle to the 'CFFT_F32_STRUCT' object
//! \return po  pointer to the output buffer
//!
static inline float32_t * CFFT_f32_getOutputPtr(CFFT_F32_STRUCT_Handle fh)
{
    return(fh->OutPtr);
}

//! \brief Set the twiddles pointer
//! \param[in] fh  handle to the 'CFFT_F32_STRUCT' object
//! \param[in] pt  pointer to the twiddles
//!
static inline void CFFT_f32_setTwiddlesPtr(CFFT_F32_STRUCT_Handle fh,
        const float32_t *pc)
{
    fh->CoefPtr = (float32_t *)pc;
}

//! \brief Get the twiddles pointer
//! \param[in] fh  handle to the 'CFFT_F32_STRUCT' object
//! \return pc  pointer to the twiddles
//!
static inline float32_t * CFFT_f32_getTwiddlesPtr(CFFT_F32_STRUCT_Handle fh)
{
    return(fh->CoefPtr);
}

//! \brief Set the current input buffer pointer
//! \param[in] fh  handle to the 'CFFT_F32_STRUCT' object
//! \param[in] pi  pointer to the current input buffer
//!
static inline void CFFT_f32_setCurrInputPtr(CFFT_F32_STRUCT_Handle fh,
        const float32_t *pi)
{
    fh->CurrentInPtr = (float32_t *)pi;
}


//! \brief Get the current input buffer pointer
//! \param[in] fh  handle to the 'CFFT_F32_STRUCT' object
//! \return pi  pointer to the current input buffer
//!
static inline float32_t * CFFT_f32_getCurrInputPtr(CFFT_F32_STRUCT_Handle fh)
{
    if(fh->Stages%2==0)
    {
        return(fh->OutPtr);
    }else{
        return(fh->InPtr);
    }
}

//! \brief Set the current output buffer pointer
//! \param[in] fh  handle to the 'CFFT_F32_STRUCT' object
//! \param[in] po  pointer to the current output buffer
//!
static inline void CFFT_f32_setCurrOutputPtr(CFFT_F32_STRUCT_Handle fh,
        const float32_t *po)
{
    fh->CurrentOutPtr = (float32_t *)po;
}

//! \brief Get the current output buffer pointer
//! \param[in] fh  handle to the 'CFFT_F32_STRUCT' object
//! \return po  pointer to the current output buffer
//!
static inline float32_t * CFFT_f32_getCurrOutputPtr(CFFT_F32_STRUCT_Handle fh)
{
    return(fh->CurrentOutPtr);
}

//! \brief Set the number of FFT stages
//! \param[in] fh  handle to the 'CFFT_F32_STRUCT' object
//! \param[in] st  number of FFT stages
//!
static inline void CFFT_f32_setStages(CFFT_F32_STRUCT_Handle fh,
                      const uint16_t st)
{
    fh->Stages = st;
}

//! \brief Get the size of the FFT
//! \param[in] fh  handle to the 'CFFT_F32_STRUCT' object
//! \return st  number of FFT stages
//!
static inline uint16_t CFFT_f32_getStages(CFFT_F32_STRUCT_Handle fh)
{
    return(fh->Stages);
}

//! \brief Set the number of FFT stages
//! \param[in] fh  handle to the 'CFFT_F32_STRUCT' object
//! \param[in] sz  size of the FFT
//!
static inline void CFFT_f32_setFFTSize(CFFT_F32_STRUCT_Handle fh,
                      const uint16_t sz)
{
    fh->FFTSize = sz;
}

//! \brief Get the size of the FFT
//! \param[in] fh  handle to the 'CFFT_F32_STRUCT' object
//! \return sz  size of the FFT
//!
static inline uint16_t CFFT_f32_getFFTSize(CFFT_F32_STRUCT_Handle fh)
{
    return(fh->FFTSize);
}

//*****************************************************************************
// function prototypes
//*****************************************************************************
extern void CFFT_f32_asm(CFFT_F32_STRUCT_Handle hndCFFT_F32);
extern void CFFT_f32_c(CFFT_F32_STRUCT_Handle hndCFFT_F32);

extern void CFFT_f32_stage_1_2_asm_opt(int32_t fft_size, float *output, float *input);
extern void CFFT_f32_stages_3_asm_opt( int32_t fft_size, float *output, float *input, int32_t twiddle_size, float *twiddle );
extern void CFFT_f32_stages_4_asm_opt( int32_t fft_size, float *output, float *input, int32_t twiddle_size, float *twiddle );
extern void CFFT_f32_stages_5plus_asm_opt( complex_T *v, complex_T *twiddle, int32_t buf_size, int32_t num_stages, int32_t swap_buffer_address);
extern void CFFT_f32_bitrev(complex_T *in, complex_T *out, uint32_t buf_size);

//! \brief Complex Fast Fourier Transform.
//! 
//! This routine computes the 32-bit floating-point FFT for an N-pt 
//! (\f$ N = 2^{n}, n = 5 : 10\f$) complex input. This function will reorder 
//! the input in bit-reversed format before proceeding with the FFT. The 
//! routine uses two buffers in ping-pong fashion i.e. after each FFT stage the
//! output and input buffers become the input and output buffers respectively 
//! for the next stage. The CFFT_F32 object uses two pointers, CurrentInPtr 
//! and CurrentOutPtr to keep track of the switching. The user can determine
//! the address of the final output by looking at the CurrentInPtr.
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention
//! -# The routine requires the use of two buffers, each of size 2N (32-bit
//! float32_t), for computation; the input buffer must be aligned to a memory
//! address of 4N words (16-bit). Refer to the CFFT linker command file to see
//! an example of this.
//! -# If alignment is not possible the user can use the alternative, albeit 
//! slower, function CFFT_f32u
//! \warning This function is not re-entrant as it uses global variables to
//! store certain parameters
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 2334
//! <tr><td> 128     <td> 5032
//! <tr><td> 256     <td> 11024
//! <tr><td> 512     <td> 24250
//! <tr><td> 1024    <td> 53220
//! </table>
//
extern void CFFT_f32(CFFT_F32_STRUCT_Handle hndCFFT_F32);

//! \brief Complex Data Bit Reversal
//! 
//! This routine will reorder an N point complex data set in bit reverse order.
//! It can be run in-place, i.e. with the input and output buffer pointers
//! pointing to the same array, or off-place with different input and output
//! buffers
//!
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention
//! -# The buffer must be aligned to a memory address of 4N words (16-bit).
//! Refer to the CFFT linker command file to see an example of this.
//! -# You could use the bit-reversal function off-place to preserve the original
//! input while calling the in-place FFT function, CFFT_f32i() or CFFT_f32it(),
//! on the bit reversed buffer
//! -# This function is mainly to be used prior to calling the in-place variant
//! of the  complex FFT functions, CFFT_f32i() or CFFT_f32it().
//! -# If the user has the space to use two N-point complex float32_t buffers then 
//! use the faster CFFT_f32() or CFFT_f32t() functions that does the bit 
//! reversal as part of its stage 1 computation
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 1621
//! <tr><td> 128     <td> 3217
//! <tr><td> 256     <td> 6352
//! <tr><td> 512     <td> 13968
//! <tr><td> 1024    <td> 28500
//! </table>
//
extern void CFFT_f32_brev(CFFT_F32_STRUCT_Handle hndCFFT_F32);

//! \brief Complex Fast Fourier Transform (In-Place).
//! 
//! This routine computes the 32-bit floating-point FFT for an N-pt 
//! (\f$ N = 2^{n}, n = 5 : 10\f$) complex input.  The routine performs its
//! computation in-place, i.e. it uses a single buffer as both input and output
//  for each FFT stage
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention
//! -# This function does not reorder the input in bit-reversed format before 
//! proceeding with the FFT. It assumes the user has already bit reversed the 
//! input by calling CFFT_f32_brev() prior to calling this function
//! -# The routine requires a single buffer of size 2N (32-bit float32_t), for 
//! computation; the buffer must be aligned to a memory address of 4N words 
//! (16-bit). Refer to the CFFT linker command file to see an example of this.
//! -# If alignment is not possible the user can use the alternative, albeit 
//! slower, function CFFT_f32u
//! \warning This function is not re-entrant as it uses global variables to
//! store certain parameters
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 2407
//! <tr><td> 128     <td> 5248
//! <tr><td> 256     <td> 11591
//! <tr><td> 512     <td> 25648
//! <tr><td> 1024    <td> 56535
//! </table>
//
extern void CFFT_f32i(CFFT_F32_STRUCT_Handle hndCFFT_F32);

//! \brief Complex Fast Fourier Transform using a Pre Generated Twiddle
//! Factor Table
//!
//! This routine computes the 32-bit floating-point FFT for an N-pt
//! (\f$ N = 2^{n}, n = 5 : 10\f$) complex input. This function will reorder
//! the input in bit-reversed format before proceeding with the FFT. The
//! routine uses two buffers in ping-pong fashion i.e. after each FFT stage the
//! output and input buffers become the input and output buffers respectively
//! for the next stage. The CFFT_F32 object uses two pointers, CurrentInPtr
//! and CurrentOutPtr to keep track of the switching. The user can determine
//! the address of the final output by looking at the CurrentInPtr.
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention
//! -# The routine requires the use of two buffers, each of size 2N (32-bit
//! float32_t), for computation; the input buffer must be aligned to a memory
//! address of 4N words (16-bit). Refer to the CFFT linker command file to see
//! an example of this.
//! -# If alignment is not possible the user can use the alternative, albeit
//! slower, function CFFT_f32ut
//! \warning This function is not re-entrant as it uses global variables to
//! store certain parameters
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 2334
//! <tr><td> 128     <td> 5032
//! <tr><td> 256     <td> 11026
//! <tr><td> 512     <td> 24250
//! <tr><td> 1024    <td> 53220
//! </table>
//
extern void CFFT_f32t(CFFT_F32_STRUCT_Handle hndCFFT_F32);

//! \brief Complex Fast Fourier Transform (In-Place) using a Pre Generated 
//! Twiddle Factor Table
//!
//! This routine computes the 32-bit floating-point FFT for an N-pt
//! (\f$ N = 2^{n}, n = 5 : 10\f$) complex input. The routine performs its
//! computation in-place, i.e. it uses a single buffer as both input and output
//  for each FFT stage.
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention
//! -# This function does not reorder the input in bit-reversed format before 
//! proceeding with the FFT. It assumes the user has already bit reversed the 
//! input by calling CFFT_f32_brev() prior to calling this function
//! -# The routine requires a single buffer of size 2N (32-bit float32_t), for 
//! computation; the buffer must be aligned to a memory address of 4N words 
//! (16-bit). Refer to the CFFT linker command file to see an example of this.
//! -# If alignment is not possible the user can use the alternative, albeit 
//! slower, function CFFT_f32ut
//! \warning This function is not re-entrant as it uses global variables to
//! store certain parameters
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 2407
//! <tr><td> 128     <td> 5248
//! <tr><td> 256     <td> 11593
//! <tr><td> 512     <td> 25648
//! <tr><td> 1024    <td> 56535
//! </table>
//
extern void CFFT_f32it(CFFT_F32_STRUCT_Handle hndCFFT_F32);

//! \brief Complex Fast Fourier Transform (Unaligned).
//!
//! This routine computes the 32-bit floating-point FFT for an N-pt
//! (\f$ N = 2^{n}, n = 5 : 10\f$) complex input. This function will reorder
//! the input in bit-reversed format before proceeding with the FFT. The
//! routine uses two buffers in ping-pong fashion i.e. after each FFT stage the
//! output and input buffers become the input and output buffers respectively
//! for the next stage. The CFFT_F32 object uses two pointers, CurrentInPtr
//! and CurrentOutPtr to keep track of the switching. The user can determine
//! the address of the final output by looking at the CurrentInPtr.
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention
//! -# The routine requires the use of two buffers, each of size 2N (32-bit
//! float32_t), for computation; the input buffer need not be aligned to any
//! boundary.
//! -# If alignment is possible the user can use the faster routine, CFFT_f32
//! \warning This function is not re-entrant as it uses global variables to
//! store certain parameters
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 2787
//! <tr><td> 128     <td> 5933
//! <tr><td> 256     <td> 12821
//! <tr><td> 512     <td> 27839
//! <tr><td> 1024    <td> 60393
//! </table>
//
extern void CFFT_f32u(CFFT_F32_STRUCT_Handle hndCFFT_F32);

//! \brief Complex Fast Fourier Transform (Unaligned) using a statically
//! generated twiddle factor table.
//!
//! This routine computes the 32-bit floating-point FFT for an N-pt
//! (\f$ N = 2^{n}, n = 5 : 10\f$) complex input. This function will reorder
//! the input in bit-reversed format before proceeding with the FFT. The
//! routine uses two buffers in ping-pong fashion i.e. after each FFT stage the
//! output and input buffers become the input and output buffers respectively
//! for the next stage. The CFFT_F32 object uses two pointers, CurrentInPtr
//! and CurrentOutPtr to keep track of the switching. The user can determine
//! the address of the final output by looking at the CurrentInPtr.
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention
//! -# The routine requires the use of two buffers, each of size 2N (32-bit
//! float32_t), for computation; the input buffer need not be aligned to any
//! boundary.
//! -# If alignment is possible the user can use the faster routine, CFFT_f32
//! \warning This function is not re-entrant as it uses global variables to
//! store certain parameters
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 2787
//! <tr><td> 128     <td> 5933
//! <tr><td> 256     <td> 12821
//! <tr><td> 512     <td> 27839
//! <tr><td> 1024    <td> 60393
//! </table>
//
extern void CFFT_f32ut(CFFT_F32_STRUCT_Handle hndCFFT_F32);

//! \brief Generate twiddle factors for the Complex FFT
//!
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention This function is written in C and compiled without optimization 
//! turned on.
//
extern void CFFT_f32_computeTwiddle(CFFT_F32_STRUCT_Handle hndCFFT_F32);

//! \brief Windowing function for the 32-bit complex FFT 
//! \param pBuffer  pointer to the buffer that needs to be windowed
//! \param pWindow  pointer to the windowing table
//! \param size     size of the buffer
//! This function applies the window to only the real portion of the N point
//! complex buffer that has already been reordered in the bit-reversed format
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 316
//! <tr><td> 128     <td> 604
//! <tr><td> 256     <td> 1180
//! <tr><td> 512     <td> 2332
//! <tr><td> 1024    <td> 4636
//! </table>
//
extern void CFFT_f32_win(float32_t *pBuffer, const float32_t *pWindow, 
                         const uint16_t size); 

//! \brief Windowing function for the 32-bit complex FFT 
//! \param pBuffer  pointer to the buffer that needs to be windowed
//! \param pWindow  pointer to the windowing table
//! \param size     size of the buffer
//! This function applies the window to both the real and imaginary parts of
//! the input complex buffer that has already been reordered in the 
//! bit-reversed format
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 595
//! <tr><td> 128     <td> 1171
//! <tr><td> 256     <td> 2323
//! <tr><td> 512     <td> 4627
//! <tr><td> 1024    <td> 9235
//! </table>
//
extern void CFFT_f32_win_dual(float32_t *pBuffer, const float32_t *pWindow, 
                              const uint16_t size); 

//! \brief Complex FFT Magnitude.
//!
//! This module computes the complex FFT magnitude. The output from 
//! CFFT_f32_mag matches the magnitude output from the FFT found in common 
//! mathematics software and Code Composer Studio FFT graphs.
//! If instead a normalized magnitude like that performed by the fixed-point
//! TMS320C28x IQmath FFT library is required, then the CFFT_f32s_mag function
//! can be used. In fixed-point algorithms scaling is performed to avoid 
//! overflowing data.
//! Floating-point calculations do not need this scaling to avoid overflow and 
//! therefore the CFFT_f32_mag function can be used instead.
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention
//! -# The Magnitude buffer does not require memory alignment to a boundary
//! -# For C28x devices that have the TMU0 (or higher) module, use
//! CFFT_f32_mag_TMU0() instead for better performance.
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 1181
//! <tr><td> 128     <td> 2333
//! <tr><td> 256     <td> 4635
//! <tr><td> 512     <td> 9243
//! <tr><td> 1024    <td> 18459
//! </table>
//
extern void CFFT_f32_mag(CFFT_F32_STRUCT_Handle hndCFFT_F32);

//! \brief Complex FFT Magnitude (Scaled).
//!
//! This module computes the scaled complex FFT magnitude. The scaling is
//! \f$\frac{1}{[2^{FFT\_STAGES-1}]}\f$, and is done to match the normalization
//! performed by the fixed-point TMS320C28x IQmath FFT library for overflow
//! avoidance. Floating-point calculations do not need this scaling to avoid
//! overflow and therefore the CFFT_f32_mag function can be used instead. The
//! output from CFFT_f32s_mag matches the magnitude output from the FFT found
//! in common mathematics software and Code Composer Studio FFT graphs.
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention
//! -# The Magnitude buffer does not require memory alignment to a boundary
//! -# For C28x devices that have the TMU0 (or higher) module, use
//! CFFT_f32s_mag_TMU0() instead for better performance.
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 1284
//! <tr><td> 128     <td> 2506
//! <tr><td> 256     <td> 4942
//! <tr><td> 512     <td> 9812
//! <tr><td> 1024    <td> 19546
//! </table>
//
extern void CFFT_f32s_mag(CFFT_F32_STRUCT_Handle hndCFFT_F32);

//! \brief Complex FFT Phase.
//!
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention
//! -# The Phase buffer does not require memory alignment to a boundary
//! -# The phase function calls the atan2 function in the runtime-support
//! library. The phase function has not been optimized at this time.
//! -# The use of the atan2 function in the FPUfastRTS library will speed up
//! this routine. The example for the CFFT has an alternate build configuration
//! (Debug_FASTRTS) where the rts2800_fpu32_fast_supplement.lib is used in
//! place of the standard runtime library rts2800_fpu32.lib.
//! -# For C28x devices that have the TMU0 (or higher) module, use
//! CFFT_f32_phase_TMU0() instead for better performance.
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Using atan2() from the fast RTS library <th>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 3797
//! <tr><td> 128     <td> 7573
//! <tr><td> 256     <td> 14866
//! <tr><td> 512     <td> 29714
//! <tr><td> 1024    <td> 60434
//! </table>
//
extern void CFFT_f32_phase(CFFT_F32_STRUCT_Handle hndCFFT_F32);

//!
//! \brief Unpack the N-point complex FFT output to get the FFT of a 2N point
//!         real sequence
//!
//! In order to get the FFT of a real N-point sequence, we treat the input as
//! an N/2-point complex sequence, take its complex FFT, use the following 
//! properties to get the N-pt Fourier transform of the real sequence
//! 
//! \f[FFT_{n}(k,f) = FFT_{N/2}(k,f_{e})+e^{\frac{-j2{\pi}k}{N}}FFT_{N/2}(k,f_{o})\f]
//! 
//! where \f$f_{e}\f$ is the even elements, \f$f_{o}\f$ the odd elements, 
//! k = 0 to \f$\frac{N}{2}-1\f$ and
//! 
//! \f[ F_{e}(k) = \frac{Z(k) + Z(\frac{N}{2}-k)^{\ast}}{2} \f]
//! \f[ F_{o}(k) = -j\frac{Z(k) - Z(\frac{N}{2}-k)^{\ast}}{2} \f]
//! 
//! We get the first N/2 points of the FFT by combining the above two equations
//! \f[ F(k) = F_{e}(k) + e^{\frac{-j2{\pi}k}{N}}F_{o}(k) \f]
//!
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \note
//! -# The unpack routine yields the spectrum of the real input data; the 
//!    spectrum has a real part that is symmetric, and an imaginary part that 
//!    is antisymmetric about the nyquist frequency. We only need calculate 
//!    half the spectrum, up to the nyquist bin, while the latter half can be 
//!    derived from the first half using the conjugate symmetry properties of 
//!    the spectrum
//!  - the latter half can be derived using the symmetry properties
//! -# The output is written to the buffer pointer to by CurrentOutPtr
//! -# Only use the CFFT_f32t version with this function
//! \sa http://www.engineeringproductivitytools.com/stuff/T0001/PT10.HTM for 
//! the entire derivation
//! \sa \ref SSEC_RFFT_F32_USING_CFFT_F32
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 562
//! <tr><td> 128     <td> 1136
//! <tr><td> 256     <td> 2098
//! <tr><td> 512     <td> 4399
//! <tr><td> 1024    <td> 8241
//! </table>
//
extern void CFFT_f32_unpack(CFFT_F32_STRUCT_Handle hndCFFT_F32);

//!
//! \brief Pack the N/2-point complex FFT output to get the spectrum of an
//!        N-point real sequence
//!
//! In order to reverse the process of the forward real FFT,
//!
//! \f[ F_{e}(k) = \frac{F(k) + F(\frac{N}{2}-k)^{\ast}}{2} \f]
//! \f[ F_{o}(k) = \frac{F(k) - F(\frac{N}{2}-k)^{\ast}}{2} e^{\frac{j2{\pi}k}{N}} \f]
//!
//! where \f$f_{e}\f$ is the even elements, \f$f_{o}\f$ the odd elements, and
//! k = 0 to \f$\frac{N}{2}-1\f$. The array for the IFFT then becomes:
//! \f[ Z(k) = F_{e}(k) + jF_{o}(k), \ k = 0...\frac{N}{2}-1 \f]
//!
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \note
//! -# The output is written to the buffer pointer to by CurrentOutPtr
//! -# Only use the CFFT_f32t version with this function
//! \sa http://www.engineeringproductivitytools.com/stuff/T0001/PT10.HTM for 
//! the entire derivation
//! \sa \ref SSEC_RFFT_F32_USING_CFFT_F32
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 564
//! <tr><td> 128     <td> 1076
//! <tr><td> 256     <td> 2100
//! <tr><td> 512     <td> 4148
//! <tr><td> 1024    <td> 8243
//! </table>
//
extern void CFFT_f32_pack(CFFT_F32_STRUCT_Handle hndCFFT_F32);

#if defined(__TMS320C28XX_TMU__)
//! \brief Complex FFT Magnitude using the TMU0 module.
//!
//! This module computes the complex FFT magnitude. The output from 
//! CFFT_f32_mag matches the magnitude output from the FFT found in common 
//! mathematics software and Code Composer Studio FFT graphs.
//! If instead a normalized magnitude like that performed by the fixed-point
//! TMS320C28x IQmath FFT library is required, then the CFFT_f32s_mag function
//! can be used. In fixed-point algorithms scaling is performed to avoid 
//! overflowing data.
//! Floating-point calculations do not need this scaling to avoid overflow and 
//! therefore the CFFT_f32_mag function can be used instead.
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention
//! -# The Magnitude buffer does not require memory alignment to a boundary
//! -# This function requires a C28x device with TMU0 or higher module.
//! For devices without the TMU, use CFFT_f32_mag() instead.
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 342
//! <tr><td> 128     <td> 662
//! <tr><td> 256     <td> 1302
//! <tr><td> 512     <td> 2582
//! <tr><td> 1024    <td> 5142
//! </table>
//
extern void CFFT_f32_mag_TMU0(CFFT_F32_STRUCT_Handle hndCFFT_F32);

//! \brief Complex FFT Magnitude using the TMU0 module (Scaled).
//!
//! This module computes the scaled complex FFT magnitude. The scaling is
//! \f$\frac{1}{[2^{FFT\_STAGES-1}]}\f$, and is done to match the normalization
//! performed by the fixed-point TMS320C28x IQmath FFT library for overflow 
//! avoidance. Floating-point calculations do not need this scaling to avoid 
//! overflow and therefore the CFFT_f32_mag function can be used instead. The 
//! output from CFFT_f32s_mag matches the magnitude output from the FFT found 
//! in common mathematics software and Code Composer Studio FFT graphs.
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention
//! -# The Magnitude buffer does not require memory alignment to a boundary
//! -# The magnitude calculation calls the sqrt function within the 
//! runtime-support library. The magnitude function has not been optimized at 
//! this time.
//! -# The use of the sqrt function in the FPUfastRTS library will speed up 
//! this routine. The example for the CFFT has an alternate build configuration 
//! (Debug_FASTRTS) where the rts2800_fpu32_fast_supplement.lib is used in 
//! place of the standard runtime library rts2800_fpu32.lib.
//! -# This function requires a C28x device with TMU0 or higher module.
//! For devices without the TMU, use CFFT_f32s_mag() instead.
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 412
//! <tr><td> 128     <td> 769
//! <tr><td> 256     <td> 1476
//! <tr><td> 512     <td> 2889
//! <tr><td> 1024    <td> 5710
//! </table>
//
extern void CFFT_f32s_mag_TMU0(CFFT_F32_STRUCT_Handle hndCFFT_F32);

//! \brief Complex FFT Phase using the TMU0 module.
//!
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention
//! -# The Phase buffer does not require memory alignment to a boundary
//! -# The phase function calls the atan2 function in the runtime-support 
//! library. The phase function has not been optimized at this time.
//! -# The use of the atan2 function in the FPUfastRTS library will speed up 
//! this routine. The example for the CFFT has an alternate build configuration 
//! (Debug_FASTRTS) where the rts2800_fpu32_fast_supplement.lib is used in 
//! place of the standard runtime library rts2800_fpu32.lib.
//! -# This function requires a C28x device with TMU0 or higher module.
//! For devices without the TMU, use CFFT_f32_phase() instead.
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 480
//! <tr><td> 128     <td> 928
//! <tr><td> 256     <td> 1824
//! <tr><td> 512     <td> 3615
//! <tr><td> 1024    <td> 7199
//! </table>
//
extern void CFFT_f32_phase_TMU0(CFFT_F32_STRUCT_Handle hndCFFT_F32);
#endif // __TMS320C28XX_TMU__

//! \example Test_FPU_CFFTF32.c An example showing the initialization of the 
//! structures and execution of the complex FFT routine (along with magnitude 
//! and phase) available in this library

//! \brief Inverse Complex FFT.
//!
//! This routine computes the 32-bit floating-point Inverse FFT for an N-pt 
//! (\f$ N = 2^{n}, n = 5 : 10\f$) complex input. It uses the forward FFT to do
//!  this by first swapping the real and imaginary parts of the input, running
//! the forward FFT and then swapping the real and imaginary parts of the
//! output to get the final answer. The routine uses two buffers in ping-pong
//! fashion i.e. after each FFT stage the output and input buffers become the
//! input and output buffers respectively for the next stage. The CFFT_F32
//! object uses two pointers, CurrentInPtr and CurrentOutPtr to keep track of
//! the switching. The user can determine the address of the final output by
//! looking at the CurrentInPtr.
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention
//! -# The routine requires the use of two buffers, each of size 2N (32-bit
//! float32_t), for computation; the input buffer must be aligned to a memory
//! address of 4N words (16-bit). Refer to the ICFFT linker command file to
//! see an example of this.
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 2808
//! <tr><td> 128     <td> 5954
//! <tr><td> 256     <td> 12843
//! <tr><td> 512     <td> 27861
//! <tr><td> 1024    <td> 60415
//! </table>
//
extern void ICFFT_f32(CFFT_F32_STRUCT_Handle hndCFFT_F32);

//! \brief Inverse Complex FFT using a statically generated twiddle
//! factor table
//!
//! This routine computes the 32-bit floating-point Inverse FFT for an N-pt
//! (\f$ N = 2^{n}, n = 5 : 10\f$) complex input. It uses the forward FFT to do
//!  this by first swapping the real and imaginary parts of the input, running
//! the forward FFT and then swapping the real and imaginary parts of the
//! output to get the final answer. The routine uses two buffers in ping-pong
//! fashion i.e. after each FFT stage the output and input buffers become the
//! input and output buffers respectively for the next stage. The CFFT_F32
//! object uses two pointers, CurrentInPtr and CurrentOutPtr to keep track of
//! the switching. The user can determine the address of the final output by
//! looking at the CurrentInPtr.
//! \param hndCFFT_F32 Pointer to the CFFT_F32 object
//! \attention
//! -# The routine requires the use of two buffers, each of size 2N (32-bit
//! float32_t), for computation; the input buffer must be aligned to a memory
//! address of 4N words (16-bit). Refer to the ICFFT linker command file to
//! see an example of this.
//!
//! <table>
//! <caption id="multi_row">Performance Data</caption>
//! <tr><th> Samples <th> Cycles 
//! <tr><td> 64      <td> 2921
//! <tr><td> 128     <td> 6180
//! <tr><td> 256     <td> 13549
//! <tr><td> 512     <td> 29271
//! <tr><td> 1024    <td> 64256
//! </table>
//
extern void ICFFT_f32t(CFFT_F32_STRUCT_Handle hndCFFT_F32);

// @} //addtogroup

#ifdef __cplusplus
}
#endif /* extern "C" */

#endif   // - end of _CFFT_F32_H_

// End of File
