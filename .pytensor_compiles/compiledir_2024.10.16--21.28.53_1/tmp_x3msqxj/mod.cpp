#include <Python.h>
#include <iostream>
#include "pytensor_mod_helper.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
//////////////////////
////  Support Code
//////////////////////

    extern "C"
    {

        void xerbla_(char*, void *);

    /***********/
    /* Level 1 */
    /***********/

    /* Single Precision */

        void srot_(const int*, float *, const int*, float *, const int*, const float *, const float *);
        void srotg_(float *,float *,float *,float *);
        void srotm_( const int*, float *, const int*, float *, const int*, const float *);
        void srotmg_(float *,float *,float *,const float *, float *);
        void sswap_( const int*, float *, const int*, float *, const int*);
        void scopy_( const int*, const float *, const int*, float *, const int*);
        void saxpy_( const int*, const float *, const float *, const int*, float *, const int*);
        float sdot_(const int*, const float *, const int*, const float *, const int*);
        void sdot_sub_(const int*, const float *, const int*, const float *, const int*, float *);
        void sdsdot_sub_( const int*, const float *, const float *, const int*, const float *, const int*, float *);
        void sscal_( const int*, const float *, float *, const int*);
        void snrm2_sub_( const int*, const float *, const int*, float *);
        void sasum_sub_( const int*, const float *, const int*, float *);
        void isamax_sub_( const int*, const float * , const int*, const int*);

    /* Double Precision */

        void drot_(const int*, double *, const int*, double *, const int*, const double *, const double *);
        void drotg_(double *,double *,double *,double *);
        void drotm_( const int*, double *, const int*, double *, const int*, const double *);
        void drotmg_(double *,double *,double *,const double *, double *);
        void dswap_( const int*, double *, const int*, double *, const int*);
        void dcopy_( const int*, const double *, const int*, double *, const int*);
        void daxpy_( const int*, const double *, const double *, const int*, double *, const int*);
        void dswap_( const int*, double *, const int*, double *, const int*);
        double ddot_(const int*, const double *, const int*, const double *, const int*);
        void dsdot_sub_(const int*, const float *, const int*, const float *, const int*, double *);
        void ddot_sub_( const int*, const double *, const int*, const double *, const int*, double *);
        void dscal_( const int*, const double *, double *, const int*);
        void dnrm2_sub_( const int*, const double *, const int*, double *);
        void dasum_sub_( const int*, const double *, const int*, double *);
        void idamax_sub_( const int*, const double * , const int*, const int*);

    /* Single Complex Precision */

        void cswap_( const int*, void *, const int*, void *, const int*);
        void ccopy_( const int*, const void *, const int*, void *, const int*);
        void caxpy_( const int*, const void *, const void *, const int*, void *, const int*);
        void cswap_( const int*, void *, const int*, void *, const int*);
        void cdotc_sub_( const int*, const void *, const int*, const void *, const int*, void *);
        void cdotu_sub_( const int*, const void *, const int*, const void *, const int*, void *);
        void cscal_( const int*, const void *, void *, const int*);
        void icamax_sub_( const int*, const void *, const int*, const int*);
        void csscal_( const int*, const float *, void *, const int*);
        void scnrm2_sub_( const int*, const void *, const int*, float *);
        void scasum_sub_( const int*, const void *, const int*, float *);

    /* Double Complex Precision */

        void zswap_( const int*, void *, const int*, void *, const int*);
        void zcopy_( const int*, const void *, const int*, void *, const int*);
        void zaxpy_( const int*, const void *, const void *, const int*, void *, const int*);
        void zswap_( const int*, void *, const int*, void *, const int*);
        void zdotc_sub_( const int*, const void *, const int*, const void *, const int*, void *);
        void zdotu_sub_( const int*, const void *, const int*, const void *, const int*, void *);
        void zdscal_( const int*, const double *, void *, const int*);
        void zscal_( const int*, const void *, void *, const int*);
        void dznrm2_sub_( const int*, const void *, const int*, double *);
        void dzasum_sub_( const int*, const void *, const int*, double *);
        void izamax_sub_( const int*, const void *, const int*, const int*);

    /***********/
    /* Level 2 */
    /***********/

    /* Single Precision */

        void sgemv_(char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void sgbmv_(char*, const int*, const int*, const int*, const int*, const float *,  const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void ssymv_(char*, const int*, const float *, const float *, const int*, const float *,  const int*, const float *, float *, const int*);
        void ssbmv_(char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void sspmv_(char*, const int*, const float *, const float *, const float *, const int*, const float *, float *, const int*);
        void strmv_( char*, char*, char*, const int*, const float *, const int*, float *, const int*);
        void stbmv_( char*, char*, char*, const int*, const int*, const float *, const int*, float *, const int*);
        void strsv_( char*, char*, char*, const int*, const float *, const int*, float *, const int*);
        void stbsv_( char*, char*, char*, const int*, const int*, const float *, const int*, float *, const int*);
        void stpmv_( char*, char*, char*, const int*, const float *, float *, const int*);
        void stpsv_( char*, char*, char*, const int*, const float *, float *, const int*);
        void sger_( const int*, const int*, const float *, const float *, const int*, const float *, const int*, float *, const int*);
        void ssyr_(char*, const int*, const float *, const float *, const int*, float *, const int*);
        void sspr_(char*, const int*, const float *, const float *, const int*, float *);
        void sspr2_(char*, const int*, const float *, const float *, const int*, const float *, const int*,  float *);
        void ssyr2_(char*, const int*, const float *, const float *, const int*, const float *, const int*,  float *, const int*);

    /* Double Precision */

        void dgemv_(char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dgbmv_(char*, const int*, const int*, const int*, const int*, const double *,  const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dsymv_(char*, const int*, const double *, const double *, const int*, const double *,  const int*, const double *, double *, const int*);
        void dsbmv_(char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dspmv_(char*, const int*, const double *, const double *, const double *, const int*, const double *, double *, const int*);
        void dtrmv_( char*, char*, char*, const int*, const double *, const int*, double *, const int*);
        void dtbmv_( char*, char*, char*, const int*, const int*, const double *, const int*, double *, const int*);
        void dtrsv_( char*, char*, char*, const int*, const double *, const int*, double *, const int*);
        void dtbsv_( char*, char*, char*, const int*, const int*, const double *, const int*, double *, const int*);
        void dtpmv_( char*, char*, char*, const int*, const double *, double *, const int*);
        void dtpsv_( char*, char*, char*, const int*, const double *, double *, const int*);
        void dger_( const int*, const int*, const double *, const double *, const int*, const double *, const int*, double *, const int*);
        void dsyr_(char*, const int*, const double *, const double *, const int*, double *, const int*);
        void dspr_(char*, const int*, const double *, const double *, const int*, double *);
        void dspr2_(char*, const int*, const double *, const double *, const int*, const double *, const int*,  double *);
        void dsyr2_(char*, const int*, const double *, const double *, const int*, const double *, const int*,  double *, const int*);

    /* Single Complex Precision */

        void cgemv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void cgbmv_(char*, const int*, const int*, const int*, const int*, const void *,  const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void chemv_(char*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void chbmv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void chpmv_(char*, const int*, const void *, const void *, const void *, const int*, const void *, void *, const int*);
        void ctrmv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
        void ctbmv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
        void ctpmv_( char*, char*, char*, const int*, const void *, void *, const int*);
        void ctrsv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
        void ctbsv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
        void ctpsv_( char*, char*, char*, const int*, const void *, void *,const int*);
        void cgerc_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
        void cgeru_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *,  const int*);
        void cher_(char*, const int*, const float *, const void *, const int*, void *, const int*);
        void cher2_(char*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
        void chpr_(char*, const int*, const float *, const void *, const int*, void *);
        void chpr2_(char*, const int*, const float *, const void *, const int*, const void *, const int*, void *);

    /* Double Complex Precision */

        void zgemv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void zgbmv_(char*, const int*, const int*, const int*, const int*, const void *,  const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void zhemv_(char*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void zhbmv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void zhpmv_(char*, const int*, const void *, const void *, const void *, const int*, const void *, void *, const int*);
        void ztrmv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
        void ztbmv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
        void ztpmv_( char*, char*, char*, const int*, const void *, void *, const int*);
        void ztrsv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
        void ztbsv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
        void ztpsv_( char*, char*, char*, const int*, const void *, void *,const int*);
        void zgerc_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
        void zgeru_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *,  const int*);
        void zher_(char*, const int*, const double *, const void *, const int*, void *, const int*);
        void zher2_(char*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
        void zhpr_(char*, const int*, const double *, const void *, const int*, void *);
        void zhpr2_(char*, const int*, const double *, const void *, const int*, const void *, const int*, void *);

    /***********/
    /* Level 3 */
    /***********/

    /* Single Precision */

        void sgemm_(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void ssymm_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void ssyrk_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, float *, const int*);
        void ssyr2k_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void strmm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);
        void strsm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);

    /* Double Precision */

        void dgemm_(char*, char*, const int*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dsymm_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dsyrk_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, double *, const int*);
        void dsyr2k_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dtrmm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);
        void dtrsm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);

    /* Single Complex Precision */

        void cgemm_(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void csymm_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void chemm_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void csyrk_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, float *, const int*);
        void cherk_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, float *, const int*);
        void csyr2k_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void cher2k_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void ctrmm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);
        void ctrsm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);

    /* Double Complex Precision */

        void zgemm_(char*, char*, const int*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void zsymm_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void zhemm_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void zsyrk_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, double *, const int*);
        void zherk_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, double *, const int*);
        void zsyr2k_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void zher2k_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void ztrmm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);
        void ztrsm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);

    }
    /** C Implementation (with NumPy back-end) of BLAS functions used in PyTensor.
 * Used instead of BLAS when PyTensor flag ``blas__ldflags`` is empty.
 * This file contains some useful header code not templated.
 * File alt_blas_template.c currently contains template code for:
 * - [sd]gemm_
 * - [sd]gemv_
 * - [sd]dot_
 **/

#define alt_fatal_error(message) { if (PyErr_Occurred()) PyErr_Print(); if(message != NULL) fprintf(stderr, message); exit(-1); }

#define alt_trans_to_bool(trans)  (*trans != 'N' && *trans != 'n')

/**Template code for BLAS functions follows in file alt_blas_template.c
 * (as Python string to be used with old formatting).
 * PARAMETERS:
 * float_type: "float" or "double".
 * float_size: 4 for float32 (sgemm_), 8 for float64 (dgemm_).
 * npy_float: "NPY_FLOAT32" or "NPY_FLOAT64".
 * precision: "s" for single, "d" for double.
 * See blas_headers.py for current use.**/
/** Alternative template NumPy-based implementation of BLAS functions used in PyTensor. **/

/* Compute matrix[i][j] = scalar for every position (i, j) in matrix. */
void alt_numpy_memset_inplace_float(PyArrayObject* matrix, const float* scalar) {
    if (PyArray_IS_C_CONTIGUOUS(matrix) && *scalar == (char)(*scalar)) {
        // This will use memset.
        PyArray_FILLWBYTE(matrix, (char)(*scalar));
        return;
    }
    NpyIter* iterator = NpyIter_New(matrix,
        NPY_ITER_READWRITE | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if(iterator == NULL)
        alt_fatal_error("Unable to iterate over a matrix for a memory assignation.");
    NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterator, NULL);
    char** data_ptr = NpyIter_GetDataPtrArray(iterator);
    npy_intp* stride_ptr = NpyIter_GetInnerStrideArray(iterator);
    npy_intp* innersize_ptr = NpyIter_GetInnerLoopSizePtr(iterator);
    do {
        char* data = *data_ptr;
        npy_intp stride = *stride_ptr;
        npy_intp count = *innersize_ptr;
        while(count) {
            *((float*)data) = *scalar;
            data += stride;
            --count;
        }
    } while(get_next(iterator));
    NpyIter_Deallocate(iterator);
}

/* Scalar * Matrix function.
 * Computes: matrix = scalar * matrix. */
void alt_numpy_scale_matrix_inplace_float(const float* scalar, PyArrayObject* matrix) {
    if (*scalar == 1)
        return;
    if (*scalar == 0) {
        alt_numpy_memset_inplace_float(matrix, scalar);
        return;
    }
    NpyIter* iterator = NpyIter_New(matrix,
        NPY_ITER_READWRITE | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if(iterator == NULL)
        alt_fatal_error("Unable to iterate over a matrix "
                        "for a scalar * matrix operation.");
    NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterator, NULL);
    char** data_ptr = NpyIter_GetDataPtrArray(iterator);
    npy_intp* stride_ptr = NpyIter_GetInnerStrideArray(iterator);
    npy_intp* innersize_ptr = NpyIter_GetInnerLoopSizePtr(iterator);
    do {
        char* data = *data_ptr;
        npy_intp stride = *stride_ptr;
        npy_intp count = *innersize_ptr;
        while(count) {
            *((float*)data) *= *scalar;
            data += stride;
            --count;
        }
    } while(get_next(iterator));
    NpyIter_Deallocate(iterator);
}

/* Matrix + Matrix function.
 * Computes: matrix2 = (scalar1 * matrix1) + (scalar2 * matrix2) */
void alt_numpy_matrix_extended_sum_inplace_float(
        const float* scalar1, PyArrayObject* matrix1,
        const float* scalar2, PyArrayObject* matrix2
) {
    if (*scalar1 == 0 && *scalar2 == 0) {
        alt_numpy_memset_inplace_float(matrix2, scalar2);
        return;
    }
    if (*scalar1 == 0) {
        alt_numpy_scale_matrix_inplace_float(scalar2, matrix2);
        return;
    }
    PyArrayObject* op[2]       = {matrix1, matrix2};
    npy_uint32     op_flags[2] = {NPY_ITER_READONLY, NPY_ITER_READWRITE};
    npy_uint32     flags       = 0;
    NpyIter*       iterators   = NpyIter_MultiNew(
            2, op, flags, NPY_CORDER, NPY_NO_CASTING, op_flags, NULL);
    if(iterators == NULL)
        alt_fatal_error("Unable to iterate over some matrices "
                        "for matrix + matrix operation.");
    NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterators, NULL);
    char** data_ptr_array = NpyIter_GetDataPtrArray(iterators);
    if (*scalar2 == 0) {
        do {
            float* from_matrix1 = (float*)data_ptr_array[0];
            float* from_matrix2 = (float*)data_ptr_array[1];
            *from_matrix2 = (*scalar1)*(*from_matrix1);
        } while(get_next(iterators));
    } else {
        do {
            float* from_matrix1 = (float*)data_ptr_array[0];
            float* from_matrix2 = (float*)data_ptr_array[1];
            *from_matrix2 = (*scalar1)*(*from_matrix1) + (*scalar2)*(*from_matrix2);
        } while(get_next(iterators));
    }
    NpyIter_Deallocate(iterators);
}

/* NumPy Wrapping function. Wraps a data into a NumPy's PyArrayObject.
 * By default, data is considered as Fortran-style array (column by column).
 * If to_transpose, data will be considered as C-style array (row by row)
 * with dimensions reversed. */
PyObject* alt_op_float(int to_transpose, float* M, int nrow, int ncol, int LDM, int numpyFlags) {
    npy_intp dims[2];
    npy_intp strides[2];
    if(to_transpose) {
        dims[0] = ncol;
        dims[1] = nrow;
        strides[0] = LDM * 4;
        strides[1] = 4;
    } else {
        dims[0] = nrow;
        dims[1] = ncol;
        strides[0] = 4;
        strides[1] = LDM * 4;
    }
    return PyArray_New(&PyArray_Type, 2, dims, NPY_FLOAT32, strides, M, 0, numpyFlags, NULL);
}

/* Special wrapping case used for matrix C in gemm_ implementation. */
inline PyObject* alt_wrap_fortran_writeable_matrix_float(
    float* matrix, const int* nrow, const int* ncol, const int* LD
) {
    npy_intp dims[2] = {*nrow, *ncol};
    npy_intp strides[2] = {4, (*LD) * 4};
    return PyArray_New(&PyArray_Type, 2, dims, NPY_FLOAT32, strides, matrix, 0, NPY_ARRAY_WRITEABLE, NULL);
}

/* gemm */
void sgemm_(
    char* TRANSA, char* TRANSB, const int* M, const int* N, const int* K,
    const float* ALPHA, float* A, const int* LDA,
    float* B, const int* LDB, const float* BETA,
    float* C, const int* LDC
) {
    if(*M < 0 || *N < 0 || *K < 0 || *LDA < 0 || *LDB < 0 || *LDC < 0)
        alt_fatal_error("The integer arguments passed to sgemm_ must all be at least 0.");
    /* If M or N is null, there is nothing to do with C,
     * as C should contain M*N == 0 items. */
    if(*M == 0 || *N == 0)
        return;
    int nrowa, ncola, nrowb, ncolb;
    int to_transpose_A = alt_trans_to_bool(TRANSA);
    int to_transpose_B = alt_trans_to_bool(TRANSB);
    if(to_transpose_A) {
        nrowa = *K;
        ncola = *M;
    } else {
        nrowa = *M;
        ncola = *K;
    }
    if(to_transpose_B) {
        nrowb = *N;
        ncolb = *K;
    } else {
        nrowb = *K;
        ncolb = *N;
    }
    int computation_flags;
    void* computation_pointer;
    npy_intp* computation_strides;
    npy_intp computation_dims[2] = {*N, *M};
    npy_intp default_computation_strides[2] = {(*LDC) * 4, 4};
    if(*BETA == 0 && *LDC == *M) {
        /* BETA == 0, so C is never read.
         * LDC == M, so C is contiguous in memory
         * (that condition is needed for dot operation, se below).
         * Then we can compute ALPHA*op(A)*op(B) directly in C. */
        computation_flags = NPY_ARRAY_WRITEABLE;
        computation_pointer = C;
        computation_strides = default_computation_strides;
    } else {
        /* Either BETA != 0 (C will be read)
         * or LDC != M (C is not read but is not contiguous in memory).
         * Then in both cases, we need to allocate a new memory
         * to compute ALPHA*op(A)*op(B). */
        computation_flags = 0;
        computation_pointer = NULL;
        computation_strides = NULL;
    }
    /* The memory buffer used to compute op(A)*op(B) (either C or
     * new allocated buffer) will be considered as C-contiguous because
     * the 3rd parameter of PyArray_MatrixProduct2 (used below)
     * expects a C-contiguous array.
     * Also, to avoid some memory copy, transposition conditions
     * for A and B will be reversed, so that the buffer will contain
     * C-contiguous opB_transposed * opA_transposed (N*M matrix).
     * After that, the code that uses the buffer (either the code calling
     * this function, or this function if BETA != 0) just has to
     * consider the buffer as a F-contiguous M*N matrix, so that
     * it will get the transposed of op_B_transposed * op_A_transposed,
     * that is op_A * op_B (M*N matrix) as expected. */
    PyObject* opA_transposed = alt_op_float(!to_transpose_A, A, nrowa, ncola, *LDA, 0);
    PyObject* opB_transposed = alt_op_float(!to_transpose_B, B, nrowb, ncolb, *LDB, 0);
    PyObject* opB_trans_dot_opA_trans = PyArray_New(&PyArray_Type, 2, computation_dims, NPY_FLOAT32,
                                                    computation_strides, computation_pointer, 0,
                                                    computation_flags, NULL);
    PyArray_MatrixProduct2(opB_transposed, opA_transposed, (PyArrayObject*)opB_trans_dot_opA_trans);
    /* PyArray_MatrixProduct2 adds a reference to the output array,
     * which we need to remove to avoid a memory leak. */
    Py_XDECREF(opB_trans_dot_opA_trans);
    if(*BETA == 0) {
        if(*ALPHA != 1.0)
            alt_numpy_scale_matrix_inplace_float(ALPHA, (PyArrayObject*)opB_trans_dot_opA_trans);
        if(*LDC != *M) {
            /* A buffer has been created to compute ALPHA*op(A)*op(B),
             * so we must copy it to the real output, that is C. */
            PyObject* matrix_C = alt_wrap_fortran_writeable_matrix_float(C, M, N, LDC);
            PyObject* alpha_opA_dot_opB = PyArray_Transpose((PyArrayObject*)opB_trans_dot_opA_trans, NULL);
            if(0 != PyArray_CopyInto((PyArrayObject*)matrix_C, (PyArrayObject*)alpha_opA_dot_opB))
                alt_fatal_error("NumPy sgemm_ implementation: unable to copy ALPHA*op(A)*op(B) into C when BETA == 0.");
            Py_XDECREF(alpha_opA_dot_opB);
            Py_XDECREF(matrix_C);
        }
    } else {
        /* C is read, so we must consider it as Fortran-style matrix. */
        PyObject* matrix_C = alt_wrap_fortran_writeable_matrix_float(C, M, N, LDC);
        PyObject* opA_dot_opB = PyArray_Transpose((PyArrayObject*)opB_trans_dot_opA_trans, NULL);
        alt_numpy_matrix_extended_sum_inplace_float(ALPHA, (PyArrayObject*)opA_dot_opB,
                                                             BETA, (PyArrayObject*)matrix_C);
        Py_XDECREF(opA_dot_opB);
        Py_XDECREF(matrix_C);
    }
    Py_XDECREF(opB_trans_dot_opA_trans);
    Py_XDECREF(opB_transposed);
    Py_XDECREF(opA_transposed);
}

/* gemv */
void sgemv_(
    char* TRANS,
    const int* M,
    const int* N,
    const float* ALPHA,
    float* A,
    const int* LDA,
    float* x,
    const int* incx,
    const float* BETA,
    float* y,
    const int* incy
) {
    /**
    If TRANS is 'n' or 'N', computes:
        y = ALPHA * A * x + BETA * y
    Else, computes:
        y = ALPHA * A.T * x + BETA * y
    A is a M*N matrix, A.T is A transposed
    x, y are vectors
    ALPHA, BETA are scalars
    **/

    // If alpha == 0 and beta == 1, we have nothing to do, as alpha*A*x + beta*y == y.
    if (*ALPHA == 0 && *BETA == 1)
        return;
    if (*M < 0 || *N < 0 || *LDA < 0)
        alt_fatal_error("NumPy sgemv_ implementation: M, N and LDA must be at least 0.");
    if (*incx == 0 || *incy == 0)
        alt_fatal_error("NumPy sgemv_ implementation: incx and incy must not be 0.");
    int transpose = alt_trans_to_bool(TRANS);
    int size_x = 0, size_y = 0;
    if (transpose) {
        size_x = *M;
        size_y = *N;
    } else {
        size_x = *N;
        size_y = *M;
    }
    if (*M == 0 || *N == 0) {
        /* A contains M * N == 0 values. y should be empty too, and we have nothing to do. */
        if (size_y != 0)
            alt_fatal_error("NumPy sgemv_ implementation: the output vector should be empty.");
        return;
    }
    /* Vector pointers points to the beginning of memory (see function `pytensor.tensor.blas_c.gemv_c_code`).
     * NumPy seems to expect that pointers points to the first element of the array. */
    if (*incx < 0)
        x += (size_x - 1) * (-*incx);
    if (*incy < 0)
        y += (size_y - 1) * (-*incy);
    PyObject* matrixA = alt_op_float(transpose, A, *M, *N, *LDA, 0);
    PyObject* matrixX = alt_op_float(1, x, 1, size_x, *incx, 0);
    PyObject* matrixY = alt_op_float(1, y, 1, size_y, *incy, NPY_ARRAY_WRITEABLE);
    if (matrixA == NULL || matrixX == NULL || matrixY == NULL)
        alt_fatal_error("NumPy sgemv_ implementation: unable to wrap A, x or y arrays.")
    if (*ALPHA == 0) {
        // Just BETA * y
        alt_numpy_scale_matrix_inplace_float(BETA, (PyArrayObject*)matrixY);
    } else if (*BETA == 0) {
        // We can directly compute alpha * A * x into y if y is C-contiguous.
        if (PyArray_IS_C_CONTIGUOUS((PyArrayObject*)matrixY)) {
            PyArray_MatrixProduct2(matrixA, matrixX, (PyArrayObject*)matrixY);
            // PyArray_MatrixProduct2 adds an extra reference to the output array.
            Py_XDECREF(matrixY);
            alt_numpy_scale_matrix_inplace_float(ALPHA, (PyArrayObject*)matrixY);
        } else {
            // If y is not contiguous, we need a temporar workspace.
            PyObject* tempAX = PyArray_MatrixProduct(matrixA, matrixX);
            if (tempAX == NULL)
                alt_fatal_error("NumPy sgemv_ implementation: Unable to get matrix product.");
            alt_numpy_scale_matrix_inplace_float(ALPHA, (PyArrayObject*)tempAX);
            if(0 != PyArray_CopyInto((PyArrayObject*)matrixY, (PyArrayObject*)tempAX)) {
                alt_fatal_error("NumPy sgemv_ implementation: unable to update output.");
            }
            Py_XDECREF(tempAX);
        }
    } else {
        // We must perform full computation.
        PyObject* tempAX = PyArray_MatrixProduct(matrixA, matrixX);
        if (tempAX == NULL)
            alt_fatal_error("NumPy sgemv_ implementation: unable to get matrix product.");
        // ALPHA * (A * x) + BETA * y.
        alt_numpy_matrix_extended_sum_inplace_float(ALPHA, (PyArrayObject*)tempAX,
                                                             BETA, (PyArrayObject*)matrixY);
        Py_XDECREF(tempAX);
    }
    Py_XDECREF(matrixY);
    Py_XDECREF(matrixX);
    Py_XDECREF(matrixA);
}

/* dot */
float sdot_(
    const int* N,
    float *SX,
    const int *INCX,
    float *SY,
    const int *INCY
) {
    if (*N < 0)
        alt_fatal_error("NumPy sdot_ implementation: N must be at least 0.");
    if (*INCX == 0 || *INCY == 0)
        alt_fatal_error("NumPy sdot_ implementation: INCX and INCY must not be 0.");
    float result = 0;
    int one = 1;
    /* Vector pointers points to the beginning of memory (see function `pytensor.tensor.blas_c.gemv_c_code`).
     * NumPy seems to expect that pointers points to the first element of the array. */
    if (*INCX < 0)
        SX += (*N - 1) * (-*INCX);
    if (*INCY < 0)
        SY += (*N - 1) * (-*INCY);
    // Create vector_x with shape (1, N)
    PyObject* vector_x = alt_op_float(0, SX, 1, *N, *INCX, 0);
    // Create vector_y with shape (N, 1)
    PyObject* vector_y = alt_op_float(1, SY, 1, *N, *INCY, 0);
    // Create output scalar z with shape (1, 1) to wrap `result`.
    PyArrayObject* dot_product = (PyArrayObject*)alt_wrap_fortran_writeable_matrix_float(&result, &one, &one, &one);

    if (vector_x == NULL || vector_y == NULL || dot_product == NULL)
        alt_fatal_error("NumPy sdot_ implementation: unable to wrap x, y or output arrays.");

    // Compute matrix product: (1, N) * (N, 1) => (1, 1)
    PyArray_MatrixProduct2(vector_x, vector_y, dot_product);
    // PyArray_MatrixProduct2 adds an extra reference to the output array.
    Py_XDECREF(dot_product);

    if (PyErr_Occurred())
        alt_fatal_error("NumPy sdot_ implementation: unable to compute dot.");

    Py_XDECREF(dot_product);
    Py_XDECREF(vector_y);
    Py_XDECREF(vector_x);
    return result;
}
/** Alternative template NumPy-based implementation of BLAS functions used in PyTensor. **/

/* Compute matrix[i][j] = scalar for every position (i, j) in matrix. */
void alt_numpy_memset_inplace_double(PyArrayObject* matrix, const double* scalar) {
    if (PyArray_IS_C_CONTIGUOUS(matrix) && *scalar == (char)(*scalar)) {
        // This will use memset.
        PyArray_FILLWBYTE(matrix, (char)(*scalar));
        return;
    }
    NpyIter* iterator = NpyIter_New(matrix,
        NPY_ITER_READWRITE | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if(iterator == NULL)
        alt_fatal_error("Unable to iterate over a matrix for a memory assignation.");
    NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterator, NULL);
    char** data_ptr = NpyIter_GetDataPtrArray(iterator);
    npy_intp* stride_ptr = NpyIter_GetInnerStrideArray(iterator);
    npy_intp* innersize_ptr = NpyIter_GetInnerLoopSizePtr(iterator);
    do {
        char* data = *data_ptr;
        npy_intp stride = *stride_ptr;
        npy_intp count = *innersize_ptr;
        while(count) {
            *((double*)data) = *scalar;
            data += stride;
            --count;
        }
    } while(get_next(iterator));
    NpyIter_Deallocate(iterator);
}

/* Scalar * Matrix function.
 * Computes: matrix = scalar * matrix. */
void alt_numpy_scale_matrix_inplace_double(const double* scalar, PyArrayObject* matrix) {
    if (*scalar == 1)
        return;
    if (*scalar == 0) {
        alt_numpy_memset_inplace_double(matrix, scalar);
        return;
    }
    NpyIter* iterator = NpyIter_New(matrix,
        NPY_ITER_READWRITE | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if(iterator == NULL)
        alt_fatal_error("Unable to iterate over a matrix "
                        "for a scalar * matrix operation.");
    NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterator, NULL);
    char** data_ptr = NpyIter_GetDataPtrArray(iterator);
    npy_intp* stride_ptr = NpyIter_GetInnerStrideArray(iterator);
    npy_intp* innersize_ptr = NpyIter_GetInnerLoopSizePtr(iterator);
    do {
        char* data = *data_ptr;
        npy_intp stride = *stride_ptr;
        npy_intp count = *innersize_ptr;
        while(count) {
            *((double*)data) *= *scalar;
            data += stride;
            --count;
        }
    } while(get_next(iterator));
    NpyIter_Deallocate(iterator);
}

/* Matrix + Matrix function.
 * Computes: matrix2 = (scalar1 * matrix1) + (scalar2 * matrix2) */
void alt_numpy_matrix_extended_sum_inplace_double(
        const double* scalar1, PyArrayObject* matrix1,
        const double* scalar2, PyArrayObject* matrix2
) {
    if (*scalar1 == 0 && *scalar2 == 0) {
        alt_numpy_memset_inplace_double(matrix2, scalar2);
        return;
    }
    if (*scalar1 == 0) {
        alt_numpy_scale_matrix_inplace_double(scalar2, matrix2);
        return;
    }
    PyArrayObject* op[2]       = {matrix1, matrix2};
    npy_uint32     op_flags[2] = {NPY_ITER_READONLY, NPY_ITER_READWRITE};
    npy_uint32     flags       = 0;
    NpyIter*       iterators   = NpyIter_MultiNew(
            2, op, flags, NPY_CORDER, NPY_NO_CASTING, op_flags, NULL);
    if(iterators == NULL)
        alt_fatal_error("Unable to iterate over some matrices "
                        "for matrix + matrix operation.");
    NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterators, NULL);
    char** data_ptr_array = NpyIter_GetDataPtrArray(iterators);
    if (*scalar2 == 0) {
        do {
            double* from_matrix1 = (double*)data_ptr_array[0];
            double* from_matrix2 = (double*)data_ptr_array[1];
            *from_matrix2 = (*scalar1)*(*from_matrix1);
        } while(get_next(iterators));
    } else {
        do {
            double* from_matrix1 = (double*)data_ptr_array[0];
            double* from_matrix2 = (double*)data_ptr_array[1];
            *from_matrix2 = (*scalar1)*(*from_matrix1) + (*scalar2)*(*from_matrix2);
        } while(get_next(iterators));
    }
    NpyIter_Deallocate(iterators);
}

/* NumPy Wrapping function. Wraps a data into a NumPy's PyArrayObject.
 * By default, data is considered as Fortran-style array (column by column).
 * If to_transpose, data will be considered as C-style array (row by row)
 * with dimensions reversed. */
PyObject* alt_op_double(int to_transpose, double* M, int nrow, int ncol, int LDM, int numpyFlags) {
    npy_intp dims[2];
    npy_intp strides[2];
    if(to_transpose) {
        dims[0] = ncol;
        dims[1] = nrow;
        strides[0] = LDM * 8;
        strides[1] = 8;
    } else {
        dims[0] = nrow;
        dims[1] = ncol;
        strides[0] = 8;
        strides[1] = LDM * 8;
    }
    return PyArray_New(&PyArray_Type, 2, dims, NPY_FLOAT64, strides, M, 0, numpyFlags, NULL);
}

/* Special wrapping case used for matrix C in gemm_ implementation. */
inline PyObject* alt_wrap_fortran_writeable_matrix_double(
    double* matrix, const int* nrow, const int* ncol, const int* LD
) {
    npy_intp dims[2] = {*nrow, *ncol};
    npy_intp strides[2] = {8, (*LD) * 8};
    return PyArray_New(&PyArray_Type, 2, dims, NPY_FLOAT64, strides, matrix, 0, NPY_ARRAY_WRITEABLE, NULL);
}

/* gemm */
void dgemm_(
    char* TRANSA, char* TRANSB, const int* M, const int* N, const int* K,
    const double* ALPHA, double* A, const int* LDA,
    double* B, const int* LDB, const double* BETA,
    double* C, const int* LDC
) {
    if(*M < 0 || *N < 0 || *K < 0 || *LDA < 0 || *LDB < 0 || *LDC < 0)
        alt_fatal_error("The integer arguments passed to dgemm_ must all be at least 0.");
    /* If M or N is null, there is nothing to do with C,
     * as C should contain M*N == 0 items. */
    if(*M == 0 || *N == 0)
        return;
    int nrowa, ncola, nrowb, ncolb;
    int to_transpose_A = alt_trans_to_bool(TRANSA);
    int to_transpose_B = alt_trans_to_bool(TRANSB);
    if(to_transpose_A) {
        nrowa = *K;
        ncola = *M;
    } else {
        nrowa = *M;
        ncola = *K;
    }
    if(to_transpose_B) {
        nrowb = *N;
        ncolb = *K;
    } else {
        nrowb = *K;
        ncolb = *N;
    }
    int computation_flags;
    void* computation_pointer;
    npy_intp* computation_strides;
    npy_intp computation_dims[2] = {*N, *M};
    npy_intp default_computation_strides[2] = {(*LDC) * 8, 8};
    if(*BETA == 0 && *LDC == *M) {
        /* BETA == 0, so C is never read.
         * LDC == M, so C is contiguous in memory
         * (that condition is needed for dot operation, se below).
         * Then we can compute ALPHA*op(A)*op(B) directly in C. */
        computation_flags = NPY_ARRAY_WRITEABLE;
        computation_pointer = C;
        computation_strides = default_computation_strides;
    } else {
        /* Either BETA != 0 (C will be read)
         * or LDC != M (C is not read but is not contiguous in memory).
         * Then in both cases, we need to allocate a new memory
         * to compute ALPHA*op(A)*op(B). */
        computation_flags = 0;
        computation_pointer = NULL;
        computation_strides = NULL;
    }
    /* The memory buffer used to compute op(A)*op(B) (either C or
     * new allocated buffer) will be considered as C-contiguous because
     * the 3rd parameter of PyArray_MatrixProduct2 (used below)
     * expects a C-contiguous array.
     * Also, to avoid some memory copy, transposition conditions
     * for A and B will be reversed, so that the buffer will contain
     * C-contiguous opB_transposed * opA_transposed (N*M matrix).
     * After that, the code that uses the buffer (either the code calling
     * this function, or this function if BETA != 0) just has to
     * consider the buffer as a F-contiguous M*N matrix, so that
     * it will get the transposed of op_B_transposed * op_A_transposed,
     * that is op_A * op_B (M*N matrix) as expected. */
    PyObject* opA_transposed = alt_op_double(!to_transpose_A, A, nrowa, ncola, *LDA, 0);
    PyObject* opB_transposed = alt_op_double(!to_transpose_B, B, nrowb, ncolb, *LDB, 0);
    PyObject* opB_trans_dot_opA_trans = PyArray_New(&PyArray_Type, 2, computation_dims, NPY_FLOAT64,
                                                    computation_strides, computation_pointer, 0,
                                                    computation_flags, NULL);
    PyArray_MatrixProduct2(opB_transposed, opA_transposed, (PyArrayObject*)opB_trans_dot_opA_trans);
    /* PyArray_MatrixProduct2 adds a reference to the output array,
     * which we need to remove to avoid a memory leak. */
    Py_XDECREF(opB_trans_dot_opA_trans);
    if(*BETA == 0) {
        if(*ALPHA != 1.0)
            alt_numpy_scale_matrix_inplace_double(ALPHA, (PyArrayObject*)opB_trans_dot_opA_trans);
        if(*LDC != *M) {
            /* A buffer has been created to compute ALPHA*op(A)*op(B),
             * so we must copy it to the real output, that is C. */
            PyObject* matrix_C = alt_wrap_fortran_writeable_matrix_double(C, M, N, LDC);
            PyObject* alpha_opA_dot_opB = PyArray_Transpose((PyArrayObject*)opB_trans_dot_opA_trans, NULL);
            if(0 != PyArray_CopyInto((PyArrayObject*)matrix_C, (PyArrayObject*)alpha_opA_dot_opB))
                alt_fatal_error("NumPy dgemm_ implementation: unable to copy ALPHA*op(A)*op(B) into C when BETA == 0.");
            Py_XDECREF(alpha_opA_dot_opB);
            Py_XDECREF(matrix_C);
        }
    } else {
        /* C is read, so we must consider it as Fortran-style matrix. */
        PyObject* matrix_C = alt_wrap_fortran_writeable_matrix_double(C, M, N, LDC);
        PyObject* opA_dot_opB = PyArray_Transpose((PyArrayObject*)opB_trans_dot_opA_trans, NULL);
        alt_numpy_matrix_extended_sum_inplace_double(ALPHA, (PyArrayObject*)opA_dot_opB,
                                                             BETA, (PyArrayObject*)matrix_C);
        Py_XDECREF(opA_dot_opB);
        Py_XDECREF(matrix_C);
    }
    Py_XDECREF(opB_trans_dot_opA_trans);
    Py_XDECREF(opB_transposed);
    Py_XDECREF(opA_transposed);
}

/* gemv */
void dgemv_(
    char* TRANS,
    const int* M,
    const int* N,
    const double* ALPHA,
    double* A,
    const int* LDA,
    double* x,
    const int* incx,
    const double* BETA,
    double* y,
    const int* incy
) {
    /**
    If TRANS is 'n' or 'N', computes:
        y = ALPHA * A * x + BETA * y
    Else, computes:
        y = ALPHA * A.T * x + BETA * y
    A is a M*N matrix, A.T is A transposed
    x, y are vectors
    ALPHA, BETA are scalars
    **/

    // If alpha == 0 and beta == 1, we have nothing to do, as alpha*A*x + beta*y == y.
    if (*ALPHA == 0 && *BETA == 1)
        return;
    if (*M < 0 || *N < 0 || *LDA < 0)
        alt_fatal_error("NumPy dgemv_ implementation: M, N and LDA must be at least 0.");
    if (*incx == 0 || *incy == 0)
        alt_fatal_error("NumPy dgemv_ implementation: incx and incy must not be 0.");
    int transpose = alt_trans_to_bool(TRANS);
    int size_x = 0, size_y = 0;
    if (transpose) {
        size_x = *M;
        size_y = *N;
    } else {
        size_x = *N;
        size_y = *M;
    }
    if (*M == 0 || *N == 0) {
        /* A contains M * N == 0 values. y should be empty too, and we have nothing to do. */
        if (size_y != 0)
            alt_fatal_error("NumPy dgemv_ implementation: the output vector should be empty.");
        return;
    }
    /* Vector pointers points to the beginning of memory (see function `pytensor.tensor.blas_c.gemv_c_code`).
     * NumPy seems to expect that pointers points to the first element of the array. */
    if (*incx < 0)
        x += (size_x - 1) * (-*incx);
    if (*incy < 0)
        y += (size_y - 1) * (-*incy);
    PyObject* matrixA = alt_op_double(transpose, A, *M, *N, *LDA, 0);
    PyObject* matrixX = alt_op_double(1, x, 1, size_x, *incx, 0);
    PyObject* matrixY = alt_op_double(1, y, 1, size_y, *incy, NPY_ARRAY_WRITEABLE);
    if (matrixA == NULL || matrixX == NULL || matrixY == NULL)
        alt_fatal_error("NumPy dgemv_ implementation: unable to wrap A, x or y arrays.")
    if (*ALPHA == 0) {
        // Just BETA * y
        alt_numpy_scale_matrix_inplace_double(BETA, (PyArrayObject*)matrixY);
    } else if (*BETA == 0) {
        // We can directly compute alpha * A * x into y if y is C-contiguous.
        if (PyArray_IS_C_CONTIGUOUS((PyArrayObject*)matrixY)) {
            PyArray_MatrixProduct2(matrixA, matrixX, (PyArrayObject*)matrixY);
            // PyArray_MatrixProduct2 adds an extra reference to the output array.
            Py_XDECREF(matrixY);
            alt_numpy_scale_matrix_inplace_double(ALPHA, (PyArrayObject*)matrixY);
        } else {
            // If y is not contiguous, we need a temporar workspace.
            PyObject* tempAX = PyArray_MatrixProduct(matrixA, matrixX);
            if (tempAX == NULL)
                alt_fatal_error("NumPy dgemv_ implementation: Unable to get matrix product.");
            alt_numpy_scale_matrix_inplace_double(ALPHA, (PyArrayObject*)tempAX);
            if(0 != PyArray_CopyInto((PyArrayObject*)matrixY, (PyArrayObject*)tempAX)) {
                alt_fatal_error("NumPy dgemv_ implementation: unable to update output.");
            }
            Py_XDECREF(tempAX);
        }
    } else {
        // We must perform full computation.
        PyObject* tempAX = PyArray_MatrixProduct(matrixA, matrixX);
        if (tempAX == NULL)
            alt_fatal_error("NumPy dgemv_ implementation: unable to get matrix product.");
        // ALPHA * (A * x) + BETA * y.
        alt_numpy_matrix_extended_sum_inplace_double(ALPHA, (PyArrayObject*)tempAX,
                                                             BETA, (PyArrayObject*)matrixY);
        Py_XDECREF(tempAX);
    }
    Py_XDECREF(matrixY);
    Py_XDECREF(matrixX);
    Py_XDECREF(matrixA);
}

/* dot */
double ddot_(
    const int* N,
    double *SX,
    const int *INCX,
    double *SY,
    const int *INCY
) {
    if (*N < 0)
        alt_fatal_error("NumPy ddot_ implementation: N must be at least 0.");
    if (*INCX == 0 || *INCY == 0)
        alt_fatal_error("NumPy ddot_ implementation: INCX and INCY must not be 0.");
    double result = 0;
    int one = 1;
    /* Vector pointers points to the beginning of memory (see function `pytensor.tensor.blas_c.gemv_c_code`).
     * NumPy seems to expect that pointers points to the first element of the array. */
    if (*INCX < 0)
        SX += (*N - 1) * (-*INCX);
    if (*INCY < 0)
        SY += (*N - 1) * (-*INCY);
    // Create vector_x with shape (1, N)
    PyObject* vector_x = alt_op_double(0, SX, 1, *N, *INCX, 0);
    // Create vector_y with shape (N, 1)
    PyObject* vector_y = alt_op_double(1, SY, 1, *N, *INCY, 0);
    // Create output scalar z with shape (1, 1) to wrap `result`.
    PyArrayObject* dot_product = (PyArrayObject*)alt_wrap_fortran_writeable_matrix_double(&result, &one, &one, &one);

    if (vector_x == NULL || vector_y == NULL || dot_product == NULL)
        alt_fatal_error("NumPy ddot_ implementation: unable to wrap x, y or output arrays.");

    // Compute matrix product: (1, N) * (N, 1) => (1, 1)
    PyArray_MatrixProduct2(vector_x, vector_y, dot_product);
    // PyArray_MatrixProduct2 adds an extra reference to the output array.
    Py_XDECREF(dot_product);

    if (PyErr_Occurred())
        alt_fatal_error("NumPy ddot_ implementation: unable to compute dot.");

    Py_XDECREF(dot_product);
    Py_XDECREF(vector_y);
    Py_XDECREF(vector_x);
    return result;
}

        template<typename dtype>
        bool batch_gemm(void (*gemm)(char*, char*, const int*, const int*, const int*, const dtype*, const dtype*, const int*, const dtype*, const int*, const dtype*, dtype*, const int*),
                        int type_size, PyArrayObject* xs, PyArrayObject* ys,
                        PyArrayObject* zs) {
            npy_intp *Nx = PyArray_DIMS(xs), *Sx = PyArray_STRIDES(xs);
            npy_intp *Ny = PyArray_DIMS(ys), *Sy = PyArray_STRIDES(ys);
            npy_intp *Nz = PyArray_DIMS(zs), *Sz = PyArray_STRIDES(zs);

            if (Nx[0] != Ny[0]) {
                PyErr_Format(PyExc_ValueError,
                             "Shape mismatch: batch sizes unequal."
                             " x.shape is (%d, %d, %d),"
                             " y.shape is (%d, %d, %d).",
                             Nx[0], Nx[1], Nx[2],
                             Ny[0], Ny[1], Ny[2]);
                return 1;
            }

            if (Nx[2] != Ny[1]) {
                PyErr_Format(PyExc_ValueError,
                             "Shape mismatch: summation axis sizes unequal."
                             " x.shape is (%d, %d, %d),"
                             " y.shape is (%d, %d, %d).",
                             Nx[0], Nx[1], Nx[2],
                             Ny[0], Ny[1], Ny[2]);
                return 1;
            }

            /* encode the stride structure of _x,_y,_z into a single integer. */
            int unit = 0;
            unit |= ((Sx[2] == type_size || Nx[2] == 1) ? 0x0 : (Sx[1] == type_size || Nx[1]==1) ? 0x1 : 0x2) << 8;
            unit |= ((Sy[2] == type_size || Ny[2] == 1) ? 0x0 : (Sy[1] == type_size || Ny[1]==1) ? 0x1 : 0x2) << 4;
            unit |= ((Sz[2] == type_size || Nz[2] == 1) ? 0x0 : (Sz[1] == type_size || Nz[1]==1) ? 0x1 : 0x2) << 0;

            /* create appropriate strides for malformed matrices that are row or column
             * vectors, or empty matrices.
             * In that case, the value of the stride does not really matter, but
             * some versions of BLAS insist that:
             *  - they are not smaller than the number of elements in the array,
             *  - they are not 0.
             */
            int sx_1 = (Nx[1] > 1) ? Sx[1]/type_size : (Nx[2] + 1);
            int sx_2 = (Nx[2] > 1) ? Sx[2]/type_size : (Nx[1] + 1);
            int sy_1 = (Ny[1] > 1) ? Sy[1]/type_size : (Ny[2] + 1);
            int sy_2 = (Ny[2] > 1) ? Sy[2]/type_size : (Ny[1] + 1);
            int sz_1 = (Nz[1] > 1) ? Sz[1]/type_size : (Nz[2] + 1);
            int sz_2 = (Nz[2] > 1) ? Sz[2]/type_size : (Nz[1] + 1);

            dtype* x = (dtype*)PyArray_DATA(xs);
            dtype* y = (dtype*)PyArray_DATA(ys);
            dtype* z = (dtype*)PyArray_DATA(zs);

            dtype a = 1.0;
            dtype b = 0.0;
            char N = 'N';
            char T = 'T';
            int Nz1 = Nz[1], Nz2 = Nz[2], Nx2 = Nx[2];

            // loop over batch axis
            for (int i = 0; i < Nz[0]; i++) {
                switch(unit)
                {
                    case 0x000: gemm(&N, &N, &Nz2, &Nz1, &Nx2, &a, y, &sy_1, x, &sx_1, &b, z, &sz_1); break;
                    case 0x100: gemm(&N, &T, &Nz2, &Nz1, &Nx2, &a, y, &sy_1, x, &sx_2, &b, z, &sz_1); break;
                    case 0x010: gemm(&T, &N, &Nz2, &Nz1, &Nx2, &a, y, &sy_2, x, &sx_1, &b, z, &sz_1); break;
                    case 0x110: gemm(&T, &T, &Nz2, &Nz1, &Nx2, &a, y, &sy_2, x, &sx_2, &b, z, &sz_1); break;
                    case 0x001: gemm(&T, &T, &Nz1, &Nz2, &Nx2, &a, x, &sx_1, y, &sy_1, &b, z, &sz_2); break;
                    case 0x101: gemm(&N, &T, &Nz1, &Nz2, &Nx2, &a, x, &sx_2, y, &sy_1, &b, z, &sz_2); break;
                    case 0x011: gemm(&T, &N, &Nz1, &Nz2, &Nx2, &a, x, &sx_1, y, &sy_2, &b, z, &sz_2); break;
                    case 0x111: gemm(&N, &N, &Nz1, &Nz2, &Nx2, &a, x, &sx_2, y, &sy_2, &b, z, &sz_2); break;
                    default: PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride"); return 1;
                };
                x += Sx[0] / type_size;
                y += Sy[0] / type_size;
                z += Sz[0] / type_size;
            }

            return 0;
        }
        

    namespace {
    struct __struct_compiled_op_m9496a72e0e2dac1a2fdd0b8892f792bb9aeb2498b80b7c829b810b0d8e94568c {
        PyObject* __ERROR;

        PyObject* storage_V5;
PyObject* storage_V3;
PyObject* storage_V1;
        

        __struct_compiled_op_m9496a72e0e2dac1a2fdd0b8892f792bb9aeb2498b80b7c829b810b0d8e94568c() {
            // This is only somewhat safe because we:
            //  1) Are not a virtual class
            //  2) Do not use any virtual classes in the members
            //  3) Deal with mostly POD and pointers

            // If this changes, we would have to revise this, but for
            // now I am tired of chasing segfaults because
            // initialization code had an error and some pointer has
            // a junk value.
            #ifndef PYTENSOR_DONT_MEMSET_STRUCT
            memset(this, 0, sizeof(*this));
            #endif
        }
        ~__struct_compiled_op_m9496a72e0e2dac1a2fdd0b8892f792bb9aeb2498b80b7c829b810b0d8e94568c(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, PyObject* storage_V5, PyObject* storage_V3, PyObject* storage_V1) {
            Py_XINCREF(storage_V5);
Py_XINCREF(storage_V3);
Py_XINCREF(storage_V1);
            this->storage_V5 = storage_V5;
this->storage_V3 = storage_V3;
this->storage_V1 = storage_V1;
            




            this->__ERROR = __ERROR;
            return 0;
        }
        void cleanup(void) {
            __label_1:

double __DUMMY_1;
__label_3:

double __DUMMY_3;
__label_5:

double __DUMMY_5;
__label_8:

double __DUMMY_8;

            Py_XDECREF(this->storage_V5);
Py_XDECREF(this->storage_V3);
Py_XDECREF(this->storage_V1);
        }
        int run(void) {
            int __failure = 0;
            
    PyObject* py_V1;
    
        PyArrayObject* V1;
        
            typedef npy_float64 dtype_V1;
            
    PyObject* py_V3;
    
        PyArrayObject* V3;
        
            typedef npy_float64 dtype_V3;
            
    PyObject* py_V5;
    
        PyArrayObject* V5;
        
            typedef npy_float64 dtype_V5;
            
{

    py_V1 = PyList_GET_ITEM(storage_V1, 0);
    {Py_XINCREF(py_V1);}
    
        if (py_V1 == Py_None)
        {
            
        V1 = NULL;
        
        }
        else
        {
            
            V1 = NULL;
            if (py_V1 == Py_None) {
                // We can either fail here or set V1 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
            }
            if (!PyArray_Check(py_V1)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V1)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V1;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V1),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V1) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V1));
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
            }
            
        V1 = (PyArrayObject*)(py_V1);
        Py_XINCREF(V1);
        
        }
        
{

    py_V3 = PyList_GET_ITEM(storage_V3, 0);
    {Py_XINCREF(py_V3);}
    
            V3 = NULL;
            if (py_V3 == Py_None) {
                // We can either fail here or set V3 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_4;}
            }
            if (!PyArray_Check(py_V3)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_4;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V3)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V3;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V3),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_4;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V3) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V3));
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_4;}
            }
            
        V3 = (PyArrayObject*)(py_V3);
        Py_XINCREF(V3);
        
{

    py_V5 = PyList_GET_ITEM(storage_V5, 0);
    {Py_XINCREF(py_V5);}
    
            V5 = NULL;
            if (py_V5 == Py_None) {
                // We can either fail here or set V5 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_6;}
            }
            if (!PyArray_Check(py_V5)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_6;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V5)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V5;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V5),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_6;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V5) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V5));
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_6;}
            }
            
        V5 = (PyArrayObject*)(py_V5);
        Py_XINCREF(V5);
        
{
// Op class BatchedDot

        int type_num = PyArray_DESCR(V5)->type_num;
        int type_size = PyArray_DESCR(V5)->elsize; // in bytes

        // xs, ys, zs will point to views onto V5, V3, V1
        PyArrayObject *xs = 0, *ys = 0, *zs = 0;

        if (PyArray_NDIM(V5) != 3) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(x) != 3. rank(x) is %d.",
                         PyArray_NDIM(V5));
            {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};
        }
        if (PyArray_NDIM(V3) != 3) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(y) != 3. rank(y) is %d.",
                         PyArray_NDIM(V3));
            {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};
        }
        if (V1 && PyArray_NDIM(V1) != 3) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(z) != 3. rank(z) is %d.",
                         PyArray_NDIM(V1));
            {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};
        }

        // allocate output
        
            if (NULL == V1 || !(PyArray_DIMS(V1)[0] == PyArray_DIMS(V5)[0] && PyArray_DIMS(V1)[1] == PyArray_DIMS(V5)[1] && PyArray_DIMS(V1)[2] == PyArray_DIMS(V3)[2])  || !(PyArray_STRIDES(V1)[1] > 0 && PyArray_STRIDES(V1)[1] % type_size == 0 && PyArray_STRIDES(V1)[2] > 0 && PyArray_STRIDES(V1)[2] % type_size == 0 && (PyArray_STRIDES(V1)[1] == type_size || PyArray_STRIDES(V1)[2] == type_size)))
            {
                npy_intp dims[3] = {PyArray_DIMS(V5)[0], PyArray_DIMS(V5)[1], PyArray_DIMS(V3)[2]};
                Py_XDECREF(V1);
                V1 = (PyArrayObject*)PyArray_SimpleNew(
                    3, dims, PyArray_TYPE(V5));
                if(!V1) {
                    PyErr_SetString(PyExc_MemoryError,
                                    "failed to alloc BatchedDot output");
                    {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;}
                }
            }
        
        // reallocate any noncontiguous arrays or arrays with invalid strides
        
                if (!(PyArray_STRIDES(V5)[1] > 0 && PyArray_STRIDES(V5)[1] % type_size == 0 && PyArray_STRIDES(V5)[2] > 0 && PyArray_STRIDES(V5)[2] % type_size == 0 && (PyArray_STRIDES(V5)[1] == type_size || PyArray_STRIDES(V5)[2] == type_size))) {
                    PyArrayObject * _copy = (PyArrayObject *) PyArray_Copy(V5);
                    if (!_copy)
                        {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;}
                    Py_XDECREF(V5);
                    V5 = _copy;
                }
            

                if (!(PyArray_STRIDES(V3)[1] > 0 && PyArray_STRIDES(V3)[1] % type_size == 0 && PyArray_STRIDES(V3)[2] > 0 && PyArray_STRIDES(V3)[2] % type_size == 0 && (PyArray_STRIDES(V3)[1] == type_size || PyArray_STRIDES(V3)[2] == type_size))) {
                    PyArrayObject * _copy = (PyArrayObject *) PyArray_Copy(V3);
                    if (!_copy)
                        {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;}
                    Py_XDECREF(V3);
                    V3 = _copy;
                }
            
        // add dims to make sure everything is tensor3
        xs = V5; Py_XINCREF(xs);
ys = V3; Py_XINCREF(ys);
zs = V1; Py_XINCREF(zs);
        // from here on, use xs, ys and zs as they are tensor3 and share memory
        // with the original V5, V3 and V1 arrays.

        if ((PyArray_DESCR(xs)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(xs)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(x) is not double or float"); {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};}

        if ((PyArray_DESCR(ys)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(ys)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(y) is not double or float"); {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};}

        if ((PyArray_DESCR(zs)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(zs)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(z) is not double or float"); {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};}

        if ((PyArray_DESCR(xs)->type_num != PyArray_DESCR(ys)->type_num)
            ||(PyArray_DESCR(xs)->type_num != PyArray_DESCR(zs)->type_num))
        { PyErr_SetString(PyExc_NotImplementedError, "type(x), type(y), type(z) are not all the same"); {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;}; }

        switch (type_num)
        {
            case NPY_FLOAT:
            if (batch_gemm<float>(sgemm_, type_size, xs, ys, zs)) {
                {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};
            }
            break;
            case NPY_DOUBLE:
            if (batch_gemm<double>(dgemm_, type_size, xs, ys, zs)) {
                {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};
            }
            break;
        }
        __label_7:

        // clean up views
        Py_XDECREF(xs); xs = 0;
        Py_XDECREF(ys); ys = 0;
        Py_XDECREF(zs); zs = 0;
        
double __DUMMY_7;

}
__label_6:

        if (V5) {
            Py_XDECREF(V5);
        }
        
    {Py_XDECREF(py_V5);}
    
double __DUMMY_6;

}
__label_4:

        if (V3) {
            Py_XDECREF(V3);
        }
        
    {Py_XDECREF(py_V3);}
    
double __DUMMY_4;

}
__label_2:

    if (!__failure) {
      
        {Py_XDECREF(py_V1);}
        if (!V1) {
            Py_INCREF(Py_None);
            py_V1 = Py_None;
        }
        else if ((void*)py_V1 != (void*)V1) {
            py_V1 = (PyObject*)V1;
        }

        {Py_XINCREF(py_V1);}

        if (V1 && !PyArray_ISALIGNED((PyArrayObject*) py_V1)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %ld"
                         " with %ld dimensions, with 3 last dims "
                         "%ld, %ld, %ld"
                         " and 3 last strides %ld %ld, %ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_V1),
                         (long int) PyArray_NDIM(V1),
                         (long int) (PyArray_NDIM(V1) >= 3 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-3] : -1),
                         (long int) (PyArray_NDIM(V1) >= 2 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-2] : -1),
                         (long int) (PyArray_NDIM(V1) >= 1 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-1] : -1),
                         (long int) (PyArray_NDIM(V1) >= 3 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-3] : -1),
                         (long int) (PyArray_NDIM(V1) >= 2 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-2] : -1),
                         (long int) (PyArray_NDIM(V1) >= 1 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-1] : -1)
        );
            {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V1, 0);
      {Py_XINCREF(py_V1);}
      PyList_SET_ITEM(storage_V1, 0, py_V1);
      {Py_XDECREF(old);}
    }
    
        if (V1) {
            Py_XDECREF(V1);
        }
        
    {Py_XDECREF(py_V1);}
    
double __DUMMY_2;

}

            
        if (__failure) {
            // When there is a failure, this code puts the exception
            // in __ERROR.
            PyObject* err_type = NULL;
            PyObject* err_msg = NULL;
            PyObject* err_traceback = NULL;
            PyErr_Fetch(&err_type, &err_msg, &err_traceback);
            if (!err_type) {err_type = Py_None;Py_INCREF(Py_None);}
            if (!err_msg) {err_msg = Py_None; Py_INCREF(Py_None);}
            if (!err_traceback) {err_traceback = Py_None; Py_INCREF(Py_None);}
            PyObject* old_err_type = PyList_GET_ITEM(__ERROR, 0);
            PyObject* old_err_msg = PyList_GET_ITEM(__ERROR, 1);
            PyObject* old_err_traceback = PyList_GET_ITEM(__ERROR, 2);
            PyList_SET_ITEM(__ERROR, 0, err_type);
            PyList_SET_ITEM(__ERROR, 1, err_msg);
            PyList_SET_ITEM(__ERROR, 2, err_traceback);
            {Py_XDECREF(old_err_type);}
            {Py_XDECREF(old_err_msg);}
            {Py_XDECREF(old_err_traceback);}
        }
        // The failure code is returned to index what code block failed.
        return __failure;
        
        }
    };
    }
    

        static int __struct_compiled_op_m9496a72e0e2dac1a2fdd0b8892f792bb9aeb2498b80b7c829b810b0d8e94568c_executor(__struct_compiled_op_m9496a72e0e2dac1a2fdd0b8892f792bb9aeb2498b80b7c829b810b0d8e94568c *self) {
            return self->run();
        }

        static void __struct_compiled_op_m9496a72e0e2dac1a2fdd0b8892f792bb9aeb2498b80b7c829b810b0d8e94568c_destructor(PyObject *capsule) {
            __struct_compiled_op_m9496a72e0e2dac1a2fdd0b8892f792bb9aeb2498b80b7c829b810b0d8e94568c *self = (__struct_compiled_op_m9496a72e0e2dac1a2fdd0b8892f792bb9aeb2498b80b7c829b810b0d8e94568c *)PyCapsule_GetContext(capsule);
            delete self;
        }
    
//////////////////////
////  Functions
//////////////////////
static PyObject * instantiate(PyObject * self, PyObject *argtuple) {
  assert(PyTuple_Check(argtuple));
  if (4 != PyTuple_Size(argtuple)){ 
     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected 4, got %i", (int)PyTuple_Size(argtuple));
     return NULL;
  }
  __struct_compiled_op_m9496a72e0e2dac1a2fdd0b8892f792bb9aeb2498b80b7c829b810b0d8e94568c* struct_ptr = new __struct_compiled_op_m9496a72e0e2dac1a2fdd0b8892f792bb9aeb2498b80b7c829b810b0d8e94568c();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
    PyObject* thunk = PyCapsule_New((void*)(&__struct_compiled_op_m9496a72e0e2dac1a2fdd0b8892f792bb9aeb2498b80b7c829b810b0d8e94568c_executor), NULL, __struct_compiled_op_m9496a72e0e2dac1a2fdd0b8892f792bb9aeb2498b80b7c829b810b0d8e94568c_destructor);
    if (thunk != NULL && PyCapsule_SetContext(thunk, struct_ptr) != 0) {
        PyErr_Clear();
        Py_DECREF(thunk);
        thunk = NULL;
    }

  return thunk; }

//////////////////////
////  Module init
//////////////////////
static PyMethodDef MyMethods[] = {
	{"instantiate", instantiate, METH_VARARGS, "undocumented"} ,
	{NULL, NULL, 0, NULL}
};
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "m9496a72e0e2dac1a2fdd0b8892f792bb9aeb2498b80b7c829b810b0d8e94568c",
  NULL,
  -1,
  MyMethods,
};

PyMODINIT_FUNC PyInit_m9496a72e0e2dac1a2fdd0b8892f792bb9aeb2498b80b7c829b810b0d8e94568c(void) {
   import_array();
   
    PyObject *m = PyModule_Create(&moduledef);
    return m;
}
