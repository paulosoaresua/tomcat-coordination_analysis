#include <Python.h>
#include <iostream>
#include "pytensor_mod_helper.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
//////////////////////
////  Support Code
//////////////////////

        /** ParamsType _Params_7084dbbf7c2181f7c442e43c44205c5693e6f346a0c76b0766c24f89c72690a6_887d99bd69a24775c29fca6fbd6102b5406ad2c8112fbd4e054ca1130411367e **/
        #ifndef _PARAMS_7084DBBF7C2181F7C442E43C44205C5693E6F346A0C76B0766C24F89C72690A6_887D99BD69A24775C29FCA6FBD6102B5406AD2C8112FBD4E054CA1130411367E
        #define _PARAMS_7084DBBF7C2181F7C442E43C44205C5693E6F346A0C76B0766C24F89C72690A6_887D99BD69A24775C29FCA6FBD6102B5406AD2C8112FBD4E054CA1130411367E
        struct _Params_7084dbbf7c2181f7c442e43c44205c5693e6f346a0c76b0766c24f89c72690a6_887d99bd69a24775c29fca6fbd6102b5406ad2c8112fbd4e054ca1130411367e {
            /* Attributes, */
            int _Params_7084dbbf7c2181f7c442e43c44205c5693e6f346a0c76b0766c24f89c72690a6_887d99bd69a24775c29fca6fbd6102b5406ad2c8112fbd4e054ca1130411367e_error;
            
        PyArrayObject* augment;
        
            typedef npy_int64 dtype_augment;
            

                typedef npy_bool dtype_inplace;
            
        npy_bool inplace;
        

        PyArrayObject* shuffle;
        
            typedef npy_int64 dtype_shuffle;
            

        PyArrayObject* transposition;
        
            typedef npy_int64 dtype_transposition;
            

            /* Constructor. */
            _Params_7084dbbf7c2181f7c442e43c44205c5693e6f346a0c76b0766c24f89c72690a6_887d99bd69a24775c29fca6fbd6102b5406ad2c8112fbd4e054ca1130411367e() {
                _Params_7084dbbf7c2181f7c442e43c44205c5693e6f346a0c76b0766c24f89c72690a6_887d99bd69a24775c29fca6fbd6102b5406ad2c8112fbd4e054ca1130411367e_error = 0;
                
        augment = NULL;
        

        inplace = 0;
        

        shuffle = NULL;
        

        transposition = NULL;
        
            }

            /* Destructor. */
            ~_Params_7084dbbf7c2181f7c442e43c44205c5693e6f346a0c76b0766c24f89c72690a6_887d99bd69a24775c29fca6fbd6102b5406ad2c8112fbd4e054ca1130411367e() {
                // cleanup() is defined below.
                cleanup();
            }

            /* Cleanup method. */
            void cleanup() {
                
        if (augment) {
            Py_XDECREF(augment);
        }
        


        if (shuffle) {
            Py_XDECREF(shuffle);
        }
        

        if (transposition) {
            Py_XDECREF(transposition);
        }
        
            }

            /* Extraction methods. */
            
            void extract_augment(PyObject* py_augment) {
                
            augment = NULL;
            if (py_augment == Py_None) {
                // We can either fail here or set augment to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {this->setErrorOccurred(); return;}
            }
            if (!PyArray_Check(py_augment)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {this->setErrorOccurred(); return;}
            }
            // We expect NPY_INT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_augment)) {
                PyArrayObject * tmp = (PyArrayObject*) py_augment;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_INT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_INT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_augment),
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
                {this->setErrorOccurred(); return;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_augment) != NPY_INT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_INT64) got %d",
                             NPY_INT64, PyArray_TYPE((PyArrayObject*) py_augment));
                {this->setErrorOccurred(); return;}
            }
            
        augment = (PyArrayObject*)(py_augment);
        Py_XINCREF(augment);
        
            }
            


            void extract_inplace(PyObject* py_inplace) {
                
            if (!PyObject_TypeCheck(py_inplace, &PyBoolArrType_Type))
            {
                PyErr_Format(PyExc_ValueError,
                    "Scalar check failed (npy_bool)");
                {this->setErrorOccurred(); return;}
            }
            
        PyArray_ScalarAsCtype(py_inplace, &inplace);
        
            }
            


            void extract_shuffle(PyObject* py_shuffle) {
                
            shuffle = NULL;
            if (py_shuffle == Py_None) {
                // We can either fail here or set shuffle to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {this->setErrorOccurred(); return;}
            }
            if (!PyArray_Check(py_shuffle)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {this->setErrorOccurred(); return;}
            }
            // We expect NPY_INT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_shuffle)) {
                PyArrayObject * tmp = (PyArrayObject*) py_shuffle;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_INT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_INT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_shuffle),
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
                {this->setErrorOccurred(); return;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_shuffle) != NPY_INT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_INT64) got %d",
                             NPY_INT64, PyArray_TYPE((PyArrayObject*) py_shuffle));
                {this->setErrorOccurred(); return;}
            }
            
        shuffle = (PyArrayObject*)(py_shuffle);
        Py_XINCREF(shuffle);
        
            }
            


            void extract_transposition(PyObject* py_transposition) {
                
            transposition = NULL;
            if (py_transposition == Py_None) {
                // We can either fail here or set transposition to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {this->setErrorOccurred(); return;}
            }
            if (!PyArray_Check(py_transposition)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {this->setErrorOccurred(); return;}
            }
            // We expect NPY_INT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_transposition)) {
                PyArrayObject * tmp = (PyArrayObject*) py_transposition;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_INT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_INT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_transposition),
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
                {this->setErrorOccurred(); return;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_transposition) != NPY_INT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_INT64) got %d",
                             NPY_INT64, PyArray_TYPE((PyArrayObject*) py_transposition));
                {this->setErrorOccurred(); return;}
            }
            
        transposition = (PyArrayObject*)(py_transposition);
        Py_XINCREF(transposition);
        
            }
            

            /* Extract method. */
            
        void extract(PyObject* object, int field_pos) {
            switch(field_pos) {
                // Extraction cases.
                case 0: extract_augment(object); break;
case 1: extract_inplace(object); break;
case 2: extract_shuffle(object); break;
case 3: extract_transposition(object); break;
                // Default case.
                default:
                    PyErr_Format(PyExc_TypeError, "ParamsType: no extraction defined for a field %d.", field_pos);
                    this->setErrorOccurred();
                    break;
            }
        }
        

            /* Other methods. */
            void setErrorOccurred() {
                ++_Params_7084dbbf7c2181f7c442e43c44205c5693e6f346a0c76b0766c24f89c72690a6_887d99bd69a24775c29fca6fbd6102b5406ad2c8112fbd4e054ca1130411367e_error;
            }
            int errorOccurred() {
                return _Params_7084dbbf7c2181f7c442e43c44205c5693e6f346a0c76b0766c24f89c72690a6_887d99bd69a24775c29fca6fbd6102b5406ad2c8112fbd4e054ca1130411367e_error;
            }
        };
        #endif
        /** End ParamsType _Params_7084dbbf7c2181f7c442e43c44205c5693e6f346a0c76b0766c24f89c72690a6_887d99bd69a24775c29fca6fbd6102b5406ad2c8112fbd4e054ca1130411367e **/
        

#define APPLY_SPECIFIC(str) str##_node_mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba_0
#define PARAMS_TYPE _Params_7084dbbf7c2181f7c442e43c44205c5693e6f346a0c76b0766c24f89c72690a6_887d99bd69a24775c29fca6fbd6102b5406ad2c8112fbd4e054ca1130411367e
#define DTYPE_PARAM_augment npy_int64
#define DTYPE_PARAM_inplace npy_bool
#define DTYPE_PARAM_shuffle npy_int64
#define DTYPE_PARAM_transposition npy_int64


int APPLY_SPECIFIC(cpu_dimshuffle)(PyArrayObject *input, PyArrayObject **res,
                                   PARAMS_TYPE *params) {

  // This points to either the original input or a copy we create below.
  // Either way, this is what we should be working on/with.
  PyArrayObject *_input;

  if (*res)
    Py_XDECREF(*res);

  if (params->inplace) {
    _input = input;
    Py_INCREF((PyObject *)_input);
  } else {
    _input = (PyArrayObject *)PyArray_FromAny(
        (PyObject *)input, NULL, 0, 0, NPY_ARRAY_ALIGNED | NPY_ARRAY_ENSURECOPY,
        NULL);
  }

  PyArray_Dims permute;

  if (!PyArray_IntpConverter((PyObject *)params->transposition, &permute)) {
    return 1;
  }

  /*
    res = res.transpose(self.transposition)
  */
  PyArrayObject *transposed_input =
      (PyArrayObject *)PyArray_Transpose(_input, &permute);

  Py_DECREF(_input);

  PyDimMem_FREE(permute.ptr);

  npy_intp *res_shape = PyArray_DIMS(transposed_input);
  npy_intp N_shuffle = PyArray_SIZE(params->shuffle);
  npy_intp N_augment = PyArray_SIZE(params->augment);
  npy_intp N = N_augment + N_shuffle;
  npy_intp *_reshape_shape = PyDimMem_NEW(N);

  if (_reshape_shape == NULL) {
    PyErr_NoMemory();
    return 1;
  }

  /*
    shape = list(res.shape[: len(self.shuffle)])
    for augm in self.augment:
        shape.insert(augm, 1)
  */
  npy_intp aug_idx = 0;
  int res_idx = 0;
  for (npy_intp i = 0; i < N; i++) {
    if (aug_idx < N_augment &&
        i == *((npy_intp *)PyArray_GetPtr(params->augment, &aug_idx))) {
      _reshape_shape[i] = 1;
      aug_idx++;
    } else {
      _reshape_shape[i] = res_shape[res_idx];
      res_idx++;
    }
  }

  PyArray_Dims reshape_shape = {.ptr = _reshape_shape, .len = (int)N};

  /* res = res.reshape(shape) */
  *res = (PyArrayObject *)PyArray_Newshape(transposed_input, &reshape_shape,
                                           NPY_CORDER);

  Py_DECREF(transposed_input);

  PyDimMem_FREE(reshape_shape.ptr);

  if (!*res) {
    return 1;
  }

  return 0;
}

#undef APPLY_SPECIFIC
#undef PARAMS_TYPE
#undef DTYPE_PARAM_augment
#undef DTYPE_PARAM_inplace
#undef DTYPE_PARAM_shuffle
#undef DTYPE_PARAM_transposition

    namespace {
    struct __struct_compiled_op_mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba {
        PyObject* __ERROR;

        PyObject* storage_V3;
PyObject* storage_V1;
PyObject* storage_V5;
        
    PyObject* py_V5;
    
        _Params_7084dbbf7c2181f7c442e43c44205c5693e6f346a0c76b0766c24f89c72690a6_887d99bd69a24775c29fca6fbd6102b5406ad2c8112fbd4e054ca1130411367e* V5;
        

        __struct_compiled_op_mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba() {
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
        ~__struct_compiled_op_mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, PyObject* storage_V3, PyObject* storage_V1, PyObject* storage_V5) {
            Py_XINCREF(storage_V3);
Py_XINCREF(storage_V1);
Py_XINCREF(storage_V5);
            this->storage_V3 = storage_V3;
this->storage_V1 = storage_V1;
this->storage_V5 = storage_V5;
            



    py_V5 = PyList_GET_ITEM(storage_V5, 0);
    {Py_XINCREF(py_V5);}
    
        /* Seems c_init() is not called for a op param. So I call `new` here. */
        V5 = new _Params_7084dbbf7c2181f7c442e43c44205c5693e6f346a0c76b0766c24f89c72690a6_887d99bd69a24775c29fca6fbd6102b5406ad2c8112fbd4e054ca1130411367e;

        { // This need a separate namespace for Clinker
        const char* fields[] = {"augment", "inplace", "shuffle", "transposition"};
        if (py_V5 == Py_None) {
            PyErr_SetString(PyExc_ValueError, "ParamsType: expected an object, not None.");
            {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        return 5;
}
        }
        for (int i = 0; i < 4; ++i) {
            PyObject* o = PyDict_GetItemString(py_V5, fields[i]);
            if (o == NULL) {
                PyErr_Format(PyExc_TypeError, "ParamsType: missing expected attribute \"%s\" in object.", fields[i]);
                {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        return 5;
}
            }
            V5->extract(o, i);
            if (V5->errorOccurred()) {
                /* The extract code from attribute type should have already raised a Python exception,
                 * so we just print the attribute name in stderr. */
                fprintf(stderr, "\nParamsType: error when extracting value for attribute \"%s\".\n", fields[i]);
                {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        return 5;
}
            }
        }
        }
        

            this->__ERROR = __ERROR;
            return 0;
        }
        void cleanup(void) {
            __label_1:

double __DUMMY_1;
__label_3:

double __DUMMY_3;
__label_5:

        delete V5;
        V5 = NULL;
        
    {Py_XDECREF(py_V5);}
    
double __DUMMY_5;
__label_8:

double __DUMMY_8;

            Py_XDECREF(this->storage_V3);
Py_XDECREF(this->storage_V1);
Py_XDECREF(this->storage_V5);
        }
        int run(void) {
            int __failure = 0;
            
    PyObject* py_V1;
    
        PyArrayObject* V1;
        
    PyObject* py_V3;
    
        PyArrayObject* V3;
        
{

    py_V1 = PyList_GET_ITEM(storage_V1, 0);
    {Py_XINCREF(py_V1);}
    
        if (py_V1 == Py_None)
        {
            
        V1 = NULL;
        
        }
        else
        {
            
        V1 = (PyArrayObject*)(py_V1);
        Py_XINCREF(V1);
        
        }
        
{

    py_V3 = PyList_GET_ITEM(storage_V3, 0);
    {Py_XINCREF(py_V3);}
    
        V3 = (PyArrayObject*)(py_V3);
        Py_XINCREF(V3);
        
{

{
// Op class DimShuffle

                #define APPLY_SPECIFIC(str) str##_node_mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba_0
#define PARAMS_TYPE _Params_7084dbbf7c2181f7c442e43c44205c5693e6f346a0c76b0766c24f89c72690a6_887d99bd69a24775c29fca6fbd6102b5406ad2c8112fbd4e054ca1130411367e
#define DTYPE_PARAM_augment npy_int64
#define DTYPE_PARAM_inplace npy_bool
#define DTYPE_PARAM_shuffle npy_int64
#define DTYPE_PARAM_transposition npy_int64
                {
                  if (APPLY_SPECIFIC(cpu_dimshuffle)(V3, &V1, V5) != 0) {
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
                #undef APPLY_SPECIFIC
#undef PARAMS_TYPE
#undef DTYPE_PARAM_augment
#undef DTYPE_PARAM_inplace
#undef DTYPE_PARAM_shuffle
#undef DTYPE_PARAM_transposition
                __label_7:

double __DUMMY_7;

}
__label_6:

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
    

        static int __struct_compiled_op_mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba_executor(__struct_compiled_op_mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba *self) {
            return self->run();
        }

        static void __struct_compiled_op_mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba_destructor(PyObject *capsule) {
            __struct_compiled_op_mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba *self = (__struct_compiled_op_mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba *)PyCapsule_GetContext(capsule);
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
  __struct_compiled_op_mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba* struct_ptr = new __struct_compiled_op_mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
    PyObject* thunk = PyCapsule_New((void*)(&__struct_compiled_op_mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba_executor), NULL, __struct_compiled_op_mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba_destructor);
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
  "mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba",
  NULL,
  -1,
  MyMethods,
};

PyMODINIT_FUNC PyInit_mdb3a1a186c6e0c789bc6494774a591ac6d8ab11d3ed19bf02a4ff96ab64fa9ba(void) {
   import_array();
   
    PyObject *m = PyModule_Create(&moduledef);
    return m;
}
