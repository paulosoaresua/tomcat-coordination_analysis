#include <Python.h>
#include <iostream>
#include "pytensor_mod_helper.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
//////////////////////
////  Support Code
//////////////////////

        /** ParamsType _Params_043ab33fa7493dd43123b3fa0d08dbb0d8a91ff7ac073f23c5e4710cbab3fbd5_8ad568d1533575930c6b022f11527299cf2c41ec3b1dfbcb4a51211023879bb2 **/
        #ifndef _PARAMS_043AB33FA7493DD43123B3FA0D08DBB0D8A91FF7AC073F23C5E4710CBAB3FBD5_8AD568D1533575930C6B022F11527299CF2C41EC3B1DFBCB4A51211023879BB2
        #define _PARAMS_043AB33FA7493DD43123B3FA0D08DBB0D8A91FF7AC073F23C5E4710CBAB3FBD5_8AD568D1533575930C6B022F11527299CF2C41EC3B1DFBCB4A51211023879BB2
        struct _Params_043ab33fa7493dd43123b3fa0d08dbb0d8a91ff7ac073f23c5e4710cbab3fbd5_8ad568d1533575930c6b022f11527299cf2c41ec3b1dfbcb4a51211023879bb2 {
            /* Attributes, */
            int _Params_043ab33fa7493dd43123b3fa0d08dbb0d8a91ff7ac073f23c5e4710cbab3fbd5_8ad568d1533575930c6b022f11527299cf2c41ec3b1dfbcb4a51211023879bb2_error;
            
        PyObject* exc_type;
        

            /* Constructor. */
            _Params_043ab33fa7493dd43123b3fa0d08dbb0d8a91ff7ac073f23c5e4710cbab3fbd5_8ad568d1533575930c6b022f11527299cf2c41ec3b1dfbcb4a51211023879bb2() {
                _Params_043ab33fa7493dd43123b3fa0d08dbb0d8a91ff7ac073f23c5e4710cbab3fbd5_8ad568d1533575930c6b022f11527299cf2c41ec3b1dfbcb4a51211023879bb2_error = 0;
                
        exc_type = NULL;
        
            }

            /* Destructor. */
            ~_Params_043ab33fa7493dd43123b3fa0d08dbb0d8a91ff7ac073f23c5e4710cbab3fbd5_8ad568d1533575930c6b022f11527299cf2c41ec3b1dfbcb4a51211023879bb2() {
                // cleanup() is defined below.
                cleanup();
            }

            /* Cleanup method. */
            void cleanup() {
                
        Py_XDECREF(exc_type);
        
            }

            /* Extraction methods. */
            
            void extract_exc_type(PyObject* py_exc_type) {
                
        Py_INCREF(py_exc_type);
        exc_type = py_exc_type;
        
            }
            

            /* Extract method. */
            
        void extract(PyObject* object, int field_pos) {
            switch(field_pos) {
                // Extraction cases.
                case 0: extract_exc_type(object); break;
                // Default case.
                default:
                    PyErr_Format(PyExc_TypeError, "ParamsType: no extraction defined for a field %d.", field_pos);
                    this->setErrorOccurred();
                    break;
            }
        }
        

            /* Other methods. */
            void setErrorOccurred() {
                ++_Params_043ab33fa7493dd43123b3fa0d08dbb0d8a91ff7ac073f23c5e4710cbab3fbd5_8ad568d1533575930c6b022f11527299cf2c41ec3b1dfbcb4a51211023879bb2_error;
            }
            int errorOccurred() {
                return _Params_043ab33fa7493dd43123b3fa0d08dbb0d8a91ff7ac073f23c5e4710cbab3fbd5_8ad568d1533575930c6b022f11527299cf2c41ec3b1dfbcb4a51211023879bb2_error;
            }
        };
        #endif
        /** End ParamsType _Params_043ab33fa7493dd43123b3fa0d08dbb0d8a91ff7ac073f23c5e4710cbab3fbd5_8ad568d1533575930c6b022f11527299cf2c41ec3b1dfbcb4a51211023879bb2 **/
        

    namespace {
    struct __struct_compiled_op_ma18e22da4af6909f2d188e25596410a39e2fccc2460e334b670b12931df93cc8 {
        PyObject* __ERROR;

        PyObject* storage_V5;
PyObject* storage_V3;
PyObject* storage_V1;
PyObject* storage_V7;
        
    PyObject* py_V7;
    
        _Params_043ab33fa7493dd43123b3fa0d08dbb0d8a91ff7ac073f23c5e4710cbab3fbd5_8ad568d1533575930c6b022f11527299cf2c41ec3b1dfbcb4a51211023879bb2* V7;
        

        __struct_compiled_op_ma18e22da4af6909f2d188e25596410a39e2fccc2460e334b670b12931df93cc8() {
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
        ~__struct_compiled_op_ma18e22da4af6909f2d188e25596410a39e2fccc2460e334b670b12931df93cc8(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, PyObject* storage_V5, PyObject* storage_V3, PyObject* storage_V1, PyObject* storage_V7) {
            Py_XINCREF(storage_V5);
Py_XINCREF(storage_V3);
Py_XINCREF(storage_V1);
Py_XINCREF(storage_V7);
            this->storage_V5 = storage_V5;
this->storage_V3 = storage_V3;
this->storage_V1 = storage_V1;
this->storage_V7 = storage_V7;
            




    py_V7 = PyList_GET_ITEM(storage_V7, 0);
    {Py_XINCREF(py_V7);}
    
        /* Seems c_init() is not called for a op param. So I call `new` here. */
        V7 = new _Params_043ab33fa7493dd43123b3fa0d08dbb0d8a91ff7ac073f23c5e4710cbab3fbd5_8ad568d1533575930c6b022f11527299cf2c41ec3b1dfbcb4a51211023879bb2;

        { // This need a separate namespace for Clinker
        const char* fields[] = {"exc_type"};
        if (py_V7 == Py_None) {
            PyErr_SetString(PyExc_ValueError, "ParamsType: expected an object, not None.");
            {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        return 7;
}
        }
        for (int i = 0; i < 1; ++i) {
            PyObject* o = PyDict_GetItemString(py_V7, fields[i]);
            if (o == NULL) {
                PyErr_Format(PyExc_TypeError, "ParamsType: missing expected attribute \"%s\" in object.", fields[i]);
                {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        return 7;
}
            }
            V7->extract(o, i);
            if (V7->errorOccurred()) {
                /* The extract code from attribute type should have already raised a Python exception,
                 * so we just print the attribute name in stderr. */
                fprintf(stderr, "\nParamsType: error when extracting value for attribute \"%s\".\n", fields[i]);
                {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        return 7;
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

double __DUMMY_5;
__label_7:

        delete V7;
        V7 = NULL;
        
    {Py_XDECREF(py_V7);}
    
double __DUMMY_7;
__label_10:

double __DUMMY_10;

            Py_XDECREF(this->storage_V5);
Py_XDECREF(this->storage_V3);
Py_XDECREF(this->storage_V1);
Py_XDECREF(this->storage_V7);
        }
        int run(void) {
            int __failure = 0;
            
    PyObject* py_V1;
    
        PyArrayObject* V1;
        
    PyObject* py_V3;
    
        PyArrayObject* V3;
        
    PyObject* py_V5;
    
        PyArrayObject* V5;
        
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

    py_V5 = PyList_GET_ITEM(storage_V5, 0);
    {Py_XINCREF(py_V5);}
    
        V5 = (PyArrayObject*)(py_V5);
        Py_XINCREF(V5);
        
{

{
// Op class Assert

            
            if(PyObject_IsTrue((PyObject *)V3) == 0) {
                PyObject * exc_type = V7->exc_type;
                Py_INCREF(exc_type);
                PyErr_SetString(exc_type, "support_shape does not match respective shape dimension");
                Py_XDECREF(exc_type);
                    {
            __failure = 9;
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError,
                    "Unexpected error in an Op's C code. "
                    "No Python exception was set.");
            }
            goto __label_9;}
            }
                    
            Py_XDECREF(V1);
            V1 = V5;
            Py_INCREF(V5);
            __label_9:

double __DUMMY_9;

}
__label_8:

double __DUMMY_8;

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
    

        static int __struct_compiled_op_ma18e22da4af6909f2d188e25596410a39e2fccc2460e334b670b12931df93cc8_executor(__struct_compiled_op_ma18e22da4af6909f2d188e25596410a39e2fccc2460e334b670b12931df93cc8 *self) {
            return self->run();
        }

        static void __struct_compiled_op_ma18e22da4af6909f2d188e25596410a39e2fccc2460e334b670b12931df93cc8_destructor(PyObject *capsule) {
            __struct_compiled_op_ma18e22da4af6909f2d188e25596410a39e2fccc2460e334b670b12931df93cc8 *self = (__struct_compiled_op_ma18e22da4af6909f2d188e25596410a39e2fccc2460e334b670b12931df93cc8 *)PyCapsule_GetContext(capsule);
            delete self;
        }
    
//////////////////////
////  Functions
//////////////////////
static PyObject * instantiate(PyObject * self, PyObject *argtuple) {
  assert(PyTuple_Check(argtuple));
  if (5 != PyTuple_Size(argtuple)){ 
     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected 5, got %i", (int)PyTuple_Size(argtuple));
     return NULL;
  }
  __struct_compiled_op_ma18e22da4af6909f2d188e25596410a39e2fccc2460e334b670b12931df93cc8* struct_ptr = new __struct_compiled_op_ma18e22da4af6909f2d188e25596410a39e2fccc2460e334b670b12931df93cc8();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3),PyTuple_GET_ITEM(argtuple, 4) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
    PyObject* thunk = PyCapsule_New((void*)(&__struct_compiled_op_ma18e22da4af6909f2d188e25596410a39e2fccc2460e334b670b12931df93cc8_executor), NULL, __struct_compiled_op_ma18e22da4af6909f2d188e25596410a39e2fccc2460e334b670b12931df93cc8_destructor);
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
  "ma18e22da4af6909f2d188e25596410a39e2fccc2460e334b670b12931df93cc8",
  NULL,
  -1,
  MyMethods,
};

PyMODINIT_FUNC PyInit_ma18e22da4af6909f2d188e25596410a39e2fccc2460e334b670b12931df93cc8(void) {
   import_array();
   
    PyObject *m = PyModule_Create(&moduledef);
    return m;
}
