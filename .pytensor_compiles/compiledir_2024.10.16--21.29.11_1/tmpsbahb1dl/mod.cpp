#include <Python.h>
#include <iostream>
#include "pytensor_mod_helper.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
//////////////////////
////  Support Code
//////////////////////

    #if PY_MAJOR_VERSION >= 3
        #ifndef PyInt_Check
            #define PyInt_Check PyLong_Check
        #endif
        #ifndef PyInt_AsLong
            #define PyInt_AsLong PyLong_AsLong
        #endif
    #endif
    
            #define MODE_ADD 0
            
            #define MODE_MUL 1
            
        #ifdef DEBUG
        int pytensor_enum_to_string_int(int in, char* out) {
            int ret = 0;
            switch(in) {
                
                   case MODE_ADD: sprintf(out, "MODE_ADD"); break;
                   
                   case MODE_MUL: sprintf(out, "MODE_MUL"); break;
                   
                default:
                    PyErr_SetString(PyExc_ValueError, "EnumList:  unknown enum value.");
                    ret = -1;
                    break;
            }
            return ret;
        }
        #endif
        

        /** ParamsType _Params_f03e63b2d9e0345a6e9b4286de42c0cce7430f7c78fcecccb0469a5cb30cf3bd_cea9a3ae8a6043783d52de028c49ab4ff2a01ab75f603c15fb60043048083ab7 **/
        #ifndef _PARAMS_F03E63B2D9E0345A6E9B4286DE42C0CCE7430F7C78FCECCCB0469A5CB30CF3BD_CEA9A3AE8A6043783D52DE028C49AB4FF2A01AB75F603C15FB60043048083AB7
        #define _PARAMS_F03E63B2D9E0345A6E9B4286DE42C0CCE7430F7C78FCECCCB0469A5CB30CF3BD_CEA9A3AE8A6043783D52DE028C49AB4FF2A01AB75F603C15FB60043048083AB7
        struct _Params_f03e63b2d9e0345a6e9b4286de42c0cce7430f7c78fcecccb0469a5cb30cf3bd_cea9a3ae8a6043783d52de028c49ab4ff2a01ab75f603c15fb60043048083ab7 {
            /* Attributes, */
            int _Params_f03e63b2d9e0345a6e9b4286de42c0cce7430f7c78fcecccb0469a5cb30cf3bd_cea9a3ae8a6043783d52de028c49ab4ff2a01ab75f603c15fb60043048083ab7_error;
            
                typedef npy_int32 dtype_c_axis;
            
        npy_int32 c_axis;
        
int mode;

            /* Constructor. */
            _Params_f03e63b2d9e0345a6e9b4286de42c0cce7430f7c78fcecccb0469a5cb30cf3bd_cea9a3ae8a6043783d52de028c49ab4ff2a01ab75f603c15fb60043048083ab7() {
                _Params_f03e63b2d9e0345a6e9b4286de42c0cce7430f7c78fcecccb0469a5cb30cf3bd_cea9a3ae8a6043783d52de028c49ab4ff2a01ab75f603c15fb60043048083ab7_error = 0;
                
        c_axis = 0;
        
mode = (int)0;
            }

            /* Destructor. */
            ~_Params_f03e63b2d9e0345a6e9b4286de42c0cce7430f7c78fcecccb0469a5cb30cf3bd_cea9a3ae8a6043783d52de028c49ab4ff2a01ab75f603c15fb60043048083ab7() {
                // cleanup() is defined below.
                cleanup();
            }

            /* Cleanup method. */
            void cleanup() {
                

            }

            /* Extraction methods. */
            
            void extract_c_axis(PyObject* py_c_axis) {
                
            if (!PyObject_TypeCheck(py_c_axis, &PyInt32ArrType_Type))
            {
                PyErr_Format(PyExc_ValueError,
                    "Scalar check failed (npy_int32)");
                {this->setErrorOccurred(); return;}
            }
            
        PyArray_ScalarAsCtype(py_c_axis, &c_axis);
        
            }
            


            void extract_mode(PyObject* py_mode) {
                
        if (PyInt_Check(py_mode)) {
            mode = (int)PyInt_AsLong(py_mode);
        } else {
            mode = (int)PyFloat_AsDouble(py_mode);
        }
        if (PyErr_Occurred()) {
            {this->setErrorOccurred(); return;}
        }
        
            }
            

            /* Extract method. */
            
        void extract(PyObject* object, int field_pos) {
            switch(field_pos) {
                // Extraction cases.
                case 0: extract_c_axis(object); break;
case 1: extract_mode(object); break;
                // Default case.
                default:
                    PyErr_Format(PyExc_TypeError, "ParamsType: no extraction defined for a field %d.", field_pos);
                    this->setErrorOccurred();
                    break;
            }
        }
        

            /* Other methods. */
            void setErrorOccurred() {
                ++_Params_f03e63b2d9e0345a6e9b4286de42c0cce7430f7c78fcecccb0469a5cb30cf3bd_cea9a3ae8a6043783d52de028c49ab4ff2a01ab75f603c15fb60043048083ab7_error;
            }
            int errorOccurred() {
                return _Params_f03e63b2d9e0345a6e9b4286de42c0cce7430f7c78fcecccb0469a5cb30cf3bd_cea9a3ae8a6043783d52de028c49ab4ff2a01ab75f603c15fb60043048083ab7_error;
            }
        };
        #endif
        /** End ParamsType _Params_f03e63b2d9e0345a6e9b4286de42c0cce7430f7c78fcecccb0469a5cb30cf3bd_cea9a3ae8a6043783d52de028c49ab4ff2a01ab75f603c15fb60043048083ab7 **/
        

    namespace {
    struct __struct_compiled_op_mf85b9d1ed6ef00f71a319facf82b508ffeeb1b1df76307d54948a37e8661276b {
        PyObject* __ERROR;

        PyObject* storage_V3;
PyObject* storage_V1;
PyObject* storage_V5;
        
    PyObject* py_V5;
    
        _Params_f03e63b2d9e0345a6e9b4286de42c0cce7430f7c78fcecccb0469a5cb30cf3bd_cea9a3ae8a6043783d52de028c49ab4ff2a01ab75f603c15fb60043048083ab7* V5;
        

        __struct_compiled_op_mf85b9d1ed6ef00f71a319facf82b508ffeeb1b1df76307d54948a37e8661276b() {
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
        ~__struct_compiled_op_mf85b9d1ed6ef00f71a319facf82b508ffeeb1b1df76307d54948a37e8661276b(void) {
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
        V5 = new _Params_f03e63b2d9e0345a6e9b4286de42c0cce7430f7c78fcecccb0469a5cb30cf3bd_cea9a3ae8a6043783d52de028c49ab4ff2a01ab75f603c15fb60043048083ab7;

        { // This need a separate namespace for Clinker
        const char* fields[] = {"c_axis", "mode"};
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
        for (int i = 0; i < 2; ++i) {
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
// Op class CumOp

                int axis = V5->c_axis;
                if (axis == 0 && PyArray_NDIM(V3) == 1)
                    axis = NPY_MAXDIMS;
                npy_intp shape[1] = { PyArray_SIZE(V3) };
                if(axis == NPY_MAXDIMS && !(V1 && PyArray_DIMS(V1)[0] == shape[0]))
                {
                    Py_XDECREF(V1);
                    V1 = (PyArrayObject*) PyArray_SimpleNew(1, shape, PyArray_TYPE((PyArrayObject*) py_V3));
                }

                else if(axis != NPY_MAXDIMS && !(V1 && PyArray_CompareLists(PyArray_DIMS(V1), PyArray_DIMS(V3), PyArray_NDIM(V3))))
                {
                    Py_XDECREF(V1);
                    V1 = (PyArrayObject*) PyArray_SimpleNew(PyArray_NDIM(V3), PyArray_DIMS(V3), PyArray_TYPE(V3));
                }

                if (!V1)
                    {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};
                {

                    PyObject * t = NULL;
                    if(V5->mode == MODE_ADD)
                        t = PyArray_CumSum(
                            V3, axis,
                            PyArray_TYPE(V3), V1);
                    else if(V5->mode == MODE_MUL)
                        t = PyArray_CumProd(
                            V3, axis,
                            PyArray_TYPE(V3), V1);

                    if (!t){
                       {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};
                    }
                    // Because PyArray_CumSum/CumProd returns a newly created reference on t.
                    Py_XDECREF(t);
                }
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
    

        static int __struct_compiled_op_mf85b9d1ed6ef00f71a319facf82b508ffeeb1b1df76307d54948a37e8661276b_executor(__struct_compiled_op_mf85b9d1ed6ef00f71a319facf82b508ffeeb1b1df76307d54948a37e8661276b *self) {
            return self->run();
        }

        static void __struct_compiled_op_mf85b9d1ed6ef00f71a319facf82b508ffeeb1b1df76307d54948a37e8661276b_destructor(PyObject *capsule) {
            __struct_compiled_op_mf85b9d1ed6ef00f71a319facf82b508ffeeb1b1df76307d54948a37e8661276b *self = (__struct_compiled_op_mf85b9d1ed6ef00f71a319facf82b508ffeeb1b1df76307d54948a37e8661276b *)PyCapsule_GetContext(capsule);
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
  __struct_compiled_op_mf85b9d1ed6ef00f71a319facf82b508ffeeb1b1df76307d54948a37e8661276b* struct_ptr = new __struct_compiled_op_mf85b9d1ed6ef00f71a319facf82b508ffeeb1b1df76307d54948a37e8661276b();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
    PyObject* thunk = PyCapsule_New((void*)(&__struct_compiled_op_mf85b9d1ed6ef00f71a319facf82b508ffeeb1b1df76307d54948a37e8661276b_executor), NULL, __struct_compiled_op_mf85b9d1ed6ef00f71a319facf82b508ffeeb1b1df76307d54948a37e8661276b_destructor);
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
  "mf85b9d1ed6ef00f71a319facf82b508ffeeb1b1df76307d54948a37e8661276b",
  NULL,
  -1,
  MyMethods,
};

PyMODINIT_FUNC PyInit_mf85b9d1ed6ef00f71a319facf82b508ffeeb1b1df76307d54948a37e8661276b(void) {
   import_array();
   
    PyObject *m = PyModule_Create(&moduledef);
    return m;
}
