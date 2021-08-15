#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <CL/sycl.hpp>
#include "dpc_common.hpp"

#include <iostream>
#include <stdexcept>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <opencv2/opencv.hpp>

using namespace cl::sycl;

extern "C";

// Simple image processing function (convolution filter)
static PyObject* image_convolution(PyObject* self, PyObject* args) {

    PyArrayObject *input_image, *conv_kernel;
    PyObject *output_image;
    if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &input_image, &PyArray_Type, &conv_kernel)) {            // Arguments are an Numpy array object
        return nullptr;
    }
    output_image = PyArray_NewLikeArray(input_image, NPY_ANYORDER, NULL, 0);    // Create object to return

    // Obtain data buffer pointers of Numpy objects
    uint8_t *in_buf  = static_cast<uint8_t*>(PyArray_DATA(input_image));   // PyArray_DATA() will return void*
    uint8_t *out_buf = static_cast<uint8_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(output_image)));
    float *kernel_buf = static_cast<float*>(PyArray_DATA(conv_kernel));
    int ndim = PyArray_NDIM(input_image);                           // Number of dimensions of the input Numpy array
    npy_intp *shape        = PyArray_SHAPE(input_image);            // Shape of the input Numpy array (npy_intp[])
    npy_intp *kernel_shape = PyArray_SHAPE(conv_kernel);            // Convolution kernel shape

    std::cout << shape[0] << "," << shape[1] << std::endl;
    std::cout << kernel_shape[0] << "," << kernel_shape[1] << std::endl;

    size_t num_rows = shape[0];             // image height
    size_t num_cols = shape[1];             // image width
    size_t kernel_height = kernel_shape[0];
    size_t kernel_width  = kernel_shape[1];
    int halfFilterWidth  = (int)(kernel_width/2); 
    int halfFilterHeight = (int)(kernel_height/2);

    // Create a SYCL queue  
    // default_selector | cpu_selector | host_selector | gpu_selector
    queue q(default_selector{});

    // Create SYCL buffer objects from existing buffers
    buffer<uint8_t, 1> image_in_buf(in_buf,   range<1>(num_rows * num_cols));
    buffer<uint8_t, 1> image_out_buf(out_buf, range<1>(num_rows * num_cols));
    range<2> num_items{num_rows, num_cols};
    buffer<float, 1> filter_buf(kernel_buf, range<1>(kernel_height * kernel_width));

    // Submit a job to SYCL
    q.submit([&](handler &h) {

        // Before the kernel code. The code here will be run by the host device (CPU)

        // Get access to the buffer objects (to be run on the host device)
        auto srcPtr = image_in_buf.get_access<access::mode::read>(h);
        auto dstPtr = image_out_buf.get_access<access::mode::write>(h);
        auto fltPtr = filter_buf.get_access<access::mode::read>(h);

        // Actual kernel code to run on the specified device
        h.parallel_for(num_items, [=](id<2> item) {
            int row = item[0];
            int col = item[1];
            float sum = 0.0f;
            // 'halfFilterHeight' is a variable in host device memory space but kernel code can refer to it.
            for (int fy = -halfFilterHeight; fy <= halfFilterHeight; fy++) {
                for (int fx = -halfFilterWidth; fx <= halfFilterWidth; fx++) {
                    int yy = row + fy;
                    int xx = col + fx;
                    yy = (yy < 0) ? 0 : yy;
                    xx = (xx < 0) ? 0 : xx;
                    yy = (yy >= num_rows) ? num_rows - 1 : yy;
                    xx = (xx >= num_cols) ? num_cols - 1 : xx;
                    float p = srcPtr[yy * num_cols + xx] * 
                              fltPtr[(fy + halfFilterHeight) * kernel_width + (fx + halfFilterWidth)];
                    sum += p;
                }
            }
            sum = (sum>255.0) ? 255.0 : (sum<0) ? 0 : sum;   // Check for uint8_t overflow
            dstPtr[row * num_cols + col] = (uint8_t)sum;
        });

        // After the kernel code. The code here will be run by the host device (CPU)

    });

    return output_image;
}



// Function definition table to export to Python
PyMethodDef method_table[] = {
    {"image_convolution", static_cast<PyCFunction>(image_convolution), METH_VARARGS, "test image processing function"},
    {NULL, NULL, 0, NULL}
};

// Module definition table
PyModuleDef test_module = {
    PyModuleDef_HEAD_INIT,
    "python_dpcpp_module",
    "DPC++ image convolution module",
    -1,
    method_table
};

// Initialize and register module function
// Function name must be 'PyInit_'+module name
// This function must be the only *non-static* function in the source code
PyMODINIT_FUNC PyInit_python_dpcpp_module(void) {
    import_array();                     // Required to receive Numpy object as arguments
    if (PyErr_Occurred()) {
        return nullptr;
    }
    return PyModule_Create(&test_module);
}
