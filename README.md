# Sample code of Python extension module using DPC++ (Data Parallel C++)  

### Description:  
This project demonstrates how to write a Python extension module with [DPC++](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/dpc-compiler.html#gs.8eyara). The Python extension will take an OpenCV image stored in a Numpy array and do image processing. The actual image processing code is using DPC++ kernel and the rest is written in standard C++ code.  
DPC++ is a Clang based compiler developed by Intel which includs Khronos [SYCL](https://www.khronos.org/sycl/) extension and Intel specific SYCL extensions. **It enables seamless heterogeneous programming and can support not only CPU but also integrated GPU, FPGA and more to come**.  

* Result example
![image](./resources/result.png)

### Prerequisites:
- DPC++ compiler (from oneAPI Base Toolkit 2021.3)

### How to build and run:
```sh
build.sh
```
 * `python_dpcpp_module.so` will be generated.
```sh
python3 test.py
```

### Note:  
Tested on Ubuntu 20.04.
