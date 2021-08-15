#!/usr/bin/env bash
mkdir -p build
pushd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cp libpython_dpcpp_module.so ../python_dpcpp_module.so
popd
