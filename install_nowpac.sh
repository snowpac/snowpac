#!/bin/bash

#clear
cd ..
[[ -d build ]] || mkdir build
cd build

cmake -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_C_COMPILER=gcc \
      -DNOWPAC_ENABLE_SHARED=ON \
      -DNOWPAC_ENABLE_TESTS=OFF \
      -DNOWPAC_ENABLE_DOXYGEN=OFF \
      ../snowpac

make doc
make install -j4
