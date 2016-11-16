#!/bin/bash

clear
cd ..
[[ -d build ]] || mkdir build
cd build

#cmake -DNOWPAC_INSTALL_PREFIX=/usr/local
cmake -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_BUILD_TYPE=Debug \
      -DNOWPAC_ENABLE_SHARED=ON \
      -DNOWPAC_ENABLE_TESTS=ON \
      -DNOWPAC_ENABLE_DOXYGEN=OFF \
      ../snowpac/

make doc
make install -j8
