#!/bin/bash

clear
cd ..
[[ -d build ]] || mkdir build
cd build

cmake -DNOWPAC_INSTALL_PREFIX=/usr/local \
      -DCMAKE_CXX_COMPILER=/usr/local/Cellar/gcc/5.3.0/bin/g++-5 \
      -DCMAKE_C_COMPILER=/usr/local/Cellar/gcc/5.3.0/bin/gcc-5 \
      -DNOWPAC_ENABLE_SHARED=ON \
      -DNOWPAC_ENABLE_TESTS=ON \
      ../snowpac/

make install -j8
