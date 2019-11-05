#! /bin/bash

if [ ! -d "lib" ]; then
  mkdir lib
fi

if [ ! -d "build" ]; then
  mkdir build
fi
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make "$@"
