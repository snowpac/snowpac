INCLUDE(ExternalProject)

#SET( NOWPAC_EIGEN_URL "http://bitbucket.org/eigen/eigen/get/3.2.2.tar.bz2")
SET( NOWPAC_EIGEN_URL "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2")

ExternalProject_Add( EIGEN

   PREFIX ${NOWPAC_INSTALL_PREFIX}/nowpac/external/eigen/build
   URL    ${NOWPAC_EIGEN_URL}
   CMAKE_ARGS -DCMAKE_MACOSX_RPATH=ON              
              -DCMAKE_INSTALL_PREFIX=${NOWPAC_INSTALL_PREFIX}/nowpac/external/eigen                 
              -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}              
              -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}               
              -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} 
              -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}        
              -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}        
              -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
   BUILD_COMMAND make
   INSTALL_COMMAND make install
   LOG_DOWNLOAD 1
   LOG_CONFIGURE 1
   LOG_BUILD 1
   LOG_INSTALL 1
   BUILD_IN_SOURCE 0

)

#SET_PROPERTY( TARGET EIGEN PROPERTY FOLDER "nowpac/external")
SET(NOWPAC_EIGEN_INCLUDE_PATH "${NOWPAC_INSTALL_PREFIX}/nowpac/external/eigen/include/eigen3")


