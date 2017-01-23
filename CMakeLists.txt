# ==================================================== #
# Specification of include and lib directory of NLopt  #
# .................................................... #
# -DNOWPAC_NLOPT_INCLUDE_PATH=/path/to/nlopt/include   #
# -DNOWPAC_NLOPT_LIBRARY_PATH=/path/to/nlopt/library   #
# ---------------------------------------------------- #
# Specification of include directory of Eigen          #
# .................................................... #
# -DNOWPAC_EIGEN_INCLUDE_PATH=/path/to/eigen/include   #
# ---------------------------------------------------- #
# Compile shared library                               #
# .................................................... #
# -DNOWPAC_ENABLE_SHARED=ON                            #
# ---------------------------------------------------- #
# Set NOWPAC install directory                         #
# .................................................... #
# -DNOWPAC_INSTALL_PREFIX=/path/to/install/directory   #
# ---------------------------------------------------- #
# Enable and run tests                                 #
# .................................................... #
# -DNOWPAC_ENABLE_TESTS=ON                             #
# ---------------------------------------------------- #
# Specification of include and lib directory of Gtest  #
# .................................................... #
# -DNOWPAC_GTEST_INCLUDE_PATH=/path/to/gtest/include   #
# -DNOWPAC_GTEST_LIBRARY_PATH=/path/to/gtest/library   #
# ---------------------------------------------------- #
# Create documentation using Oxygen                    #
# .................................................... #
# -DNOWPAC_ENABLE_DOXYGEN=ON                           #
# ==================================================== #

# ==================================================== #
# Libraries will be automatically downloaded,          #
# if the above directories are not specified           #
# and the libraries are not found automatically.       #
# ==================================================== #


CMAKE_MINIMUM_REQUIRED (VERSION 2.8) 
 
PROJECT(NOWPAC)

IF ( NOT NOWPAC_INSTALL_PREFIX )
  SET( NOWPAC_INSTALL_PREFIX ${NOWPAC_BINARY_DIR} )
#  SET( NOWPAC_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} )
ENDIF ( )

LIST(APPEND CMAKE_MODULE_PATH ${NOWPAC_SOURCE_DIR}/cmake)

SET ( NOWPAC_BUILD_SHARED_LIBRARY "OFF" )
SET ( NOWPAC_SHARED_LIBRARY_EXTENSION "a")
IF ( NOWPAC_ENABLE_SHARED STREQUAL "ON" )
   SET ( NOWPAC_BUILD_SHARED_LIBRARY "ON")
   SET ( NOWPAC_SHARED_LIBRARY_EXTENSION "dylib")
   IF ( UNIX AND NOT APPLE )
      SET ( NOWPAC_SHARED_LIBRARY_EXTENSION "so")
   ENDIF ()
ENDIF ()

INCLUDE (confCompiler)

FIND_PACKAGE( EIGEN )
IF ( NOT EIGEN_FOUND )
   INCLUDE ( buildEIGEN )
ENDIF ()

FIND_PACKAGE( NLOPT )
IF ( NOT NLOPT_FOUND ) 
   INCLUDE ( buildNLOPT )
 ENDIF ()

IF ( NOWPAC_ENABLE_DOXYGEN STREQUAL "ON" )
   FIND_PACKAGE( DOXYGEN REQUIRED )
   IF ( DOXYGEN_FOUND ) 
      CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in 
                     ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
      ADD_CUSTOM_TARGET(doc ${DOXYGEN_EXECUTABLE} 
                            ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                       )
   ENDIF ()
ENDIF ()

IF ( NOWPAC_ENABLE_TESTS STREQUAL "ON" ) 
   FIND_PACKAGE( Threads REQUIRED )
   FIND_PACKAGE( GTEST )
   IF ( NOT GTEST_FOUND ) 
      INCLUDE ( buildGTEST )
    ENDIF ()
ENDIF ()

SET ( NOWPAC_SOURCE_FILES
    ${NOWPAC_SOURCE_DIR}/src/DeterministicTrainingConditional.cpp
    ${NOWPAC_SOURCE_DIR}/src/FullyIndependentTrainingConditional.cpp
    ${NOWPAC_SOURCE_DIR}/src/SubsetOfRegressors.cpp
    ${NOWPAC_SOURCE_DIR}/src/BlackBoxData.cpp
    ${NOWPAC_SOURCE_DIR}/src/CholeskyFactorization.cpp
    ${NOWPAC_SOURCE_DIR}/src/GaussianProcess.cpp
    ${NOWPAC_SOURCE_DIR}/src/GaussianProcessSupport.cpp
    ${NOWPAC_SOURCE_DIR}/src/ImprovePoisedness.cpp
    ${NOWPAC_SOURCE_DIR}/src/LegendreBasisForMinimumFrobeniusNormModel.cpp
    ${NOWPAC_SOURCE_DIR}/src/MonomialBasisForMinimumFrobeniusNormModel.cpp
    ${NOWPAC_SOURCE_DIR}/src/MinimumFrobeniusNormModel.cpp
    ${NOWPAC_SOURCE_DIR}/src/QuadraticMinimization.cpp
    ${NOWPAC_SOURCE_DIR}/src/QuadraticMonomial.cpp
    ${NOWPAC_SOURCE_DIR}/src/QuadraticLegendre.cpp
    ${NOWPAC_SOURCE_DIR}/src/TriangularMatrixOperations.cpp
    ${NOWPAC_SOURCE_DIR}/src/VectorOperations.cpp)
SET ( NOWPAC_HEADER_FILES
    ${NOWPAC_SOURCE_DIR}/include/DeterministicTrainingConditional.hpp
    ${NOWPAC_SOURCE_DIR}/include/FullyIndependentTrainingConditional.hpp
    ${NOWPAC_SOURCE_DIR}/include/SubsetOfRegressors.hpp
    ${NOWPAC_SOURCE_DIR}/include/BasisForSurrogateModelBaseClass.hpp
    ${NOWPAC_SOURCE_DIR}/include/BlackBoxBaseClass.hpp
    ${NOWPAC_SOURCE_DIR}/include/BlackBoxData.hpp
    ${NOWPAC_SOURCE_DIR}/include/CholeskyFactorization.hpp
    ${NOWPAC_SOURCE_DIR}/include/GaussianProcess.hpp
    ${NOWPAC_SOURCE_DIR}/include/GaussianProcessBaseClass.hpp
    ${NOWPAC_SOURCE_DIR}/include/GaussianProcessKernelBaseClass.hpp
    ${NOWPAC_SOURCE_DIR}/include/GaussianProcessSupport.hpp
    ${NOWPAC_SOURCE_DIR}/include/ImprovePoisedness.hpp
    ${NOWPAC_SOURCE_DIR}/include/ImprovePoisednessBaseClass.hpp
    ${NOWPAC_SOURCE_DIR}/include/LegendreBasisForMinimumFrobeniusNormModel.hpp
    ${NOWPAC_SOURCE_DIR}/include/MinimumFrobeniusNormModel.hpp
    ${NOWPAC_SOURCE_DIR}/include/MonomialBasisForMinimumFrobeniusNormModel.hpp
    ${NOWPAC_SOURCE_DIR}/include/NoiseDetection.hpp
    ${NOWPAC_SOURCE_DIR}/include/NOWPAC.hpp
    ${NOWPAC_SOURCE_DIR}/include/QuadraticMinimization.hpp
    ${NOWPAC_SOURCE_DIR}/include/QuadraticMonomial.hpp
    ${NOWPAC_SOURCE_DIR}/include/QuadraticLegendre.hpp
    ${NOWPAC_SOURCE_DIR}/include/SubproblemDefinitions.hpp
    ${NOWPAC_SOURCE_DIR}/include/SubproblemOptimization.hpp
    ${NOWPAC_SOURCE_DIR}/include/SurrogateModelBaseClass.hpp
    ${NOWPAC_SOURCE_DIR}/include/TransformNode.hpp
    ${NOWPAC_SOURCE_DIR}/include/TriangularMatrixOperations.hpp
    ${NOWPAC_SOURCE_DIR}/include/VectorOperations.hpp
    ${NOWPAC_SOURCE_DIR}/include/PostProcessModels.hpp
)
SET ( NOWPAC_GTEST_HEADER_FILES
    ${NOWPAC_SOURCE_DIR}/test/include/gtest_basisforminimumfrobeniusnormmodel.hpp
    ${NOWPAC_SOURCE_DIR}/test/include/gtest_blackboxdata.hpp
    ${NOWPAC_SOURCE_DIR}/test/include/gtest_choleskyfactorization.hpp
    ${NOWPAC_SOURCE_DIR}/test/include/gtest_improvepoisedness.hpp
    ${NOWPAC_SOURCE_DIR}/test/include/gtest_minimumfrobeniusnormmodel.hpp
    ${NOWPAC_SOURCE_DIR}/test/include/gtest_quadraticminimization.hpp
    ${NOWPAC_SOURCE_DIR}/test/include/gtest_quadraticmonimial.hpp
    ${NOWPAC_SOURCE_DIR}/test/include/gtest_triangularmatrixoperations.hpp
    ${NOWPAC_SOURCE_DIR}/test/include/gtest_vectoroperations.hpp
)
SET ( NOWPAC_GTEST_SOURCE_FILES
    ${NOWPAC_SOURCE_DIR}/test/src/gtest_main.cpp)

INCLUDE_DIRECTORIES( ${NOWPAC_SOURCE_DIR}/include )
INCLUDE_DIRECTORIES( ${NOWPAC_EIGEN_INCLUDE_PATH} )
INCLUDE_DIRECTORIES( ${NOWPAC_NLOPT_INCLUDE_PATH} )

ADD_LIBRARY(nowpacOlib OBJECT ${NOWPAC_SOURCE_FILES} ${NOWPAC_HEADER_FILES})
IF (NOT EIGEN_FOUND)
   ADD_DEPENDENCIES(nowpacOlib EIGEN)
ENDIF ()
IF (NOT NLOPT_FOUND)
   ADD_DEPENDENCIES(nowpacOlib NLOPT)
ENDIF ()


SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${NOWPAC_INSTALL_PREFIX}/nowpac/lib)
ADD_LIBRARY(nowpac STATIC $<TARGET_OBJECTS:nowpacOlib>)
ADD_DEPENDENCIES(nowpac nowpacOlib)

IF ( NOWPAC_ENABLE_SHARED STREQUAL "ON" ) 
   SET ( CMAKE_SKIP_RPATH "ON" )
   SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${NOWPAC_INSTALL_PREFIX}/nowpac/lib)
   ADD_LIBRARY(nowpacshared SHARED $<TARGET_OBJECTS:nowpacOlib>)
   TARGET_LINK_LIBRARIES(nowpacshared ${NOWPAC_NLOPT_LIBRARY_PATH}/libnlopt.${NOWPAC_SHARED_LIBRARY_EXTENSION})
   ADD_DEPENDENCIES(nowpacshared nowpacOlib)
ENDIF ()

IF ( NOWPAC_ENABLE_TESTS STREQUAL "ON" ) 
   ADD_EXECUTABLE(nowpac_test ${NOWPAC_GTEST_SOURCE_FILES} )
   SET_TARGET_PROPERTIES( nowpac_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NOWPAC_INSTALL_PREFIX}/nowpac/bin)
   TARGET_INCLUDE_DIRECTORIES(nowpac_test PUBLIC ${NOWPAC_SOURCE_DIR}/include)
   TARGET_INCLUDE_DIRECTORIES(nowpac_test PUBLIC ${NOWPAC_SOURCE_DIR}/test/include)
   TARGET_INCLUDE_DIRECTORIES(nowpac_test PUBLIC ${NOWPAC_EIGEN_INCLUDE_PATH})
   TARGET_INCLUDE_DIRECTORIES(nowpac_test PUBLIC ${NOWPAC_NLOPT_INCLUDE_PATH})
   TARGET_INCLUDE_DIRECTORIES(nowpac_test PUBLIC ${NOWPAC_GTEST_INCLUDE_PATH})
   TARGET_LINK_LIBRARIES(nowpac_test nowpac)
   TARGET_LINK_LIBRARIES(nowpac_test ${NOWPAC_GTEST_LIBRARY_PATH}/libgtest.a)
   TARGET_LINK_LIBRARIES(nowpac_test pthread)
   IF (NOT EIGEN_FOUND)
      ADD_DEPENDENCIES(nowpac_test EIGEN)
   ENDIF ()
   IF (NOT NLOPT_FOUND)
      ADD_DEPENDENCIES(nowpac_test NLOPT)
   ENDIF ()
   IF (NOT GTEST_FOUND)
      ADD_DEPENDENCIES(nowpac_test GTEST)
   ENDIF ()
   ADD_DEPENDENCIES(nowpac_test nowpac)
ENDIF ()

SET ( NOWPAC_INCLUDE_PATH "${NOWPAC_INSTALL_PREFIX}/nowpac/include" CACHE PATH "NOWPAC include directory" FORCE)
SET ( NOWPAC_LIBRARY_PATH "${NOWPAC_INSTALL_PREFIX}/nowpac/lib" CACHE PATH "NOWPAC library directory" FORCE)
SET ( NOWPAC_EIGEN_INCLUDE_PATH "${NOWPAC_EIGEN_INCLUDE_PATH}" CACHE PATH "Eigen include directory" FORCE)
SET ( NOWPAC_NLOPT_INCLUDE_PATH "${NOWPAC_NLOPT_INCLUDE_PATH}" CACHE PATH "Nlopt include directory" FORCE)
SET ( NOWPAC_NLOPT_LIBRARY_PATH "${NOWPAC_NLOPT_LIBRARY_PATH}" CACHE PATH "Nlopt library directory" FORCE)

INSTALL(FILES ${NOWPAC_HEADER_FILES} DESTINATION "${NOWPAC_INCLUDE_PATH}")

IF ( NOWPAC_ENABLE_TESTS STREQUAL "ON" ) 
   ADD_CUSTOM_COMMAND(TARGET nowpac_test POST_BUILD COMMAND ${NOWPAC_INSTALL_PREFIX}/nowpac/bin/nowpac_test VERBATIM)
ENDIF ()

message(STATUS "**************************************************")
message(STATUS "Final output will be written to:\n     ${NOWPAC_INSTALL_PREFIX}/nowpac/")
message(STATUS "**************************************************")


