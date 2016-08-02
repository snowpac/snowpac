# ==================================================== #
# Specification of include directory of NLopt          #
# .................................................... #
# -DNOWPAC_NLOPT_INCLUDE_PATH=/path/to/nlopt/include   #
# -DNOWPAC_NLOPT_LIBRARY_PATH=/path/to/nlopt/library   #
# ---------------------------------------------------- #
# Specification of include directory of Eigen          #
# .................................................... #
# -DNOWPAC_EIGEN_INCLUDE_PATH=/path/to/eigen/include   #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Libraries will be automatically downloaded,          #
# if the above directories are not specified           #
# and the libraries are not found automatically.       #
# ==================================================== #

CMAKE_MINIMUM_REQUIRED (VERSION 2.8) 
 
PROJECT(NOWPAC)
 
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

SET ( NOWPAC_SOURCE_FILES
    ${NOWPAC_SOURCE_DIR}/src/BasisForMinimumFrobeniusNormModel.cpp
    ${NOWPAC_SOURCE_DIR}/src/BlackBoxData.cpp
    ${NOWPAC_SOURCE_DIR}/src/CholeskyFactorization.cpp
    ${NOWPAC_SOURCE_DIR}/src/GaussianProcess.cpp
    ${NOWPAC_SOURCE_DIR}/src/GaussianProcessSupport.cpp
    ${NOWPAC_SOURCE_DIR}/src/ImprovePoisedness.cpp
    ${NOWPAC_SOURCE_DIR}/src/MinimumFrobeniusNormModel.cpp
    ${NOWPAC_SOURCE_DIR}/src/QuadraticMinimization.cpp
    ${NOWPAC_SOURCE_DIR}/src/QuadraticMonomial.cpp
    ${NOWPAC_SOURCE_DIR}/src/TriangularMatrixOperations.cpp
    ${NOWPAC_SOURCE_DIR}/src/VectorOperations.cpp
)
SET ( NOWPAC_HEADER_FILES
    ${NOWPAC_SOURCE_DIR}/include/BasisForMinimumFrobeniusNormModel.hpp
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
    ${NOWPAC_SOURCE_DIR}/include/MinimumFrobeniusNormModel.hpp
    ${NOWPAC_SOURCE_DIR}/include/NoiseDetection.hpp
    ${NOWPAC_SOURCE_DIR}/include/NOWPAC.hpp
    ${NOWPAC_SOURCE_DIR}/include/QuadraticMinimization.hpp
    ${NOWPAC_SOURCE_DIR}/include/QuadraticMonomial.hpp
    ${NOWPAC_SOURCE_DIR}/include/SubproblemDefinitions.hpp
    ${NOWPAC_SOURCE_DIR}/include/SubproblemOptimization.hpp
    ${NOWPAC_SOURCE_DIR}/include/SurrogateModelBaseClass.hpp
    ${NOWPAC_SOURCE_DIR}/include/TransformNode.hpp
    ${NOWPAC_SOURCE_DIR}/include/TriangularMatrixOperations.hpp
    ${NOWPAC_SOURCE_DIR}/include/VectorOperations.hpp
)

INCLUDE_DIRECTORIES(${NOWPAC_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES( ${NOWPAC_EIGEN_INCLUDE_PATH} )
INCLUDE_DIRECTORIES( ${NOWPAC_NLOPT_INCLUDE_PATH} )

ADD_LIBRARY(nowpacOlib OBJECT ${NOWPAC_SOURCE_FILES} ${NOWPAC_HEADER_FILES})
IF (NOT EIGEN_FOUND)
   ADD_DEPENDENCIES(nowpacOlib EIGEN)
ENDIF ()
IF (NOT NLOPT_FOUND)
   ADD_DEPENDENCIES(nowpacOlib NLOPT)
ENDIF ()

SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/nowpac/lib)
ADD_LIBRARY(nowpac STATIC $<TARGET_OBJECTS:nowpacOlib>)
ADD_DEPENDENCIES(nowpac nowpacOlib)

IF ( NOWPAC_ENABLE_SHARED STREQUAL "ON" ) 
    SET ( CMAKE_SKIP_RPATH "ON" )
    SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/nowpac/lib)
    ADD_LIBRARY(nowpacshared SHARED $<TARGET_OBJECTS:nowpacOlib>)
    
    TARGET_LINK_LIBRARIES(nowpacshared ${NOWPAC_NLOPT_LIBRARY_PATH}/libnlopt.${NOWPAC_SHARED_LIBRARY_EXTENSION})
    ADD_DEPENDENCIES(nowpacshared nowpacOlib)
ENDIF ()

SET ( NOWPAC_INCLUDE_PATH "${CMAKE_INSTALL_PREFIX}/nowpac/include" CACHE PATH "NOWPAC include directory" FORCE)
SET ( NOWPAC_LIBRARY_PATH "${CMAKE_INSTALL_PREFIX}/nowpac/lib" CACHE PATH "NOWPAC library directory" FORCE)
SET ( NOWPAC_EIGEN_INCLUDE_PATH "${NOWPAC_EIGEN_INCLUDE_PATH}" CACHE PATH "Eigen include directory" FORCE)
SET ( NOWPAC_NLOPT_INCLUDE_PATH "${NOWPAC_NLOPT_INCLUDE_PATH}" CACHE PATH "Nlopt include directory" FORCE)
SET ( NOWPAC_NLOPT_LIBRARY_PATH "${NOWPAC_NLOPT_LIBRARY_PATH}" CACHE PATH "Nlopt library directory" FORCE)

INSTALL(FILES ${NOWPAC_HEADER_FILES} DESTINATION "${NOWPAC_INCLUDE_PATH}")

message(STATUS "**************************************************")
message(STATUS "Final output will be written to:\n     ${CMAKE_INSTALL_PREFIX}/nowpac/")
message(STATUS "**************************************************")


