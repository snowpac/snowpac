# Install script for directory: /Users/Florian/home/developments/git/snowpac

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/nowpac/include/BasisForMinimumFrobeniusNormModel.hpp;/usr/local/nowpac/include/BasisForSurrogateModelBaseClass.hpp;/usr/local/nowpac/include/BlackBoxBaseClass.hpp;/usr/local/nowpac/include/BlackBoxData.hpp;/usr/local/nowpac/include/CholeskyFactorization.hpp;/usr/local/nowpac/include/GaussianProcess.hpp;/usr/local/nowpac/include/GaussianProcessBaseClass.hpp;/usr/local/nowpac/include/GaussianProcessKernelBaseClass.hpp;/usr/local/nowpac/include/GaussianProcessSupport.hpp;/usr/local/nowpac/include/ImprovePoisedness.hpp;/usr/local/nowpac/include/ImprovePoisednessBaseClass.hpp;/usr/local/nowpac/include/MinimumFrobeniusNormModel.hpp;/usr/local/nowpac/include/NoiseDetection.hpp;/usr/local/nowpac/include/NOWPAC.hpp;/usr/local/nowpac/include/QuadraticMinimization.hpp;/usr/local/nowpac/include/QuadraticMonomial.hpp;/usr/local/nowpac/include/SubproblemDefinitions.hpp;/usr/local/nowpac/include/SubproblemOptimization.hpp;/usr/local/nowpac/include/SurrogateModelBaseClass.hpp;/usr/local/nowpac/include/TransformNode.hpp;/usr/local/nowpac/include/TriangularMatrixOperations.hpp;/usr/local/nowpac/include/VectorOperations.hpp")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/nowpac/include" TYPE FILE FILES
    "/Users/Florian/home/developments/git/snowpac/include/BasisForMinimumFrobeniusNormModel.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/BasisForSurrogateModelBaseClass.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/BlackBoxBaseClass.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/BlackBoxData.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/CholeskyFactorization.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/GaussianProcess.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/GaussianProcessBaseClass.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/GaussianProcessKernelBaseClass.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/GaussianProcessSupport.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/ImprovePoisedness.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/ImprovePoisednessBaseClass.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/MinimumFrobeniusNormModel.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/NoiseDetection.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/NOWPAC.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/QuadraticMinimization.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/QuadraticMonomial.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/SubproblemDefinitions.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/SubproblemOptimization.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/SurrogateModelBaseClass.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/TransformNode.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/TriangularMatrixOperations.hpp"
    "/Users/Florian/home/developments/git/snowpac/include/VectorOperations.hpp"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/Users/Florian/home/developments/git/snowpac/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
