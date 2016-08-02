#include "gtest_basisforminimumfrobeniusnormmodel.cpp"
#include "gtest_minimumfrobeniusnormmodel.cpp"
#include "gtest_improvepoisedness.cpp"
#include "gtest_choleskyfactorization.cpp"
#include "gtest_quadraticminimization.cpp"
#include "gtest_triangularmatrixoperations.cpp"
#include "gtest_vectoroperations.cpp"
#include "gtest_quadraticmonomial.cpp"
#include "gtest_blackboxdata.cpp"
#include "gtest/gtest.h"
#include <iostream>



int main ( int argc, char** argv ) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();

}
