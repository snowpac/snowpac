#include "gtest_basisforminimumfrobeniusnormmodel.hpp"
#include "gtest_minimumfrobeniusnormmodel.hpp"
#include "gtest_improvepoisedness.hpp"
#include "gtest_choleskyfactorization.hpp"
#include "gtest_quadraticminimization.hpp"
#include "gtest_triangularmatrixoperations.hpp"
#include "gtest_vectoroperations.hpp"
#include "gtest_quadraticmonomial.hpp"
#include "gtest_blackboxdata.hpp"
#include "gtest/gtest.h"
#include <iostream>


int main ( int argc, char** argv ) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();

}
