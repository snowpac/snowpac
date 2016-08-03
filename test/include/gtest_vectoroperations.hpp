#include "VectorOperations.hpp"
#include "gtest/gtest.h"
/*
//--------------------------------------------------------------------------------
class Wrapper_VectorOperations : public VectorOperations {
  public:
    int dot_product_test ( std::vector<double>) {
      return 1;
    }
} ;
//--------------------------------------------------------------------------------
*/

//--------------------------------------------------------------------------------
TEST ( VectorOperationsTest, set_zero_test ) 
{
  VectorOperations vo;
  int dim = 5;
  std::vector<double> v(dim);
  vo.set_zero(v);
  for ( int i = 0; i < dim; ++i )
    EXPECT_NEAR( v[i], 0.0, 1e-6 );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( VectorOperationsTest, scale_vector_test ) 
{
  VectorOperations vo;
  int dim = 5;
  std::vector<double> v(dim);
  std::vector<double> w(dim);
  v[0] = 1.0; v[1] = 2.0; v[2] = 4.0;  v[3] = 5.0; v[4] = 10.0;
  double scale = 0.5;
  vo.scale(scale, v, w);
  for ( int i = 0; i < dim; ++i )
    EXPECT_NEAR( w[i], scale*v[i], 1e-6 );
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( VectorOperationsTest, scale_matrix_test ) 
{
  VectorOperations vo;
  int dim = 3;
  std::vector< std::vector<double> > V(dim);
  std::vector< std::vector<double> > W(dim);
  for ( int i = 0; i < dim; ++i ) {
    V[i].resize(dim);
    W[i].resize(dim);
  }
  for ( int i = 0; i < dim; ++i ) {
    for ( int j = 0; j < dim; ++j ) {
      V[i][j] = ((double) i)*((double) dim) + ((double) j);
    }
  }
  double scale = 10.0;
  vo.scale(scale, V, W);
  double cmp_dbl;
  for ( int i = 0; i < dim; ++i ) {
    for ( int j = 0; j < dim; ++j ) {
      cmp_dbl = ((double) i)*((double) dim) + ((double) j);
      EXPECT_NEAR( W[i][j], scale*cmp_dbl, 1e-6 );
    }
  }
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( VectorOperationsTest, scale_add_vector_test ) 
{
  VectorOperations vo;
  int dim = 5;
  std::vector<double> v(dim);
  std::vector<double> w(dim);
  std::vector<double> r(dim);
  v[0] = 1.0; v[1] = 2.0; v[2] = 4.0;  v[3] = 5.0; v[4] = 10.0;
  w[0] = 1.0; w[1] = 0.5; w[2] = 0.25; w[3] = 0.2; w[4] = 0.1;
  double scale = 3.0;
  r[0] = 4.0; r[1] = 6.5; r[2] = 12.25; r[3] = 15.2; r[4] = 30.1;
  vo.add(scale, v, w);
  for ( int i = 0; i < dim; ++i )
    EXPECT_NEAR( w[i], r[i], 1e-6 );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( VectorOperationsTest, add_vector_test ) 
{
  VectorOperations vo;
  int dim = 5;
  std::vector<double> v(dim);
  std::vector<double> w(dim);
  std::vector<double> r(dim);
  v[0] = 1.0; v[1] = 2.0; v[2] = 4.0;  v[3] = 5.0; v[4] = 10.0;
  w[0] = 1.0; w[1] = 0.5; w[2] = 0.25; w[3] = 0.2; w[4] = 0.1;
  r[0] = 2.0; r[1] = 2.5; r[2] = 4.25; r[3] = 5.2; r[4] = 10.1;
  vo.add(v, w);
  for ( int i = 0; i < dim; ++i )
    EXPECT_NEAR( w[i], r[i], 1e-6 );
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( VectorOperationsTest, scale_add_matrix_test ) 
{
  VectorOperations vo;
  int dim = 3;
  double scale = 10.0;
  std::vector< std::vector<double> > V(dim);
  std::vector< std::vector<double> > W(dim);
  std::vector< std::vector<double> > R(dim);
  for ( int i = 0; i < dim; ++i ) {
    V[i].resize(dim);
    W[i].resize(dim);
    R[i].resize(dim);
  }
  for ( int i = 0; i < dim; ++i ) {
    for ( int j = 0; j < dim; ++j ) {
      V[i][j] = ((double) i)*((double) dim) + ((double) j);
      W[i][j] = ((double) i)*((double) dim) + ((double) j);
      R[i][j] = (scale+1.0)*(((double) i)*((double) dim) + ((double) j));
    }
  }

  vo.add(scale, V, W);
  double cmp_dbl;
  for ( int i = 0; i < dim; ++i ) {
    for ( int j = 0; j < dim; ++j ) {
      EXPECT_NEAR( W[i][j], R[i][j], 1e-6 );
    }
  }
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( VectorOperationsTest, minus_vector_test ) 
{
  VectorOperations vo;
  int dim = 5;
  std::vector<double> v1(dim);
  std::vector<double> v2(dim);
  std::vector<double> w(dim);
  std::vector<double> r(dim);
  v1[0] = 1.0; v1[1] = 2.0; v1[2] = 4.0;  v1[3] = 5.0; v1[4] = 10.0;
  v2[0] = 1.0; v2[1] = 0.5; v2[2] = 0.25; v2[3] = 0.2; v2[4] = 0.1;
  r[0] = 0.0; r[1] = 1.5; r[2] = 3.75; r[3] = 4.8; r[4] = 9.9;
  vo.minus(v1, v2, w);
  for ( int i = 0; i < dim; ++i )
    EXPECT_NEAR( w[i], r[i], 1e-6 );
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( VectorOperationsTest, rescale_vector_test ) 
{
  VectorOperations vo;
  int dim = 5;
  double scale = 2.0;
  std::vector<double> v1(dim);
  std::vector<double> v2(dim);
  std::vector<double> w(dim);
  std::vector<double> r(dim);
  v1[0] = 1.0; v1[1] = 2.0; v1[2] = 4.0;  v1[3] = 5.0; v1[4] = 10.0;
  v2[0] = 1.0; v2[1] = 0.5; v2[2] = 0.25; v2[3] = 0.2; v2[4] = 0.1;
  r[0] = 0.0; r[1] = 1.5; r[2] = 3.75; r[3] = 4.8; r[4] = 9.9;
  vo.rescale(scale, v1, v2, w);
  for ( int i = 0; i < dim; ++i )
    EXPECT_NEAR( w[i], r[i]*scale, 1e-6 );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( VectorOperationsTest, diff_norm_vector_test ) 
{
  VectorOperations vo;
  int dim = 5;
  std::vector<double> v1(dim);
  std::vector<double> v2(dim);
  std::vector<double> r(dim);
  v1[0] = 1.0; v1[1] = 2.0; v1[2] = 4.0;  v1[3] = 5.0; v1[4] = 10.0;
  v2[0] = 1.0; v2[1] = 0.5; v2[2] = 0.25; v2[3] = 0.2; v2[4] = 0.1;
  r[0] = 0.0; r[1] = 1.5; r[2] = 3.75; r[3] = 4.8; r[4] = 9.9;
  double diffnorm = 0.0;
  for ( int i = 0; i < dim; ++i )
    diffnorm += r[i]*r[i];
  diffnorm = sqrt(diffnorm);
  for ( int i = 0; i < dim; ++i )
    EXPECT_NEAR( vo.diff_norm(v1, v2), diffnorm, 1e-6 );
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( VectorOperationsTest, dot_product_test ) 
{
  VectorOperations vo;
  int dim = 5;
  std::vector<double> v(dim);
  std::vector<double> w(dim);
  v[0] = 1.0; v[1] = 2.0; v[2] = 4.0;  v[3] = 5.0; v[4] = 10.0;
  w[0] = 1.0; w[1] = 0.5; w[2] = 0.25; w[3] = 0.2; w[4] = 0.1;
  EXPECT_NEAR( vo.dot_product(v, w), 5.0, 1e-6 );
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( VectorOperationsTest, mat_product_test ) 
{
  VectorOperations vo;
  int dim = 100;
  std::vector< std::vector<double> > V1(dim);
  std::vector< std::vector<double> > V2(dim);
  std::vector< std::vector<double> > W(dim);
  std::vector< std::vector<double> > R(dim);
  for ( int i = 0; i < dim; ++i ) {
    V1[i].resize(dim);
    V2[i].resize(dim);
    W[i].resize(dim);
    R[i].resize(dim);
  }
  for ( int i = 0; i < dim; ++i ) {
    for ( int j = 0; j < dim; ++j ) {
      V1[i][j] = sin(((double) i)*((double) dim) + ((double) j));
      V2[i][j] = cos(((double) i)*((double) dim) + ((double) j));
    }
  }
  for ( int i = 0; i < dim; ++i ) {
    for ( int j = 0; j < dim; ++j ) {
      R[i][j] = 0.0;
      for ( int k = 0; k < dim; ++k ) {
        R[i][j] += V1[i][k]*V2[k][j];
      }
    }
  }

/*
  std::cout << std::endl;
  for ( unsigned int i = 0; i < dim; ++i ) {
    for ( unsigned int j = 0; j < dim; ++j )
      std::cout << R[i][j] << ", ";
    std::cout << std::endl;
  }
*/

  vo.mat_product(V1, V2, W);
  double cmp_dbl;
  for ( int i = 0; i < dim; ++i ) {
    for ( int j = 0; j < dim; ++j ) {
      EXPECT_NEAR( W[i][j], R[i][j], 1e-6 );
    }
  }
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( VectorOperationsTest, mat_vec_product_test ) 
{
  VectorOperations vo;
  int dim = 97;
  std::vector< std::vector<double> > V1(dim);
  std::vector<double> v2(dim);
  std::vector<double> w(dim);
  std::vector<double> r(dim);
  for ( int i = 0; i < dim; ++i ) 
    V1[i].resize(dim);

  for ( int i = 0; i < dim; ++i ) {
    for ( int j = 0; j < dim; ++j )
      V1[i][j] = sin(((double) i)*((double) dim) + ((double) j));
    v2[i] = cos(((double) i)*((double) dim));
  }
  for ( int i = 0; i < dim; ++i ) {
    r[i] = 0.0;
    for ( int j = 0; j < dim; ++j )
      r[i] += V1[i][j]*v2[j];
  }

/*
  std::cout << std::endl;
  for ( unsigned int i = 0; i < dim; ++i )
      std::cout << r[i] << ", ";
  std::cout << std::endl;
*/

  vo.mat_vec_product(V1, v2, w);
  double cmp_dbl;
  for ( int i = 0; i < dim; ++i )
      EXPECT_NEAR( w[i], r[i], 1e-6 );

}
//--------------------------------------------------------------------------------



//--------------------------------------------------------------------------------
TEST ( VectorOperationsTest, mat_square_test ) 
{
  VectorOperations vo;
  int dim = 54;
  std::vector< std::vector<double> > V(dim);
  std::vector< std::vector<double> > W(dim);
  std::vector< std::vector<double> > R(dim);
  for ( int i = 0; i < dim; ++i ) {
    V[i].resize(dim);
    W[i].resize(dim);
    R[i].resize(dim);
  }
  for ( int i = 0; i < dim; ++i ) {
    for ( int j = 0; j < dim; ++j )
      V[i][j] = sin(((double) i)*((double) dim) + ((double) j));
  }
  for ( int i = 0; i < dim; ++i ) {
    for ( int j = 0; j < dim; ++j ) {
      R[i][j] = 0.0;
      for ( int k = 0; k < dim; ++k )
        R[i][j] += V[k][i]*V[k][j];
    }
  }

/*
  std::cout << std::endl;
  for ( unsigned int i = 0; i < dim; ++i ) {
    for ( unsigned int j = 0; j < dim; ++j )
      std::cout << R[i][j] << ", ";
    std::cout << std::endl;
  }
*/

  vo.mat_square(V, W);
  double cmp_dbl;
  for ( int i = 0; i < dim; ++i ) {
    for ( int j = 0; j < dim; ++j ) {
      EXPECT_NEAR( W[i][j], R[i][j], 1e-6 );
    }
  }
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( VectorOperationsTest, norm_test ) 
{
  VectorOperations vo;
  int dim = 5;
  std::vector<double> v(dim);
  v[0] = 1.0; v[1] = 2.0; v[2] = 4.0;  v[3] = 5.0; v[4] = 10.0;
  EXPECT_NEAR( vo.norm(v), sqrt(146), 1e-6 );
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( VectorOperationsTest, mat_norm_test ) 
{
  VectorOperations vo;
  int dim = 75;
  std::vector< std::vector<double> > V(dim);
  for ( int i = 0; i < dim; ++i )
    V[i].resize(dim);
  double Vnorm = 0.0;
  for ( int i = 0; i < dim; ++i ) {
    for ( int j = 0; j < dim; ++j ) {
      V[i][j] = sin(((double) i)*((double) dim) + ((double) j));
      Vnorm += V[i][j]*V[i][j];
    }
  }
  Vnorm = sqrt(Vnorm);

  //std::cout << "||V||_F = " << Vnorm;
  //std::cout << "||V||_F = " << vo.norm(V);

  EXPECT_NEAR( vo.norm(V), Vnorm, 1e-6 );

}
//--------------------------------------------------------------------------------
