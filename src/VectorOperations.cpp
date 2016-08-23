#include "VectorOperations.hpp"
#include <math.h>
#include <cassert>
#include <iostream>

//--------------------------------------------------------------------------------
void VectorOperations::set_zero( std::vector<double> &v ) 
{
  // computes v = 0;
  size = v.size();
  for ( int i = 0; i < size; ++i )
    v[i] = 0e0;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void VectorOperations::scale(double s, std::vector<double> const& v, 
                             std::vector<double> &w ) 
{
  // computes w = s*v;
  size = w.size();
  for ( int i = 0; i < size; ++i )
    w[i] = s * v[i];
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void VectorOperations::scale(double s, std::vector< std::vector<double> > const& V, 
                             std::vector< std::vector<double> > &W ) 
{
  // computes W = s*V;
  assert( W[0].size() == W.size() );
  size = W[0].size();
  for ( int i = 0; i < size; ++i ) {
    for ( int j = 0; j < size; ++j )
      W[i][j] = s * V[i][j];
  }
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void VectorOperations::add( double s, std::vector<double> const &v,
                            std::vector<double> &w)
{
  // computes w = w + s*v
  size = w.size();
  for ( int i = 0; i < size; i++)
    w[i] += s * v[i];
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void VectorOperations::add( std::vector<double> const &v,
                            std::vector<double> &w)
{
  // computes w = w + v
  size = w.size();
  for ( int i = 0; i < size; i++)
    w[i] += v[i];
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void VectorOperations::add( double s, std::vector< std::vector<double> > const &V,
                            std::vector< std::vector<double> > &W)
{
  // computes W = W + s*V
  assert( W[0].size() == W.size() );
  size = W[0].size();
  for ( int i = 0; i < size; i++) {
    for ( int j = 0; j < size; j++)
      W[i][j] += s * V[i][j];
  }
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void VectorOperations::minus(std::vector<double> const& v1, 
                             std::vector<double> const& v2,
                             std::vector<double> &w ) 
{
  // computes w = v1 - v2;
  for ( unsigned int i = 0; i < w.size(); ++i )
    w[i] = ( v1[i] - v2[i] );
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void VectorOperations::rescale(double s, std::vector<double> const& v1, 
                               std::vector<double> const& v2,
                               std::vector<double> &w ) 
{
  // computes w = (v1 - v2) * s;
  size = w.size();
  assert ( w.size() == v1.size() ); 
  for ( int i = 0; i < size; ++i )
    w[i] = ( v1[i] - v2[i] ) * s;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double VectorOperations::diff_norm( std::vector<double> const &v1, 
                                    std::vector<double> const &v2) 
{
  // computes ||v1-v2||
  dbl = 0e0;
  size = v1.size();
  for ( int i = 0; i < size; ++i )
    dbl += pow( v1[i] - v2[i], 2e0 );
  return sqrt(dbl);
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double VectorOperations::dot_product( std::vector<double> const &v1, 
                                      std::vector<double> const &v2) 
{
  // computes v1.dot(v2)
  dbl = 0e0;
  size = v1.size();
  for ( int i = 0; i < size; ++i )
    dbl += v1[i] * v2[i];
  return dbl;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void VectorOperations::mat_transpose(std::vector<std::vector<double> > const & V, std::vector<std::vector<double> > & V_T) {
  for(int i = 0; i < V.size(); ++i){
    for(int j = 0; j < V_T.size(); ++j){
      V_T[j][i] = V[i][j];
    }
  }
  return ;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void VectorOperations::mat_product( std::vector< std::vector<double> > const &V1, 
                                    std::vector< std::vector<double> > const &V2,
                                    std::vector< std::vector<double> > &W ) 
{
  // computes W = V1 * V2
  size  = V1[0].size();
  size1 = V1.size();
  size2 = V2.size();
  for ( int i = 0; i < V1.size(); ++i ) {
    for ( int j = 0; j < V2.at(0).size(); ++j ) {
      W.at(i).at(j) = 0e0;
      for ( int k = 0; k < V1.at(0).size(); ++k )
        W.at(i).at(j) += V1.at(i).at(k) * V2.at(k).at(j);
    }
  }
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void VectorOperations::mat_vec_product( std::vector< std::vector<double> > const &V1, 
                                        std::vector<double> const &v2,
                                        std::vector<double> &w ) 
{
  // computes w = V1 * v2
  size1 = V1.size();
  size2 = v2.size();
  for ( int i = 0; i < V1.size(); ++i ) {
    w.at(i) = 0e0;
    for ( int j = 0; j < V1.at(i).size(); ++j )
        w.at(i) += V1.at(i).at(j) * v2.at(j);
  }
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void VectorOperations::mat_square( std::vector< std::vector<double> > const &V, 
                                   std::vector< std::vector<double> > &W ) 
{
  // computes W = V' * V
  size  = V[0].size();
  size1 = V.size();
  for ( int i = 0; i < size1; ++i ) {
    for ( int j = 0; j < size1; ++j ) {
      W[i][j] = 0e0;
      for ( int k = 0; k < size; ++k )
        W[i][j] += V[k][i] * V[k][j];
    }
  }
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double VectorOperations::norm( std::vector<double> const &v )
{
  // computes || v ||
  dbl = 0e0;
  size = v.size();
  for ( int i = 0; i < size; ++i)
    dbl += pow( v[i], 2e0 );
  return sqrt(dbl);
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double VectorOperations::norm( std::vector< std::vector<double> > const &V )
{
  // computes || V ||_F
  assert(V[0].size() == V.size());
  dbl = 0e0;
  size = V.size();
  for ( int i = 0; i < size; ++i ) {
    for ( int j = 0; j < size; ++j ) 
      dbl += pow( V[i][j], 2e0 );
  }
  return sqrt(dbl);
}

void VectorOperations::vec_mat_product(std::vector<std::vector<double> > const &V1, std::vector<double> const &v2,
                                       std::vector<double> &w)
{
  // computes w = v2^T * V1
  size1 = V1.size();
  size2 = v2.size();
  for ( int j = 0; j < w.size(); ++j ) {
    w.at(j) = 0e0;
    for ( int i = 0; i < size1; ++i )
      w.at(j) += v2.at(i) * V1.at(i).at(j);
  }
  return;

}

void VectorOperations::print_matrix(std::vector<std::vector<double> > const & M) {
    std::cout << "[";
    for(int i = 0; i < M.size(); ++i){
        for(int j = 0; j < M.at(i).size(); ++j){
            std::cout << M.at(i).at(j) << " ";
        }
        std::cout << ";" << std::endl;
    }
    std::cout << "]" << std::endl;
}

void VectorOperations::print_vector(std::vector<double> const & v) {
    std::cout << "[";
    for(int i = 0; i < v.size(); ++i){
        std::cout << v.at(i) << " ";
    }
    std::cout << "]" << std::endl;
}
//--------------------------------------------------------------------------------
