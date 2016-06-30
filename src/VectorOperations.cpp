#include "VectorOperations.hpp"
#include <math.h>
#include <cassert>

//--------------------------------------------------------------------------------
void VectorOperations::set_zero( std::vector<double> &v ) 
{
  // computes v = 0;
  size = v.size();
  for ( unsigned int i = 0; i < size; ++i )
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
  for ( unsigned int i = 0; i < size; ++i )
    w[i] = s * v[i];
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void VectorOperations::add( double s, std::vector<double> const &v,
                            std::vector<double> &w)
{
  // computes w = w + s*v
  size = w.size();
  for ( unsigned int i = 0; i < size; i++)
    w[i] += s * v[i];
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
  for ( unsigned int i = 0; i < size; ++i )
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
  for ( unsigned int i = 0; i < size; ++i )
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
  for ( unsigned int i = 0; i < size; ++i )
    dbl += v1[i] * v2[i];
  return dbl;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double VectorOperations::norm( std::vector<double> const &v )
{
  // computes || v ||
  dbl = 0e0;
  size = v.size();
  for ( unsigned int i = 0; i < size; ++i)
    dbl += pow( v[i], 2e0 );
  return sqrt(dbl);
}
//--------------------------------------------------------------------------------
