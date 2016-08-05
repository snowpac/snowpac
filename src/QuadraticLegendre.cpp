#include "QuadraticLegendre.hpp"
#include <cassert>
// coefficients in quadratic model:
// m = (c0 - 0.5*\sum_{i=1}^n c_{n+i})  
//     + \sum_{i=1}^n x_i * c_i 
//     + 0.5*x'*H*x
// with diag(H) = (c_{n+1}, ..., c_{2n})/3.0
// H_{1,2:n} = (c_{2n+1}, ..., c_{3n-1})
// H_{2,3:n} = (c_{3n}, ..., c_{4n-2})
// H_{3,4:n} = (c_{4n-1}, ..., c_{5n-3})
// ...

//--------------------------------------------------------------------------------
double QuadraticLegendre::evaluate_basis( 
  int basis_number, std::vector<double> const &x ) 
{
  if ( basis_number == 0 )
    return 1e0;
  else if ( basis_number >= 1 && basis_number <= dim )
    return x.at( basis_number-1 );
  else if ( basis_number >= dim+1 && basis_number <= 2*dim )
    return (3e0 * pow( x.at( basis_number-dim-1 ), 2e0 ) - 1e0 ) / 2e0;
  else {
    basis_number -= 2*dim;
    for ( int j = 1; j <= dim-1; ++j ) {
      if ( basis_number <= dim-j ) 
        return x.at( j-1 ) * x.at( basis_number-1+j );
      basis_number -= dim-j;
    }
  }
  assert ( false );
  return 0.0;
}
//--------------------------------------------------------------------------------
