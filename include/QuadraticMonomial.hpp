#ifndef HQuadraticMonomial
#define HQuadraticMonomial

#include <vector>
#include <math.h>
//! Quadratic monomials
/*! 
 Evaluates quadratic monomials in dim dimensions.
 Monimial 0 = 1
 Monomial 1 ... dim = x_i
 Monomial dim+1 ... 2*dim = 0.5*x_i^2
 Monomial 2*dim+1 ... 3*dim = x_1*x_2 ... x_1*x_n
 Monomial 3*dim+1 ... 4*dim-1 = x_2*x_3 ... x_2*x_n
 ...
 Monoial (dim^2 + 3*dim +2)/2 = x_{n-1}x_n
*/
class QuadraticMonomial {
  private:
    int dim;
  protected:
    //! Evaluation of monomials
    /*!
     Evaluation of monomial number p
     \param p Number of monomial as discribed
     \param x Point where monomial is evaluated
     \see QuadraticMonomial
    */
    double evaluate_basis ( int, std::vector<double> const& );
  public:
    //! Constructor
    /*!
     \param dim_input Dimension of monomials
    */
    QuadraticMonomial ( int dim_input ) : dim ( dim_input ) { }
};

#endif
