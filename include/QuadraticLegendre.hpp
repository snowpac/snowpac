#ifndef HQuadraticLegendre
#define HQuadraticLegendre

#include <vector>
#include <math.h>
//! Quadratic Legendre polynomials
/*! 
 Evaluates quadratic Legendre polynomials in dim dimensions.
 Polynomial 0 = 1
 Polynomial 1 ... dim = x_i
 Polynomial dim+1 ... 2*dim = (3*x_i^2 - 1e0)/2e0
 Polynomial 2*dim+1 ... 3*dim = x_1*x_2 ... x_1*x_n
 Polynomial 3*dim+1 ... 4*dim-1 = x_2*x_3 ... x_2*x_n
 ...
 Polynomial (dim^2 + 3*dim +2)/2 = x_{n-1}x_n
*/
class QuadraticLegendre {
  private:
    int dim;
  protected:
    //! Evaluation of Legendre polynomials
    /*!
     Evaluation of Legendre polynomial number p
     \param p Number of Legendre polynomials as discribed
     \param x Point where Legendre polynomial is evaluated
     \see QuadraticPolynomial
    */
    double evaluate_basis ( int, std::vector<double> const& );
  public:
    //! Constructor
    /*!
     \param dim_input Dimension of Legendre polynomials
    */
    QuadraticLegendre ( int dim_input ) : dim ( dim_input ) { }
};

#endif
