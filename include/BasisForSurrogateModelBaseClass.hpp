#ifndef HBasisForSurrogateModelBaseClass
#define HBasisForSurrogateModelBaseClass

#include "BlackBoxData.hpp"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

//! Base class for definiton of basis for surrogate models
/*!
 Defines the required structure of basis for surrogate models to work with NOWPAC
*/
class BasisForSurrogateModelBaseClass {
  protected:
    //! Number of arguments of surrogate model
    int dim;
    //! Number of basic basis functions
    /*!  
     Number of basis function, for example quadratic monomials, 
     for surrogate basis functions
    */
    int nb_basis_functions;
    //! Coefficients of surrogate basis functions in terms of basic basis function
    std::vector<Eigen::VectorXd> basis_coefficients;
  public:
    //! Constructor
    /*! 
     Constructor to set number of arguments (dimension) of the basis
     \param n Number of arguments (dimension)
    */
    BasisForSurrogateModelBaseClass ( int n ) : dim ( n ) { };
    //! Destructor
    ~BasisForSurrogateModelBaseClass ( ) { };
    //! Function to compute the basis coefficients for the basis functions
    /*!
     Function to compute the basis coefficients for the basis functions
     \param x vector vectors of nodes x[0], ..., x[n] contain the interpolation nodes
    */
    virtual void compute_basis_coefficients ( std::vector< std::vector<double> > const& ) = 0;
    //! Function to evaluate the basis functions
    /*!
     Function to evaluate the basis functions. It returns a vector of basis values at
     the point x.
     \param x point x at which the basis functions are evaluated
     \returns the values of all basis functions evaluated at point x
    */
    virtual std::vector<double> &evaluate ( std::vector<double> const& ) = 0;
    //! Returns the value of a basis function at zero
    /*!
     Returns the value of basis function i at zero
     \param i number of basis function whose value is queried
     \returns the value of the basis function i at zero
    */
    virtual double &value ( int ) = 0;
    //! Returns the gradient of a basis function zero
    /*!
     Returns the gradient of basis function i at zero
     \param i number of basis function whose gradient at zero is queried
     \returns the gradient of the basis function i at zero
    */
    virtual std::vector<double> &gradient ( int ) = 0;
    //! Returns the Hessian matrix of a basis function zero
    /*!
     Returns the Hessian matrix of basis function i at zero
     \param i number of basis function whose Hessian matrix at zero is queried
     \returns the Hessian matrix of the basis function i at zero
    */
    virtual std::vector< std::vector<double> > &hessian ( int ) = 0;
    //! Function to evaluate a basis function 
    /*!
     Function to evaluate the i-th basis function. It returns the value of the i-th 
     basis function.
     \param x point x at which the i-th basis function are evaluated
     \param i number of basis function to be evaluated
     \returns the value of the i-th bais function at point x
    */
    virtual double evaluate ( std::vector<double> const&, int) = 0;
    //! Function to query the dimension of the domain the basis functions 
    /*! 
     Function to query the dimension of the domain the basis functions
     \returns the dimension of the domain
    */
    int dimension ( ) { return dim; }
};

#endif
