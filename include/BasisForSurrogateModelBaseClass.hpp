#ifndef HBasisForSurrogateModelBaseClass
#define HBasisForSurrogateModelBaseClass

#include "BlackBoxData.hpp"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

//! Base class for definiton of surrogate model
/*!
 Defines the required structure of surrogate models to work with NOWPAC
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
     \param dim_input Number of arguments (dimension)
    */
    BasisForSurrogateModelBaseClass ( int n ) : dim ( n ) { };
    //! Destructor
    ~BasisForSurrogateModelBaseClass ( ) { };
    virtual void get_mat_vec_representation ( int, std::vector<double>&, 
                                              std::vector< std::vector<double> >& ) = 0;
    virtual void compute_basis_coefficients ( std::vector< std::vector<double> > const& ) = 0;
    virtual std::vector<double> &evaluate ( std::vector<double> const& ) = 0;
    virtual std::vector<double> &gradient ( std::vector<double> const&, int ) = 0;
    virtual std::vector<double> &gradient ( int ) = 0;
    virtual std::vector< std::vector<double> > &hessian ( std::vector<double> const&, int ) = 0;
    virtual std::vector< std::vector<double> > &hessian ( int ) = 0;
    virtual double evaluate ( std::vector<double> const&, int) = 0;
    int dimension ( ) { return dim; }
};

#endif
