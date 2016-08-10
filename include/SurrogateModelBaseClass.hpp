#ifndef HSurrogateModelBaseClass
#define HSurrogateModelBaseClass

#include "BasisForSurrogateModelBaseClass.hpp"

//! Base clase for definition of surrogate models
/*!
 Defines the required structure of surrogate models to work with NOWPAC
*/
class SurrogateModelBaseClass {
  protected: 
    double model_constant;
    std::vector<double> model_gradient;
    std::vector< std::vector< double> > model_hessian;
    BasisForSurrogateModelBaseClass *basis;
//    BasisForSurrogateModelBaseClass *get_basis() { return basis; }
  public:
    //! Constructor
    /*!
     Constructor to set the basis for the surrogate model
     \see BasisForSurrogateModelBaseClass
     \param surrogate_basis basis for the surrogate model
    */
    SurrogateModelBaseClass ( BasisForSurrogateModelBaseClass &surrogate_basis ) 
                            { basis = &surrogate_basis; }
    //! Function to evaluate the surrogate model
    /*!
     Function to evaluate the surrogate model at point x
     \param x point at which the surrogate model is evaluated
     \returns value of the surrogate model at point x
    */
    virtual double evaluate ( std::vector<double> const& ) = 0;    
    //! Returns the gradient of the surrogate model at zero
    /*!
     \returns gradient of the surrogate model at zero
    */     
    virtual std::vector<double> &gradient ( ) = 0;
    //! Returns the Hessian matrix of the surrogate model at zero
    /*!
     \returns Hessian matrix of the surrogate model at zero
    */     
    virtual std::vector< std::vector<double> > &hessian ( ) = 0;
    //! Function to set the funtion values to be interpolated
    /*!
     Function to set the funtion values to be interpolated 
     \see BasisForSurrogateModelBaseClass
     \param values values of the function at nodes that defined in the basis
    */
    virtual void set_function_values ( std::vector<double> const& ) = 0;
    //! Function to query the dimension of the domain the surrogate model
    /*! 
     Function to query the dimension of the domain the surrogate model
     \returns the dimension of the domain
    */
    int dimension ( ) { return basis->dimension( ); }

};

#endif
