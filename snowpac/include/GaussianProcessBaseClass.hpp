#ifndef HGaussianProcessBaseClass
#define HGaussianProcessBaseClass

#include "GaussianProcessKernelBaseClass.hpp"
#include <Eigen/Dense>
#include <vector>

//! Gaussian process regression
/*!
 Interface for Gaussian process regression
 \see GaussianProcessKernelBaseClasss
*/
class GaussianProcessBaseClass : GaussianProcessKernelBaseClass {
  public:
    //! Constructor
    GaussianProcessBaseClass ( ) { }
    //! Destructor
    ~GaussianProcessBaseClass ( ) { }
    //! Build the Gaussian process
    virtual void build ( std::vector< std::vector<double> > const&,
                         std::vector<double> const&,
                         std::vector<double> const& ) = 0;
    //! Update the Gaussian process
    virtual void update ( std::vector<double> const&,
                          double&, double& ) = 0;
    //! Evaluate Gaussian process
    virtual void evaluate ( std::vector<double> const&,
                            double&, double& ) = 0;
};


#endif
