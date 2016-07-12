#ifndef HGaussianProcessKernelBaseClass
#define HGaussianProcessKernelBaseClass

#include <Eigen/Core>
#include <vector>

//! Interface for Gaussian kernel defintion
class GaussianProcessKernelBaseClass {
  public:
    //! Virtual member function for the estimation of hyper parameters of the kernel
    virtual void estimate_hyper_parameters ( std::vector< std::vector<double> > const&,
                                             std::vector<double> const&, 
                                             std::vector<double> const& ) = 0;
    //! Virtual membber function for the evaluation of the kernel
    virtual double evaluate_kernel ( std::vector<double> const&, 
                                     std::vector<double> const& ) = 0;
};

#endif
