#ifndef HGaussianProcess
#define HGaussianProcess

#include "GaussianProcessBaseClass.hpp"
#include "CholeskyFactorization.hpp"
#include "TriangularMatrixOperations.hpp"
#include "VectorOperations.hpp"
#include "nlopt.hpp"
#include <memory>

//! Gaussian process regression
/*!
 Computes a Gaussian process of given data points, function evaluations and noise estimates.
 \see GaussianProcessBaseClass
 \see CholeskyFactorization
 \see TriangularMatrixOperations
*/
class GaussianProcess : public GaussianProcessBaseClass,
                        protected CholeskyFactorization, 
                        protected TriangularMatrixOperations,
                        protected VectorOperations {

  private:
    //auxiliary variables
    int pos;
    double rho;
    std::vector<double> K0;
    double kernel_evaluation, dist;
    std::vector<double> gp_parameters;
    std::vector<double> lb, ub;
    //general/shared auxiliary variables
    std::vector< std::vector<double> > L;
    std::vector<double> alpha;    

    GaussianProcess *gp_pointer;
    std::vector< std::vector<double> > gp_nodes;
    std::vector<double> scaled_function_values;
    double min_function_value, max_function_value;
    std::vector<double> gp_noise;
    int dim, nb_gp_nodes;
    double *delta;
    double noise_regularization = 1e-6;

    static double parameter_estimation_objective(std::vector<double> const&, 
                                                 std::vector<double>&, void*);
  protected:
    //! Evaluation of Gaussian process kernel
    /*!
     Evaluates the square exponential kernel.
    */
    double evaluate_kernel ( std::vector<double> const&, std::vector<double> const& );
    double evaluate_kernel ( std::vector<double> const&, std::vector<double> const&,
                             std::vector<double> const& );
    //! Evaluation of the derivative of the Gaussina process kernel
    double d_evaluate_kernel ( std::vector<double> const&, std::vector<double> const&, 
                               std::vector<double> const&, int );
  public:
    //! Constructor
    /*!
     Class constructor.
     \param n dimension of the Gaussian process.
    */
    GaussianProcess( int, double& );
    //! Destructor
    ~GaussianProcess() { }
    //! Estimation of hyper parameters
    /*!
     Estimates the hyper parameters of the Gaussian process.\n 
     The hyper parameters are the variance and the length scale parameters in the exponential kernel.\n
     \param nodes regression points
     \param function values 
     \param noise in function values
    */
    void estimate_hyper_parameters ( std::vector< std::vector<double> > const&,
                                     std::vector<double> const&, 
                                     std::vector<double> const&);
    //! Build the Gaussian process
    /*!
     Computes the Gaussian process\n
     Requires the estimation of hyper parameters
     \param nodes regression points
     \param function values
     \param noise in function values
     \see estimate_hyper_parameters
    */
    void build ( std::vector< std::vector<double> > const&,
                 std::vector<double> const&, std::vector<double> const&);
    //! Update the Gaussian process
    /*!
     Includees a new point into the Gaussian process
     \param x new point to be included into the Gaussian process
     \param value new function value at new point
     \param noise new noise estimate at new function value
    */
    void update ( std::vector<double> const&, double&, double& );
    //! Evaluate Gaussian process
    /*!
     Computes the mean and variance of the Gaussian process.\n
     Requires the building of the Gaussian process.
     \param x point at which the Gaussian process is evaluated
     \param mean mean of the Gaussian process at point x
     \param variance variance of the Gaussina process at point x
     \see build
    */
    void evaluate ( std::vector<double> const&, double&, double& );
};

#endif
