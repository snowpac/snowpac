/*
 * FullyIndependentTrainingConditional.hpp
 *
 *  Created on: Nov 8, 2016
 *      Author: menhorn
 */

#ifndef INCLUDE_FULLYINDEPENDENTTRAININGCONDITIONAL_HPP_
#define INCLUDE_FULLYINDEPENDENTTRAININGCONDITIONAL_HPP_

#include "GaussianProcess.hpp"

#include <Eigen/Core>

using namespace Eigen;

class FullyIndependentTrainingConditional: public GaussianProcess{

protected:
    LLT<MatrixXd> L_eigen;
    MatrixXd K_u_f;
    MatrixXd K_u_u;
    VectorXd Lambda;
    VectorXd alpha_eigen;
    VectorXd L_u;
    VectorXd K_u_f_LambdaInv_f;
    VectorXd K0_eigen;
    MatrixXd u;
    MatrixXd gp_nodes_eigen;
    VectorXd gp_noise_eigen;
    double u_ratio = 0.1;
    int min_nb_u_nodes = 1;

    void sample_u(const int &nb_u_nodes);

public:
    void get_induced_nodes(std::vector< std::vector<double> >&) const;
    //! Constructor
    /*!
     Class constructor.
     \param n dimension of the Approximated Gaussian process.
    */
    FullyIndependentTrainingConditional( int, double& );

    //! Destructor
    ~FullyIndependentTrainingConditional() { };

    double evaluate_kernel ( VectorXd const &x,
                                          VectorXd const &y );

    double evaluate_kernel ( VectorXd const &x,
                                          VectorXd const &y,
                                          std::vector<double> const &p );
    //! Build the approximated Gaussian process
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

    //! Update the approximated Gaussian process
    /*!
     Includees a new point into the Gaussian process
     \param x new point to be included into the Gaussian process
     \param value new function value at new point
     \param noise new noise estimate at new function value
    */
    void update ( std::vector<double> const&, double&, double& );
    //! Evaluate approximated Gaussian process
    /*!
     Computes the mean and variance of the Gaussian process.\n
     Requires the building of the Gaussian process.
     \param x point at which the Gaussian process is evaluated
     \param mean mean of the Gaussian process at point x
     \param variance variance of the Gaussina process at point x
     \see build
    */
    void evaluate ( std::vector<double> const&, double&, double& );
   /*
    void estimate_hyper_parameters ( std::vector< std::vector<double> > const &nodes,
                                                      std::vector<double> const &values,
                                                      std::vector<double> const &noise );

    double parameter_estimation_objective(std::vector<double> const &x,
                                                           std::vector<double> &grad,
                                                           void *data);
   */
};


#endif /* INCLUDE_FULLYINDEPENDENTTRAININGCONDITIONAL_HPP_ */
