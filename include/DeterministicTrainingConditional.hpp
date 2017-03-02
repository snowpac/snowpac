//
// Created by friedrich on 16.08.16.
//

#ifndef NOWPAC_DETERMINISTICTRAININGCONDITIONAL_H
#define NOWPAC_DETERMINISTICTRAININGCONDITIONAL_H

#include "SubsetOfRegressors.hpp"

class DeterministicTrainingConditional : public SubsetOfRegressors{

protected:
    void run_optimizer(std::vector<double> const &values);
public:
    //! Constructor
    /*!
     Class constructor.
     \param n dimension of the Approximated Gaussian process.
    */
    DeterministicTrainingConditional( int, double& );

    DeterministicTrainingConditional( int, double& , std::vector<double> );

    //! Destructor
    ~DeterministicTrainingConditional() { };

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

    //void estimate_hyper_parameters ( std::vector< std::vector<double> > const &nodes,
    //                                              std::vector<double> const &values,
    //                                              std::vector<double> const &noise );

    void estimate_hyper_parameters_all ( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise );

    void estimate_hyper_parameters_induced_only ( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise );
    void estimate_hyper_parameters_ls_only ( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise );


    static double parameter_estimation_objective(std::vector<double> const &x,
                                                           std::vector<double> &grad,
                                                           void *data);

    static double parameter_estimation_objective_w_gradients(std::vector<double> const &x,
                                                       std::vector<double> &grad,
                                                       void *data);

    static void trust_region_constraint(unsigned int m, double* c, unsigned int n, const double* x, double* grad,
                                                                void *data);
};

#endif //NOWPAC_DETERMINISTICTRAININGCONDITIONAL_H
