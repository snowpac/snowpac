//
// Created by friedrich on 16.08.16.
//

#ifndef NOWPAC_APPROXGAUSSIANPROCESSESAUGMENTEDSOR_H
#define NOWPAC_APPROXGAUSSIANPROCESSESAUGMENTEDSOR_H

#include "GaussianProcess.hpp"
#include "SubsetOfRegressors.hpp"

class AugmentedSubsetOfRegressors : public SubsetOfRegressors{
private:
    std::vector< double > augmented_u;
    double u_ratio = 0.1;
    int min_nb_u_nodes = 1;

public:
    const std::vector< int > get_induced_indices() const;

    void get_induced_nodes(std::vector< std::vector<double> >&) const;
    //! Constructor
    /*!
     Class constructor.
     \param n dimension of the Approximated Gaussian process.
    */
    AugmentedSubsetOfRegressors( int, double& );

    //! Destructor
    virtual ~AugmentedSubsetOfRegressors() { };

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

    //! Build the approximated Gaussian process
    /*!
     Computes the Gaussian process\n
     Requires the estimation of hyper parameters
     \param nodes regression points
     \param function values
     \param noise in function values
     \see estimate_hyper_parameters
    */
    void build_with_u ( std::vector< std::vector<double> > const&,
                 std::vector<double> const&, std::vector<double> const&);

    //! Build the approximated Gaussian process
    /*!
     Computes the Gaussian process\n
     Requires the estimation of hyper parameters
     \param nodes regression points
     \param function values
     \param noise in function values
     \see estimate_hyper_parameters
    */
    void build_from_evaluate ( std::vector< std::vector<double> > const&,
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
};

#endif //NOWPAC_APPROXGAUSSIANPROCESSESAUGMENTEDSOR_H
