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
    LDLT<MatrixXd> L_eigen;
    MatrixXd K_u_f;
    MatrixXd K_u_u;
    LDLT<MatrixXd> LLTofK_u_u;
    VectorXd Lambda;
    VectorXd Gamma;
    VectorXd alpha_eigen;
    VectorXd L_u;
    VectorXd K_u_f_LambdaInv_f;
    VectorXd K0_eigen;
    MatrixXd u;
    MatrixXd gp_nodes_eigen;
    VectorXd gp_noise_eigen;
    VectorXd scaled_function_values_eigen;
    std::vector<double> gp_parameters_hp;
    bool optimize_global = true;
    bool optimize_local = true;
    //bool resample_u = true;
    //bool do_hp_estimation = true;
    int print = 0;
    double K_u_u_nugget = 0.00001;
    double K_u_u_nugget_min = 1e-10;
    double K_u_u_nugget_max = 1e-2;

    double Kuu_opt_nugget = 0.00000001;
    double Lambda_opt_nugget = 0.00000001;
    double Qff_opt_nugget = 0.00000001;
    double nugget_max = 1e-5;
    double nugget_min = 1e-10;


    double constraint_ball_radius;
    VectorXd constraint_ball_center;
	void compute_Kuf_and_Kuu();
	void compute_Qff(const MatrixXd& K_f_u, VectorXd& diag_Q_f_f);
	void compute_Kff(VectorXd& diag_K_f_f);
	void compute_diff_Kff_Qff(const VectorXd& diag_K_f_f,
			const VectorXd& diag_Q_f_f, VectorXd& diff_Kff_Qff);
	void compute_Lambda(const VectorXd& diff_Kff_Qff,
			const std::vector<double>& noise);
	void compute_Lambda_times_Kfu(const MatrixXd& K_f_u, MatrixXd& Lambda_K_f_u);
	void compute_KufLambdaKfu(const MatrixXd& Lambda_K_f_u, MatrixXd& K_u_f_Lambda_f_u);
	void compute_LambdaInvF(VectorXd& LambdaInv_f);
    void compute_LambdaDot(const MatrixXd& Kffdot,
                                    const MatrixXd& Kufdot,
                                    const MatrixXd& Kfudot,
                                    const MatrixXd& Kuudot,
                                    VectorXd& LambdaDot);
    void compute_GammaDotDoubleBar(const MatrixXd& Kffdot,
                                    const MatrixXd& Kufdot,
                                    const MatrixXd& Kfudot,
                                    const MatrixXd& Kuudot,
                                    VectorXd& GammaRes);

    virtual void copy_data_to_members( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise);
    void set_optimizer(std::vector<double> const &values, nlopt::opt*& local_opt, nlopt::opt*& global_opt);

    virtual void run_optimizer(std::vector<double> const &values);

    virtual void update_induced_points();

    double evaluate_kernel ( VectorXd const &x,
                                          VectorXd const &y );

    double evaluate_kernel ( VectorXd const &x,
                                          VectorXd const &y,
                                          std::vector<double> const &p );


    double evaluate_kernel1D_exp_term(double const &x,
                              double const &y,
                              double const &l );

    void derivate_K_u_u_wrt_uik(std::vector<double> const &p, 
                                                int const &i, 
                                                int const &k,
                                                MatrixXd &deriv_matrix );

    void derivate_K_u_f_wrt_uik(std::vector<double> const &p, 
                                                int const &i, 
                                                int const &k,
                                                MatrixXd &deriv_matrix);

    void derivate_K_u_f_wrt_sigmaf(std::vector<double> const &p, 
                                        MatrixXd &deriv_matrix);

    void derivate_K_f_f_wrt_sigmaf(std::vector<double> const &p, 
                                        MatrixXd &deriv_matrix);

    void derivate_K_u_u_wrt_sigmaf(std::vector<double> const &p,
                                        MatrixXd &deriv_matrix);

    void derivate_K_u_f_wrt_l(std::vector<double> const &p,
                                             int const &k,
                                        MatrixXd &deriv_matrix);
    void derivate_K_f_f_wrt_l(std::vector<double> const &p,
                                             int const &k,
                                        MatrixXd &deriv_matrix);

    void derivate_K_u_u_wrt_l(std::vector<double> const &p,
                                             int const &k,
                                        MatrixXd &deriv_matrix);

    void set_hyperparameters();
    void set_hyperparameters_induced_only();
    void set_hyperparameters_ls_only();

    void copy_hyperparameters();

public:

    void set_constraint_ball_radius(const double& radius);

    void set_constraint_ball_center(const std::vector<double>& center);

    void get_induced_nodes(std::vector< std::vector<double> >&) const;
    //! Constructor
    /*!
     Class constructor.
     \param n dimension of the Approximated Gaussian process.
    */
    FullyIndependentTrainingConditional( int, double& );

    FullyIndependentTrainingConditional( int, double& , std::vector<double> );

    //! Destructor
    ~FullyIndependentTrainingConditional() { };

    bool test_for_parameter_estimation(const int& nb_values,
                                                const int& update_interval_length,
                                                const int& next_update,
                                                const std::vector<int>& update_at_evaluations);

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
   
    virtual void estimate_hyper_parameters ( std::vector< std::vector<double> > const &nodes,
                                                      std::vector<double> const &values,
                                                      std::vector<double> const &noise );

    virtual void estimate_hyper_parameters_all ( std::vector< std::vector<double> > const &nodes,
                                                      std::vector<double> const &values,
                                                      std::vector<double> const &noise );

    virtual void estimate_hyper_parameters_induced_only ( std::vector< std::vector<double> > const &nodes,
                                                      std::vector<double> const &values,
                                                      std::vector<double> const &noise );


    virtual void estimate_hyper_parameters_ls_only ( std::vector< std::vector<double> > const &nodes,
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

    static void trust_region_constraint_w_gradients(std::vector<double> &c, std::vector<double> const &x,
                                                           std::vector<double> &grad,
                                                           void *data);
    //virtual void set_hp_estimation(bool);

    //void do_resample_u();

    void sample_u(const int &nb_u_nodes);
    
    void clear_u();
    void decrease_nugget();
    bool increase_nugget();
};


#endif /* INCLUDE_FULLYINDEPENDENTTRAININGCONDITIONAL_HPP_ */
