/*
 * FullyIndependentTrainingConditional.cpp
 *
 *  Created on: Nov 8, 2016
 *      Author: menhorn
 */

#include <FullyIndependentTrainingConditional.hpp>
#include <random>
#include <assert.h>
#include <algorithm>
#include <iostream>
#include <limits.h>
#include <cmath>

 #include <Eigen/Dense>

//--------------------------------------------------------------------------------
FullyIndependentTrainingConditional::FullyIndependentTrainingConditional(int n,
		double &delta_input) :
		GaussianProcess(n, delta_input, nullptr) {

}
//--------------------------------------------------------------------------------
FullyIndependentTrainingConditional::FullyIndependentTrainingConditional(int n,
		double &delta_input, std::vector<double> gp_parameters_input) :
		GaussianProcess(n, delta_input, nullptr, gp_parameters_input) {

}
//--------------------------------------------------------------------------------
double FullyIndependentTrainingConditional::evaluate_kernel ( VectorXd const &x,
                                          VectorXd const &y )
{
  return evaluate_kernel ( x, y, gp_parameters );
}
//--------------------------------------------------------------------------------
double FullyIndependentTrainingConditional::evaluate_kernel ( VectorXd const &x,
                                          VectorXd const &y,
                                          std::vector<double> const &p )
{
  dist = 0e0;
  for ( int i = 0; i < dim; ++i )
    dist += pow( (x(i) - y(i)), 2e0) /  p.at( i+1 );
  kernel_evaluation = exp(-dist / 2e0 );
  
  return kernel_evaluation * p.at( 0 ) ;
}

double FullyIndependentTrainingConditional::evaluate_kernel1D_exp_term(double const &x,
                              double const &y,
                              double const &l ){
	return exp(-0.5*((x-y)*(x-y))/ l);
}

void FullyIndependentTrainingConditional::derivate_K_u_u_wrt_uik(std::vector<double> const &p,
																	 int const &i, 
																	 int const &k,
																	 MatrixXd &deriv_matrix ){
	const int nb_u_nodes = u.rows();
	deriv_matrix.resize(nb_u_nodes, nb_u_nodes);
	deriv_matrix.setZero();
	double C = 0;
	for(int j = 0; j < nb_u_nodes; ++j){
		C = p[0]; //sigma_f^2
		for(int d = 0; d < dim; ++d){
			C *= evaluate_kernel1D_exp_term(u(i, d), u(j, d), p[1+d]);
		}
		deriv_matrix(i,j) = C * (-1) * (u(i, k)-u(j, k))/p[k+1];
	}
	for(int j = 0; j < nb_u_nodes; ++j){
		deriv_matrix(j,i) = deriv_matrix(i,j);
	}
	/*//Good
	MatrixXd K_u_u_fd1(nb_u_nodes, nb_u_nodes);
	MatrixXd K_u_u_fd2(nb_u_nodes, nb_u_nodes);

	MatrixXd ufd1(nb_u_nodes, dim);
	MatrixXd ufd2(nb_u_nodes, dim);
	for(int j = 0; j < nb_u_nodes; ++j){
		for(int d = 0; d < dim; ++d){
			ufd1(j,d) = u(j,d);
			ufd2(j,d) = u(j,d);
		}
	}
	ufd1(i, k) += 0.1;
	ufd2(i, k) -= 0.1;
	for (int h = 0; h < nb_u_nodes; ++h) {
		for (int j = 0; j < nb_u_nodes; ++j) {
			K_u_u_fd1(h, j) = evaluate_kernel(ufd1.row(h), ufd1.row(j));
			K_u_u_fd2(h, j) = evaluate_kernel(ufd2.row(h), ufd2.row(j));
		}
	}
	K_u_u_fd1 = 1.0/0.2 * (K_u_u_fd1-K_u_u_fd2);
	std::cout << "Kufdot " << deriv_matrix << std::endl;
	std::cout << "K_u_f_fd1 " << K_u_u_fd1 << std::endl;
	exit(-1);
	*/
}


void FullyIndependentTrainingConditional::derivate_K_u_f_wrt_uik(std::vector<double> const &p,
																	 int const &i, 
																	 int const &k,
																	 MatrixXd &deriv_matrix ){
	const int nb_u_nodes = u.rows();
	deriv_matrix.resize(nb_u_nodes, nb_gp_nodes);
	deriv_matrix.setZero();
	double C = 0;
	for(int j = 0; j < nb_gp_nodes; ++j){
		C = p[0]; //sigma_f^2
		for(int d = 0; d < dim; ++d){
			C *= evaluate_kernel1D_exp_term(u(i, d), gp_nodes_eigen(j, d), p[1+d]);
		}
		deriv_matrix(i,j) = C * (-1) * (u(i, k)-gp_nodes_eigen(j, k))/p[k+1];
	}
	
	/*Good
	MatrixXd K_f_u_fd1(nb_gp_nodes, nb_u_nodes);
	MatrixXd K_f_u_fd2(nb_gp_nodes, nb_u_nodes);

	MatrixXd ufd1(nb_u_nodes, dim);
	MatrixXd ufd2(nb_u_nodes, dim);
	for(int j = 0; j < nb_u_nodes; ++j){
		for(int d = 0; d < dim; ++d){
			ufd1(j,d) = u(j,d);
			ufd2(j,d) = u(j,d);
		}
	}
	ufd1(i, k) += 0.1;
	ufd2(i, k) -= 0.1;
	for (int h = 0; h < nb_u_nodes; ++h) {
		for (int j = 0; j < nb_gp_nodes; ++j) {
			K_f_u_fd1(j, h) = evaluate_kernel(gp_nodes_eigen.row(j), ufd1.row(h));
			K_f_u_fd2(j, h) = evaluate_kernel(gp_nodes_eigen.row(j), ufd2.row(h));
		}
	}
	K_f_u_fd1 = 1.0/0.2 * (K_f_u_fd1-K_f_u_fd2);
	std::cout << "Kufdot " << deriv_matrix << std::endl;
	std::cout << "K_f_u_fd1 " << K_f_u_fd1 << std::endl;
	exit(-1);
	*/
	
}

void FullyIndependentTrainingConditional::derivate_K_u_f_wrt_sigmaf(std::vector<double> const &p, 
                                                    MatrixXd &deriv_matrix){
	const int nb_u_nodes = u.rows();
	deriv_matrix.resize(nb_u_nodes, nb_gp_nodes);
	deriv_matrix.setZero();
	double C = 0;
	//Set up matrix d(K_u_f)/d(sigma_f^2)
	for (int i = 0; i < nb_u_nodes; ++i) {
		for (int j = 0; j < nb_gp_nodes; ++j) {
			C = 1.0;
			for(int d = 0; d < dim; ++d){
				C *= evaluate_kernel1D_exp_term(u(i, d), gp_nodes_eigen(j, d), p[1+d]);
			}
			deriv_matrix(i,j) = C;
		}
	}
	/*Good
	MatrixXd K_f_u_fd1(nb_gp_nodes, nb_u_nodes);
	MatrixXd K_f_u_fd2(nb_gp_nodes, nb_u_nodes);

	std::vector<double> gpfd1 = gp_parameters;
	gpfd1[0] += 0.1; 
	std::vector<double> gpfd2 = gp_parameters;
	gpfd2[0] -= 0.1;
	for (int h = 0; h < nb_u_nodes; ++h) {
		for (int j = 0; j < nb_gp_nodes; ++j) {
			K_f_u_fd1(j, h) = evaluate_kernel(gp_nodes_eigen.row(j), u.row(h), gpfd1);
			K_f_u_fd2(j, h) = evaluate_kernel(gp_nodes_eigen.row(j), u.row(h), gpfd2);
		}
	}
	K_f_u_fd1 = 1.0/0.2 * (K_f_u_fd1-K_f_u_fd2);
	std::cout << "Kufdot " << deriv_matrix << std::endl;
	std::cout << "K_u_f_fd1 " << K_f_u_fd1 << std::endl;
	exit(-1);
	*/
}

void FullyIndependentTrainingConditional::derivate_K_f_f_wrt_sigmaf(std::vector<double> const &p, 
                                                    MatrixXd &deriv_matrix){
	deriv_matrix.resize(nb_gp_nodes, nb_gp_nodes);
	deriv_matrix.setZero();
	double C = 0;
	//Set up matrix d(K_u_f)/d(sigma_f^2)
	for (int i = 0; i < nb_gp_nodes; ++i) {
		for (int j = 0; j < nb_gp_nodes; ++j) {
			C = 1.0;
			for(int d = 0; d < dim; ++d){
				C *= evaluate_kernel1D_exp_term(gp_nodes_eigen(i, d), gp_nodes_eigen(j, d), p[1+d]);
			}
			deriv_matrix(i,j) = C;
		}
	}
	/*Good
	MatrixXd K_f_f_fd1(nb_gp_nodes, nb_gp_nodes);
	MatrixXd K_f_f_fd2(nb_gp_nodes, nb_gp_nodes);

	std::vector<double> gpfd1 = gp_parameters;
	gpfd1[0] += 0.1; 
	std::vector<double> gpfd2 = gp_parameters;
	gpfd2[0] -= 0.1;
	for (int h = 0; h < nb_gp_nodes; ++h) {
		for (int j = 0; j < nb_gp_nodes; ++j) {
			K_f_f_fd1(j, h) = evaluate_kernel(gp_nodes_eigen.row(j), gp_nodes_eigen.row(h), gpfd1);
			K_f_f_fd2(j, h) = evaluate_kernel(gp_nodes_eigen.row(j), gp_nodes_eigen.row(h), gpfd2);
		}
	}
	K_f_f_fd1 = 1.0/0.2 * (K_f_f_fd1-K_f_f_fd2);
	std::cout << "Kufdot " << deriv_matrix << std::endl;
	std::cout << "K_f_f_fd1 " << K_f_f_fd1 << std::endl;
	exit(-1);
	*/
}

void FullyIndependentTrainingConditional::derivate_K_u_u_wrt_sigmaf(std::vector<double> const &p, 
                                                    MatrixXd &deriv_matrix){
	const int nb_u_nodes = u.rows();
	deriv_matrix.resize(nb_u_nodes, nb_u_nodes);
	deriv_matrix.setZero();
	double C = 0;
	//Set up matrix d(K_u_f)/d(sigma_f^2)
	for (int i = 0; i < nb_u_nodes; ++i) {
		for (int j = 0; j < nb_u_nodes; ++j) {
			C = 1.0;//sigma_f^2
			for(int d = 0; d < dim; ++d){
				C *= evaluate_kernel1D_exp_term(u(i, d), u(j, d), p[1+d]);
			}
			deriv_matrix(i,j) = C;
		}
	}
	/*//Good
	MatrixXd K_u_u_fd1(nb_u_nodes, nb_u_nodes);
	MatrixXd K_u_u_fd2(nb_u_nodes, nb_u_nodes);

	std::vector<double> gpfd1 = gp_parameters;
	gpfd1[0] += 0.1; 
	std::vector<double> gpfd2 = gp_parameters;
	gpfd2[0] -= 0.1;
	for (int h = 0; h < nb_u_nodes; ++h) {
		for (int j = 0; j < nb_u_nodes; ++j) {
			K_u_u_fd1(j, h) = evaluate_kernel(u.row(j), u.row(h), gpfd1);
			K_u_u_fd2(j, h) = evaluate_kernel(u.row(j), u.row(h), gpfd2);
		}
	}
	K_u_u_fd1 = 1.0/0.2 * (K_u_u_fd1-K_u_u_fd2);
	std::cout << "Kuudot " << deriv_matrix << std::endl;
	std::cout << "K_u_u_fd1 " << K_u_u_fd1 << std::endl;
	exit(-1);
	*/
}

void FullyIndependentTrainingConditional::derivate_K_u_f_wrt_l(std::vector<double> const &p,
																	 int const &k,
                                        							MatrixXd &deriv_matrix){
	const int nb_u_nodes = u.rows();
	deriv_matrix.resize(nb_u_nodes, nb_gp_nodes);
	deriv_matrix.setZero();
	double C = 0;
	//Set up matrix d(K_u_f)/d(sigma_f^2)
	for (int i = 0; i < nb_u_nodes; ++i) {
		for (int j = 0; j < nb_gp_nodes; ++j) {
			C = p[0];
			for(int d = 0; d < dim; ++d){
				C *= evaluate_kernel1D_exp_term(u(i, d), gp_nodes_eigen(j, d), p[1+d]);
			}
			deriv_matrix(i,j) = C * (u(i, k)-gp_nodes_eigen(j, k))*(u(i, k)-gp_nodes_eigen(j, k))/
											(2*p[k+1]*p[k+1]);
		}
	}
	/*//Good
	MatrixXd K_f_u_fd1(nb_gp_nodes, nb_u_nodes);
	MatrixXd K_f_u_fd2(nb_gp_nodes, nb_u_nodes);

	std::vector<double> gpfd1 = gp_parameters;
	gpfd1[k+1] += 0.1; 
	std::vector<double> gpfd2 = gp_parameters;
	gpfd2[k+1] -= 0.1;
	for (int h = 0; h < nb_u_nodes; ++h) {
		for (int j = 0; j < nb_gp_nodes; ++j) {
			K_f_u_fd1(j, h) = evaluate_kernel(gp_nodes_eigen.row(j), u.row(h), gpfd1);
			K_f_u_fd2(j, h) = evaluate_kernel(gp_nodes_eigen.row(j), u.row(h), gpfd2);
		}
	}
	K_f_u_fd1 = 1.0/0.2 * (K_f_u_fd1-K_f_u_fd2);
	std::cout << "Kufdot " << deriv_matrix << std::endl;
	std::cout << "K_u_f_fd1 " << K_f_u_fd1 << std::endl;
	exit(-1);
	*/
}

void FullyIndependentTrainingConditional::derivate_K_f_f_wrt_l(std::vector<double> const &p,
																	 int const &k,
                                        							MatrixXd &deriv_matrix){
	deriv_matrix.resize(nb_gp_nodes, nb_gp_nodes);
	deriv_matrix.setZero();
	double C = 0;
	//Set up matrix d(K_u_f)/d(sigma_f^2)
	for (int i = 0; i < nb_gp_nodes; ++i) {
		for (int j = 0; j < nb_gp_nodes; ++j) {
			C = p[0];
			for(int d = 0; d < dim; ++d){
				C *= evaluate_kernel1D_exp_term(gp_nodes_eigen(i, d), gp_nodes_eigen(j, d), p[1+d]);
			}
			deriv_matrix(i,j) = C * (gp_nodes_eigen(i, k)-gp_nodes_eigen(j, k))*
									(gp_nodes_eigen(i, k)-gp_nodes_eigen(j, k))/
											(2*p[k+1]*p[k+1]);
		}
	}
	/*//Good
	MatrixXd K_f_f_fd1(nb_gp_nodes, nb_gp_nodes);
	MatrixXd K_f_f_fd2(nb_gp_nodes, nb_gp_nodes);

	std::vector<double> gpfd1 = gp_parameters;
	gpfd1[k+1] += 0.1; 
	std::vector<double> gpfd2 = gp_parameters;
	gpfd2[k+1] -= 0.1;
	for (int h = 0; h < nb_gp_nodes; ++h) {
		for (int j = 0; j < nb_gp_nodes; ++j) {
			K_f_f_fd1(j, h) = evaluate_kernel(gp_nodes_eigen.row(j), gp_nodes_eigen.row(h), gpfd1);
			K_f_f_fd2(j, h) = evaluate_kernel(gp_nodes_eigen.row(j), gp_nodes_eigen.row(h), gpfd2);
		}
	}
	K_f_f_fd1 = 1.0/0.2 * (K_f_f_fd1-K_f_f_fd2);
	std::cout << "Kffdot " << deriv_matrix << std::endl;
	std::cout << "K_f_f_fd1 " << K_f_f_fd1 << std::endl;
	exit(-1);
	*/
}

void FullyIndependentTrainingConditional::derivate_K_u_u_wrt_l(std::vector<double> const &p,
                                         int const &k,
                                    MatrixXd &deriv_matrix){
	const int nb_u_nodes = u.rows();
	deriv_matrix.resize(nb_u_nodes, nb_u_nodes);
	deriv_matrix.setZero();
	double C = 0;
	//Set up matrix d(K_u_f)/d(sigma_f^2)
	for (int i = 0; i < nb_u_nodes; ++i) {
		for (int j = 0; j < nb_u_nodes; ++j) {
			C = p[0];
			for(int d = 0; d < dim; ++d){
				C *= evaluate_kernel1D_exp_term(u(i, d), u(j, d), p[1+d]);
			}
			deriv_matrix(i,j) = C * (u(i, k)-u(j, k))*(u(i, k)-u(j, k))/
											(2*p[k+1]*p[k+1]);
		}
	}
	/*//Good
	MatrixXd K_u_u_fd1(nb_u_nodes, nb_u_nodes);
	MatrixXd K_u_u_fd2(nb_u_nodes, nb_u_nodes);

	std::vector<double> gpfd1 = gp_parameters;
	gpfd1[k+1] += 0.1; 
	std::vector<double> gpfd2 = gp_parameters;
	gpfd2[k+1] -= 0.1;
	for (int h = 0; h < nb_u_nodes; ++h) {
		for (int j = 0; j < nb_u_nodes; ++j) {
			K_u_u_fd1(j, h) = evaluate_kernel(u.row(j), u.row(h), gpfd1);
			K_u_u_fd2(j, h) = evaluate_kernel(u.row(j), u.row(h), gpfd2);
		}
	}
	K_u_u_fd1 = 1.0/0.2 * (K_u_u_fd1-K_u_u_fd2);
	std::cout << "Kuudot " << deriv_matrix << std::endl;
	std::cout << "K_u_u_fd1 " << K_u_u_fd1 << std::endl;
	exit(-1);
	*/
}


void FullyIndependentTrainingConditional::compute_Kuf_and_Kuu() {
	int nb_u_nodes = u.rows();
	//Set up matrix K_u_f and K_f_u
	K_u_f.resize(nb_u_nodes, nb_gp_nodes);
	for (int i = 0; i < nb_u_nodes; ++i) {
		for (int j = 0; j < nb_gp_nodes; ++j) {
			K_u_f(i, j) = evaluate_kernel(u.row(i), gp_nodes_eigen.row(j));
		}
	}
	//Set up matrix K_u_u
	K_u_u.resize(nb_u_nodes, nb_u_nodes);
	for (int i = 0; i < nb_u_nodes; ++i) {
		for (int j = 0; j < nb_u_nodes; ++j) {
			K_u_u(i, j) = evaluate_kernel(u.row(i), u.row(j));
			if (i == j)
				K_u_u(i, j) += K_u_u_nugget;
		}
	}
	LLTofK_u_u.compute(K_u_u);
}

 void FullyIndependentTrainingConditional::compute_Qff(const MatrixXd& K_f_u, VectorXd& diag_Q_f_f) {
	int nb_u_nodes = u.rows();
	MatrixXd K_u_u_u_f = LLTofK_u_u.solve(K_u_f);

	diag_Q_f_f.resize(nb_gp_nodes);
	for (int i = 0; i < nb_gp_nodes; ++i) {
		diag_Q_f_f(i) = 0.0;
		for (int j = 0; j < nb_u_nodes; ++j) {
			diag_Q_f_f(i) += (K_f_u(i, j) * K_u_u_u_f(j, i));
		}
	}
}

void FullyIndependentTrainingConditional::compute_Kff(VectorXd& diag_K_f_f) {
	diag_K_f_f.resize(nb_gp_nodes);
	for (int i = 0; i < nb_gp_nodes; ++i) {
		diag_K_f_f(i) = evaluate_kernel(gp_nodes_eigen.row(i),
				gp_nodes_eigen.row(i));
	}
}

void FullyIndependentTrainingConditional::compute_diff_Kff_Qff(const VectorXd& diag_K_f_f,
		const VectorXd& diag_Q_f_f, VectorXd& diff_Kff_Qff) {
	diff_Kff_Qff.resize(nb_gp_nodes);
	for (int i = 0; i < nb_gp_nodes; ++i) {
		diff_Kff_Qff(i) = (diag_K_f_f(i) - diag_Q_f_f(i));
	}
}

void FullyIndependentTrainingConditional::compute_Lambda(const VectorXd& diff_Kff_Qff, const std::vector<double>& noise) {
	Lambda.resize(nb_gp_nodes);
	for (int i = 0; i < nb_gp_nodes; ++i) {
		Lambda(i) = (diff_Kff_Qff(i) + pow(noise.at(i) / 2e0 + noise_regularization, 2e0));
	}
}

void FullyIndependentTrainingConditional::compute_Lambda_times_Kfu(const MatrixXd& K_f_u, MatrixXd& Lambda_K_f_u) {
	int nb_u_nodes = u.rows();
	Lambda_K_f_u.resize(nb_gp_nodes, nb_u_nodes);
	for (int i = 0; i < nb_gp_nodes; i++) {
		for (int j = 0; j < nb_u_nodes; j++) {
			Lambda_K_f_u(i, j) = ((1.0 / Lambda(i)) * K_f_u(i, j));
		}
	}
}

void FullyIndependentTrainingConditional::compute_KufLambdaKfu(const MatrixXd& Lambda_K_f_u, MatrixXd& K_u_f_Lambda_f_u) {
	int nb_u_nodes = u.rows();
	K_u_f_Lambda_f_u.resize(nb_u_nodes, nb_u_nodes);
	K_u_f_Lambda_f_u = K_u_f * Lambda_K_f_u;
}

void FullyIndependentTrainingConditional::compute_LambdaInvF(VectorXd& LambdaInv_f) {
	LambdaInv_f.resize(nb_gp_nodes);
	for (int i = 0; i < nb_gp_nodes; ++i) {
		LambdaInv_f(i) = 1.0 / Lambda(i) * scaled_function_values[i];
	}
}

void FullyIndependentTrainingConditional::compute_LambdaDot(const MatrixXd& Kffdot,
                                    const MatrixXd& Kufdot,
                                    const MatrixXd& Kfudot,
                                    const MatrixXd& Kuudot,
                                    VectorXd& LambdaDot){
	LambdaDot.resize(nb_gp_nodes);
	LambdaDot.setZero();
	MatrixXd KfudotKuinvKuf = Kfudot * LLTofK_u_u.solve(K_u_f);
	MatrixXd K_f_u = K_u_f.transpose();
	//MatrixXd KfuKuinvKufdot = K_f_u * LLTofK_u_u.solve(Kufdot);
	MatrixXd KfuKuinfKudotKuinvKuf = K_f_u*(LLTofK_u_u.solve(Kuudot*(LLTofK_u_u.solve(K_u_f))));

	for(int i = 0; i < nb_gp_nodes; ++i){
		LambdaDot(i) = Kffdot(i,i) - 2*KfudotKuinvKuf(i,i) + KfuKuinfKudotKuinvKuf(i,i);// - KfuKuinvKufdot(i,i);
	}

}

void FullyIndependentTrainingConditional::compute_GammaDotDoubleBar(const MatrixXd& Kffdot,
																	const MatrixXd& Kufdot,
																	const MatrixXd& Kfudot,
																	const MatrixXd& Kuudot,
																	VectorXd& GammaRes){
	GammaRes.resize(nb_gp_nodes);
	GammaRes.setZero();
	MatrixXd KfudotKuinvKuf = Kfudot * LLTofK_u_u.solve(K_u_f);
	MatrixXd K_f_u = K_u_f.transpose();
	MatrixXd KfuKuinvKufdot = K_f_u * LLTofK_u_u.solve(Kufdot);
	MatrixXd KfuKuinfKudotKuinvKuf = K_f_u*(LLTofK_u_u.solve(Kuudot*(LLTofK_u_u.solve(K_u_f))));
	
	double diag_ii;
	for(int i = 0; i < nb_gp_nodes; ++i){
		diag_ii = Kffdot(i,i) - KfudotKuinvKuf(i,i) + KfuKuinfKudotKuinvKuf(i,i) - KfuKuinvKufdot(i,i);
		GammaRes(i) = 1/Lambda(i) * diag_ii;
	}

}

bool FullyIndependentTrainingConditional::test_for_parameter_estimation(const int& nb_values,
                                                const int& update_interval_length,
                                                const int& next_update,
                                                const std::vector<int>& update_at_evaluations){

  bool do_parameter_estimation = GaussianProcess::test_for_parameter_estimation(nb_values, update_interval_length, next_update, update_at_evaluations);
  if (do_parameter_estimation){
  	std::cout << "Regular update step." << std::endl;
  	return do_parameter_estimation;
  } 

  //check if induced points are still in bounds;
  IOFormat HeavyFmt(FullPrecision, 0, ", ", ";\n", "", "", "[", "]");
  double dist = 0.0;
  const double constraint_ball_radius_squared = constraint_ball_radius*constraint_ball_radius;
  for(int i = 0; i < u.rows(); ++i){
  	dist = 0.0;
  	for(int j = 0; j < u.cols(); ++j){
  		dist += (u(i,j)-constraint_ball_center(j))*(u(i,j)-constraint_ball_center(j));
  	}
  	dist = sqrt(dist);
	if(dist > (constraint_ball_radius*1.000001)){
  		std::cout << "Induced point out of bounds! Index: " << i << " Radius: " << constraint_ball_radius << " Dist: " << dist << std::endl;
  		for(int j = 0; j < u.cols(); ++j){
  			std::cout << "u: " << u(i, j) << " |c: " << constraint_ball_center(j) << std::endl;
  		}
  		do_parameter_estimation = true;
  		return do_parameter_estimation;
  	}
  }
  do_parameter_estimation = false;

  return do_parameter_estimation;
}

//--------------------------------------------------------------------------------
void FullyIndependentTrainingConditional::build(
		std::vector<std::vector<double> > const &nodes,
		std::vector<double> const &values, std::vector<double> const &noise) {

	int nb_u_nodes = u.rows();

	if (nb_u_nodes > 0) {
		std::cout << "FITC build with [" << nodes.size() << "," << nb_u_nodes
			<< "]" << std::endl;
		std::cout << "With Parameters: " << std::endl;
	    for ( int i = 0; i < dim+1; ++i )
	      std::cout << "gp_param = " << gp_parameters[i] << std::endl;
	    std::cout << std::endl;
		nb_gp_nodes = nodes.size();
		gp_nodes.clear();
		gp_noise.clear();
		gp_nodes_eigen.resize(nb_gp_nodes, dim);
		gp_noise_eigen.resize(nb_gp_nodes);
		for (int i = 0; i < nb_gp_nodes; ++i) {
			gp_nodes.push_back(nodes.at(i));
			gp_noise.push_back(noise.at(i));
			for(int j = 0; j < dim; ++j){
				gp_nodes_eigen(i,j) = nodes[i][j];	
			}
			gp_noise_eigen(i) = noise[i];
		}

		//Set up matrix K_u_f and K_f_u
		compute_Kuf_and_Kuu();
		MatrixXd K_f_u = K_u_f.transpose();

		VectorXd diag_Q_f_f;
		compute_Qff(K_f_u, diag_Q_f_f);

		VectorXd diag_K_f_f;
		compute_Kff(diag_K_f_f);

		VectorXd diff_Kff_Qff; //this is different for FITC, DTC, SoR
		compute_diff_Kff_Qff(diag_K_f_f, diag_Q_f_f, diff_Kff_Qff);

		compute_Lambda(diff_Kff_Qff, noise);

		MatrixXd Lambda_K_f_u;
		compute_Lambda_times_Kfu(K_f_u, Lambda_K_f_u);

		MatrixXd K_u_f_Lambda_f_u;
		compute_KufLambdaKfu(Lambda_K_f_u, K_u_f_Lambda_f_u);

		L_eigen.compute(K_u_u + K_u_f_Lambda_f_u);

		scaled_function_values.clear();
		scaled_function_values.resize(nb_gp_nodes);
		for (int i = 0; i < nb_gp_nodes; i++) {
			scaled_function_values.at(i) = values.at(i);
		}

		VectorXd LambdaInv_f;
		compute_LambdaInvF(LambdaInv_f);

		VectorXd alpha_eigen_rhs = K_u_f * LambdaInv_f;

		//Solve Sigma_not_inv^(-1)*alpha
		alpha_eigen = L_eigen.solve(alpha_eigen_rhs);

	} else {
		GaussianProcess::build(nodes, values, noise);
	}
	return;
}

//--------------------------------------------------------------------------------
void FullyIndependentTrainingConditional::update(std::vector<double> const &x,
		double &value, double &noise) {

	int nb_u_nodes = u.rows();
	if (nb_u_nodes > 0) {
		std::cout << "FITC update [" << gp_nodes.size()+1 << "," << nb_u_nodes <<"]" << std::endl;
		//std::cout << "#Update" << std::endl;
		std::vector<std::vector<double> > temp_nodes;
		temp_nodes.resize(gp_nodes.size());
		std::vector<double> temp_values;
		std::vector<double> temp_noise;
		for (int i = 0; i < gp_nodes.size(); ++i) {
			for (int j = 0; j < gp_nodes.at(i).size(); ++j) {
				temp_nodes.at(i).push_back(gp_nodes.at(i).at(j));
			}
			temp_values.push_back(scaled_function_values.at(i));
			temp_noise.push_back(gp_noise.at(i));
		}
		temp_nodes.push_back(x);
		temp_noise.push_back(noise);
		temp_values.push_back(value);
		//  scaled_function_values.push_back ( ( value -  min_function_value ) /
		//                                     ( 5e-1*( max_function_value-min_function_value ) ) - 1e0 );

		this->build(temp_nodes, temp_values, temp_noise);
	} else {
		GaussianProcess::update(x, value, noise);
	}
	return;
}
//--------------------------------------------------------------------------------

void FullyIndependentTrainingConditional::evaluate(std::vector<double> const &x,
		double &mean, double &variance) {
	int nb_u_nodes = u.rows();
	if (nb_u_nodes > 0) {
		//std::cout << "FITC evalute [" << gp_nodes_eigen.rows() << "," << nb_u_nodes <<"]" << std::endl;
		VectorXd x_eigen;
		x_eigen.resize(x.size());
		for(int i = 0; i < x.size(); ++i){
			x_eigen(i) = x[i];
		}
		K0_eigen.resize(nb_u_nodes);
		for (int i = 0; i < nb_u_nodes; i++) {
			K0_eigen(i) = evaluate_kernel(x_eigen, u.row(i));
		}

//		std::cout << "alpha= (K_u_u + K_u_f Lambda_inv K_f_u)\(K_u_f*LambdaInv*f):" << std::endl;
//		VectorOperations::print_vector(alpha);
		mean = K0_eigen.dot(alpha_eigen);
//		std::cout << "mean:" << mean << std::endl;
		//std::cout << "Mean: " << mean << std::endl;

		double variance_term3 = K0_eigen.dot(L_eigen.solve(K0_eigen));

		double variance_term2 = K0_eigen.dot(LLTofK_u_u.solve(K0_eigen));
		/*
		 std::cout << "Variance: " << variance << std::endl;
		 std::cout << "######################################" << std::endl;
		 evaluate_counter++;
		 assert(evaluate_counter<20*3);
//		 */
		//exit(-1);

//		std::cout << "K**:" << evaluate_kernel(x,x) << std::endl;
		variance = evaluate_kernel(x_eigen,x_eigen)-variance_term2+variance_term3;
		 //std::cout << "FITC evalute [" << gp_nodes_eigen.rows() << "," << nb_u_nodes <<"] mean,variance " << mean << ", " << variance << std::endl;
//		std::cout << "variance:" << variance << std::endl;
	} else {
		GaussianProcess::evaluate(x, mean, variance);
	}
	return;

}

void FullyIndependentTrainingConditional::get_induced_nodes(
		std::vector<std::vector<double> > &induced_nodes) const {
	induced_nodes.resize(u.rows());
	for (int i = 0; i < u.rows(); ++i) {
		for (int j = 0; j < dim; ++j){
			induced_nodes[i].push_back(u(i, j));
		}
	}

	return;
}

void FullyIndependentTrainingConditional::sample_u(const int& nb_u_nodes) {

	u.resize(nb_u_nodes, dim);

	u.setConstant(-1);
	std::vector<int> u_idx_left_over;
	std::vector<int> u_idx_from_active;
	//Set distribution for sampling the indices for the u samples
	std::random_device rd;
	int random_seed = rd();//1;////rd();
	std::mt19937 random_generator(random_seed);
	std::vector<double> nodes_weights_vector;

	u.resize(nb_u_nodes, dim);
	std::normal_distribution<double> dis_radius(0.0, 1.0);
	std::uniform_real_distribution<double> dis_unit_radius(0.0, 1.0);
	std::uniform_real_distribution<double> dis_trust_radius(-constraint_ball_radius, constraint_ball_radius);
	VectorXd cur_u(dim);
	double cur_U_i, radius, rand_radius, n_minus_1_angle;
	bool not_found_point = true;
	for(int i = 0; i < nb_u_nodes; ++i){
		not_found_point = true;
		radius = -1.0;
		while(not_found_point){
			not_found_point = true;
			radius = 0.0;
			for(int j = 0; j < dim; ++j){
				cur_u(j) = dis_trust_radius(random_generator);
				radius += cur_u(j) *cur_u(j);
			}
			radius = sqrt(radius);
			if(radius < constraint_ball_radius){
				for(int j = 0; j < dim; ++j){
					u(i, j) = cur_u(j) + constraint_ball_center(j);
				}
				not_found_point = false;
			}
		}
	}
	return;
}

void FullyIndependentTrainingConditional::clear_u(){
	u.resize(0,0);
}

void FullyIndependentTrainingConditional::copy_data_to_members( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise){
  nb_gp_nodes = nodes.size();
  gp_nodes.clear();
  gp_noise.clear();
	gp_nodes_eigen.resize(nb_gp_nodes, dim);
	gp_noise_eigen.resize(nb_gp_nodes);
  for ( int i = 0; i < nb_gp_nodes; ++i ) {
    gp_nodes.push_back ( nodes.at(i) );
    gp_noise.push_back ( noise.at(i) );
	for(int j = 0; j < dim; ++j){
		gp_nodes_eigen(i,j) = nodes[i][j];	
	}
	gp_noise_eigen(i) = noise[i];
}

//  auto minmax = std::minmax_element(values.begin(), values.end());
//  min_function_value = values.at((minmax.first - values.begin()));
//  max_function_value = values.at((minmax.second - values.begin()));

  L.clear();
  L.resize( nb_gp_nodes );
  for ( int i = 0; i < nb_gp_nodes; ++i)
    L.at(i).resize( i+1 );

  scaled_function_values.resize(nb_gp_nodes);
  scaled_function_values_eigen.resize(nb_gp_nodes);
  for ( int i = 0; i < nb_gp_nodes; ++i) {
    scaled_function_values.at(i) = values.at(i);
    scaled_function_values_eigen(i) = values.at(i);
//    scaled_function_values.at(i) = values.at(i) - min_function_value;
//    scaled_function_values.at(i) /= 5e-1*( max_function_value-min_function_value );
//    scaled_function_values.at(i) -= 1e0;
  }

}

void FullyIndependentTrainingConditional::set_optimizer(std::vector<double> const &values, nlopt::opt*& local_opt, nlopt::opt*& global_opt){

  optimize_global = true;
  optimize_local = false;

  int dimp1 = gp_parameters_hp.size();

  lb.resize(dimp1);
  ub.resize(dimp1);
  
  int offset;
  if(dimp1==1 + dim){
  		auto minmax = std::minmax_element(values.begin(), values.end());
	    min_function_value = values.at((minmax.first - values.begin()));
	    max_function_value = fabs(values.at((minmax.second - values.begin())));
	    if ( fabs(min_function_value) > max_function_value )
	    max_function_value = fabs( min_function_value );
	  	
		  lb[0] = 1e-3; 
		  ub[0] = 1e3;
		  lb[0] = max_function_value - 1e2;
		  if ( lb[0] < 1e-3 ) lb[0] = 1e-3;
		  ub[0] = max_function_value + 1e2; 
		  if ( ub[0] > 1e3 ) ub[0] = 1e3;
		  if ( ub[0] <= lb[0]) lb[0] = 1e-3;
		//if (ub[0] < 1.0) ub[0] = 1.0;
		double delta_threshold = *delta;
		if (delta_threshold < 1e-2) delta_threshold = 1e-2;
		for (int i = 0; i < dim; ++i) {
		    lb[i+1] = 1e-2 * delta_threshold; // 1e1
		    ub[i+1] = 2.0 * delta_threshold; // 1e2
		}
	  	for (int i = 0; i < dim+1; ++i) {
  		  if ( gp_parameters_hp[i] <= lb[i] ) {
  			std::cout << "LS Too small: " << gp_parameters_hp[i] << " for " << lb[i] << std::endl;
      		gp_parameters_hp[i] = 1.0001 * lb[i];
      		}
	      if ( gp_parameters_hp[i] >= ub[i] ) {
	  			std::cout << "LS Too big: " << gp_parameters_hp[i] << " for " << ub[i] << std::endl;
	      		gp_parameters_hp[i] = 0.9999 * ub[i];
	      }
	      if ( gp_parameters_hp[i] <= lb[i] ||  gp_parameters_hp[i] >= ub[i]){
			std::cout << "LS still in between: " << gp_parameters_hp[i];
	      	gp_parameters_hp[i] = lb[i] + (ub[i]-lb[i])/0.5;
	      	std::cout << "LS fixed: "<< gp_parameters_hp[i] << std::endl;
        	}
        }
	  	
  }else{
	  if(gp_parameters_hp.size() > u.rows()*dim){ //optimizing also over lengthscale and sigma_f
	  	auto minmax = std::minmax_element(values.begin(), values.end());
	    min_function_value = values.at((minmax.first - values.begin()));
	    max_function_value = fabs(values.at((minmax.second - values.begin())));
	    if ( fabs(min_function_value) > max_function_value )
	    max_function_value = fabs( min_function_value );
	  	lb[0] = 1e-3; 
		  ub[0] = 1e3;
		  lb[0] = max_function_value - 1e2;
		  if ( lb[0] < 1e-3 ) lb[0] = 1e-3;
		  ub[0] = max_function_value + 1e2; 
		  if ( ub[0] > 1e3 ) ub[0] = 1e3;
		  if ( ub[0] <= lb[0]) lb[0] = 1e-3;
		//if (ub[0] < 1.0) ub[0] = 1.0;
		double delta_threshold = *delta;
		if (delta_threshold < 1e-2) delta_threshold = 1e-2;
		for (int i = 0; i < dim; ++i) {
		    lb[i+1] = 1e-2 * delta_threshold; // 1e1
		    ub[i+1] = 2.0 * delta_threshold; // 1e2
		}
	    offset = 1+dim;
	  }else{
	  	offset = 0;
	  }
	  //Set box constraints such that the constraint ball is inside
	  std::vector<double> lb_u(dim);
	  std::vector<double> ub_u(dim);
	  for (int i = 0; i < dim; ++i) {
	      lb_u[i] = constraint_ball_center[i] - 1.5*constraint_ball_radius;
	      ub_u[i] = constraint_ball_center[i] + 1.5*constraint_ball_radius;
	  }
	  for (int i = 0; i < dim; ++i) {
		  for(int j = offset + i*u.rows(); j < offset + (i+1)*u.rows(); ++j){
	          lb[j] = lb_u[i];
	          ub[j] = ub_u[i];
		  }
	  }

	  
	  if(gp_parameters_hp.size() > u.rows()*dim){//optimizing also over lengthscale and sigma_f
		  if (gp_parameters_hp[0] < 0e0) {
		  	std::cout << "In here: " << gp_parameters_hp[0] << std::endl;
		    gp_parameters_hp[0] = max_function_value;
		    for (int i = 1; i < dim+1; ++i) {
		      gp_parameters_hp[i] = (lb[i]*5e-1 + 5e-1*ub[i]);
		    }
		  } else {
		    for (int i = 0; i < dim+1; ++i) {
		      if ( gp_parameters_hp[i] <= lb[i] ) {
		  			std::cout << "2Too small: " << gp_parameters_hp[i] << std::endl;
		      		gp_parameters_hp[i] = 1.0001 * lb[i];
		      	}
		      if ( gp_parameters_hp[i] >= ub[i] ) {
		  			std::cout << "2Too big: " << gp_parameters_hp[i] << std::endl;
		      		gp_parameters_hp[i] = 0.9999 * ub[i];
		      }
		      if ( gp_parameters_hp[i] <= lb[i] ||  gp_parameters_hp[i] >= ub[i]){
				std::cout << "2still in between: " << gp_parameters_hp[i];
		      	gp_parameters_hp[i] = lb[i] + (ub[i]-lb[i])/0.5;
		      	std::cout << "2fixed: "<< gp_parameters_hp[i] << std::endl;
		      }
		    }
		  }
		}
		
	for (int i = offset; i < dimp1; ++i) {
      if ( gp_parameters_hp[i] <= lb[i] ) {
  			std::cout << "U Too small: " << gp_parameters_hp[i] << std::endl;
      		gp_parameters_hp[i] = 1.0001 * lb[i];
      	}
      if ( gp_parameters_hp[i] >= ub[i] ) {
  			std::cout << "U Too big: " << gp_parameters_hp[i] << std::endl;
      		gp_parameters_hp[i] = 0.9999 * ub[i];
      }
      if ( gp_parameters_hp[i] <= lb[i] ||  gp_parameters_hp[i] >= ub[i]){
		std::cout << "U still in between: " << gp_parameters_hp[i];
      	gp_parameters_hp[i] = lb[i] + (ub[i]-lb[i])/0.5;
      	std::cout << "U fixed: "<< gp_parameters_hp[i] << std::endl;
      }
    	}
	}
  

  local_opt = new nlopt::opt(nlopt::LD_MMA, dimp1);
  global_opt = new nlopt::opt(nlopt::GN_ISRES, dimp1);

  global_opt->set_lower_bounds( lb );
  global_opt->set_upper_bounds( ub );
  global_opt->set_maxtime(1.0);
  //global_opt->set_maxeval(10000);

  local_opt->set_lower_bounds( lb );
  local_opt->set_upper_bounds( ub );
  local_opt->set_maxtime(60.0);
  local_opt->set_maxeval(1000);
}

void FullyIndependentTrainingConditional::run_optimizer(std::vector<double> const &values){
	double optval;

  int exitflag;

  nlopt::opt* local_opt;
  nlopt::opt* global_opt;

  set_optimizer(values, local_opt, global_opt);

  int dimp1 = gp_parameters_hp.size();

  std::vector<double> tol(dimp1);
  for(int i = 0; i < dimp1; ++i){
  	tol[i] = 0.0;
  }
  if (optimize_global){
  	  print = 0;
 	  std::cout << "Global optimization" << std::endl;
	  exitflag=-20;
	  global_opt->add_inequality_mconstraint(trust_region_constraint, gp_pointer, tol);
	  global_opt->set_min_objective( parameter_estimation_objective, gp_pointer);
	  exitflag = global_opt->optimize(gp_parameters_hp, optval);

	  std::cout << "exitflag = "<< exitflag<<std::endl;
  	  std::cout << "Function calls: " << print << std::endl;
	  std::cout << "OPTVAL .... " << optval << std::endl;
	  //for ( int i = 0; i < 1+dim; ++i )
	    //std::cout << "gp_param = " << gp_parameters_hp[i] << std::endl;
	  //std::cout << std::endl;
  }
  if (optimize_local){
  	  print = 0;
  	  std::cout << "Local optimization" << std::endl;
	  exitflag=-20;
	  //try {
	  local_opt->add_inequality_mconstraint(trust_region_constraint, gp_pointer, tol);
	  local_opt->set_min_objective( parameter_estimation_objective_w_gradients, gp_pointer);
	  exitflag = local_opt->optimize(gp_parameters_hp, optval);

	  std::cout << "exitflag = "<< exitflag<<std::endl;
  	  std::cout << "Function calls: " << print << std::endl;
	  std::cout << "OPTVAL .... " << optval << std::endl;
	  //for ( int i = 0; i < 1+dim; ++i )
	    //std::cout << "gp_param = " << gp_parameters_hp[i] << std::endl;
	  //std::cout << std::endl;
  }
  
  delete local_opt;
  delete global_opt;

  return;
}

void FullyIndependentTrainingConditional::update_induced_points(){
  int u_counter;
  int offset = 1+dim;
  for (int i = 0; i < dim; ++i) {
  	  u_counter = 0;
  	  for(int j = offset + i*u.rows(); j < offset + (i+1)*u.rows(); ++j){
            u(u_counter,i) = gp_parameters[j];
            u_counter++;
  	  }
  }
}

void FullyIndependentTrainingConditional::estimate_hyper_parameters ( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise ){
	this->estimate_hyper_parameters_ls_only(nodes, values, noise);
}

void FullyIndependentTrainingConditional::estimate_hyper_parameters_all ( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise )
{
  if (u.rows() > 0) {	
	  std::cout << "FITC Estimator" << std::endl;
	  copy_data_to_members(nodes, values, noise);

	  gp_pointer = this;

	  sample_u(u.rows());

	  set_hyperparameters();

	  run_optimizer(values);

	  copy_hyperparameters();

	  for ( int i = 0; i < 1+dim; ++i )
	    std::cout << "gp_param = " << gp_parameters[i] << std::endl;
  }else{
		GaussianProcess::estimate_hyper_parameters(nodes, values, noise);
  }

  return;
}

void FullyIndependentTrainingConditional::estimate_hyper_parameters_induced_only ( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise )
{
  if (u.rows() > 0) {	
	  std::cout << "FITC Estimator" << std::endl;
	  copy_data_to_members(nodes, values, noise);

	  gp_pointer = this;

	  sample_u(u.rows());

	  set_hyperparameters_induced_only();

	  run_optimizer(values);

	  copy_hyperparameters();

	  for ( int i = 0; i < 1+dim; ++i )
	    std::cout << "gp_param = " << gp_parameters[i] << std::endl;

  }else{
		GaussianProcess::estimate_hyper_parameters(nodes, values, noise);
  }

  return;
}
void FullyIndependentTrainingConditional::estimate_hyper_parameters_ls_only ( std::vector< std::vector<double> > const &nodes,
                                                      std::vector<double> const &values,
                                                      std::vector<double> const &noise ){
  if (u.rows() > 0) {	
	  std::cout << "FITC Estimator" << std::endl;
	  copy_data_to_members(nodes, values, noise);

	  gp_pointer = this;

	  sample_u(u.rows());

	  set_hyperparameters_ls_only();

	  run_optimizer(values);

	  copy_hyperparameters();

	  for ( int i = 0; i < 1+dim; ++i )
	    std::cout << "gp_param = " << gp_parameters[i] << std::endl;

  }else{
		GaussianProcess::estimate_hyper_parameters(nodes, values, noise);
  }

  return;
}
//--------------------------------------------------------------------------------

void FullyIndependentTrainingConditional::set_hyperparameters(){
  int dimp1 = 1+dim+u.rows()*dim;	
  gp_parameters_hp.resize(dimp1);
  int u_counter;
  int offset = dim + 1;
  for (int i = 0; i < dim + 1; ++i){
  	gp_parameters_hp[i] = gp_parameters[i];
  }
  for (int i = 0; i < dim; ++i) {
  	  u_counter = 0;
  	  for(int j = offset + i*u.rows(); j < offset + (i+1)*u.rows(); ++j){
            gp_parameters_hp[j] = u(u_counter,i);    		
            u_counter++;
  	  }
  }
  return;
}

void FullyIndependentTrainingConditional::set_hyperparameters_ls_only(){
  int dimp1 = dim+1;	
  gp_parameters_hp.resize(dimp1);
  for (int i = 0; i < dimp1; ++i){
  	gp_parameters_hp[i] = gp_parameters[i];
  }
  return;
}

void FullyIndependentTrainingConditional::set_hyperparameters_induced_only(){
  int dimp1 = u.rows()*dim;	
  gp_parameters_hp.resize(dimp1);
  int u_counter;
  for (int i = 0; i < dim; ++i) {
  	  u_counter = 0;
  	  for(int j = i*u.rows(); j < (i+1)*u.rows(); ++j){
            gp_parameters_hp[j] = u(u_counter,i);    		
            u_counter++;
  	  }
  }
  return;
}

void FullyIndependentTrainingConditional::copy_hyperparameters(){
  int dimp1 = gp_parameters_hp.size();
  int u_counter;
  int offset;
  if(gp_parameters_hp.size()==dim+1){
	for (int i = 0; i < dim + 1; ++i){
  	  gp_parameters[i] = gp_parameters_hp[i];
    }
  }else{
	  if(gp_parameters_hp.size() > u.rows()*dim){//optimizing also over lengthscale and sigma_f, copy them back
	    for (int i = 0; i < dim + 1; ++i){
	  	  gp_parameters[i] = gp_parameters_hp[i];
	    }
	    offset = 1+dim;
	  }else{
	  	offset = 0;
	  }
	  for (int i = 0; i < dim; ++i) {
	  	  u_counter = 0;
	  	  for(int j = offset + i*u.rows(); j < offset + (i+1)*u.rows(); ++j){
	        u(u_counter,i) = gp_parameters_hp[j];
	        u_counter++;
	  	  }
	  }
  }
  return;
}

//--------------------------------------------------------------------------------
double FullyIndependentTrainingConditional::parameter_estimation_objective(std::vector<double> const &x,
                                                       std::vector<double> &grad,
                                                       void *data)
{
  FullyIndependentTrainingConditional *d = reinterpret_cast<FullyIndependentTrainingConditional*>(data);
  int offset;
  std::vector<double> local_params(d->dim+1);
  if(x.size()==1+d->dim){
  	  offset = 0;
  	  for(int i = 0; i < local_params.size(); ++i){
	  		local_params[i] = x[i];
  	  }
  }else{
	  if(x.size() > d->u.rows()*d->dim){
	  	offset = 1+d->dim;
	  	for(int i = 0; i < local_params.size(); ++i){
	  		local_params[i] = x[i];
	  	}
	  }else{
	  	offset = 0;
	  	for(int i = 0; i < local_params.size(); ++i){
	  		local_params[i] = d->gp_parameters[i];
	  	}
	  }
	  int u_counter;
	  for (int i = 0; i < d->dim; ++i) {
	  	  u_counter = 0;
	  	  for(int j = offset + i*d->u.rows(); j < offset + (i+1)*d->u.rows(); ++j){
	            d->u(u_counter,i) = x[j];
	            u_counter++;
	  	  }
	  }
  }

  //Compute Kuf, Kuu
  //Set up matrix K_u_f and K_f_u
	int nb_gp_nodes = d->gp_nodes.size();
	int nb_u_nodes = d->u.rows();
	d->K_u_f.resize(nb_u_nodes, nb_gp_nodes);
	//std::cout << d->u.rows() << " " << nb_gp_nodes << " " << d->gp_nodes_eigen.rows() << std::endl;
	for (int i = 0; i < nb_u_nodes; ++i) {
		for (int j = 0; j < nb_gp_nodes; ++j) {
			d->K_u_f(i,j) = d->evaluate_kernel(d->u.row(i), d->gp_nodes_eigen.row(j), local_params);
		}
	}
	MatrixXd K_f_u = d->K_u_f.transpose();
	//std::cout << "Kuf\n" << d->K_u_f << std::endl;

	//Set up matrix K_u_u
	d->K_u_u.resize(nb_u_nodes, nb_u_nodes);
	for (int i = 0; i < nb_u_nodes; ++i) {
		for (int j = 0; j < nb_u_nodes; ++j) {
			d->K_u_u(i,j) = d->evaluate_kernel(d->u.row(i), d->u.row(j), local_params);
			if(i==j)
				d->K_u_u(i,j) += d->Kuu_opt_nugget;
		}
	}
	//std::cout << "Kuu\n" << d->K_u_u << std::endl;
	d->LLTofK_u_u.compute(d->K_u_u);
	MatrixXd K_u_u_u_f = d->LLTofK_u_u.solve(d->K_u_f);

	VectorXd diag_Q_f_f;
	diag_Q_f_f.resize(nb_gp_nodes);

	for (int i = 0; i < nb_gp_nodes; ++i) {
		diag_Q_f_f(i) = 0.0;
		for (int j = 0; j < nb_u_nodes; ++j) {
			diag_Q_f_f(i) += (K_f_u(i,j) * K_u_u_u_f(j,i));
		}
	}
	VectorXd diag_K_f_f;
	diag_K_f_f.resize(nb_gp_nodes);
	for (int i = 0; i < nb_gp_nodes; ++i) {
		diag_K_f_f(i) = d->evaluate_kernel(d->gp_nodes_eigen.row(i),
									 d->gp_nodes_eigen.row(i), local_params);
	}

	d->Lambda.resize(nb_gp_nodes);
	//std::cout << "noise\n" << std::endl;
	for (int i = 0; i < nb_gp_nodes; ++i) {
		d->Lambda(i) = (diag_K_f_f(i) - diag_Q_f_f(i) + pow( d->gp_noise.at(i) / 2e0 + d->noise_regularization, 2e0 ));
	}
	//std::cout << std::endl;

	MatrixXd Lambda_K_f_u;
	Lambda_K_f_u.resize(nb_gp_nodes, nb_u_nodes);
	for(int i = 0; i < nb_gp_nodes; i++){
		for(int j = 0; j < nb_u_nodes; j++){
			Lambda_K_f_u(i,j) = ((1.0/(d->Lambda(i) + d->Lambda_opt_nugget)) * K_f_u(i,j));
		}
	}
	MatrixXd K_u_f_Lambda_f_u;
	K_u_f_Lambda_f_u.resize(nb_u_nodes, nb_u_nodes);
	K_u_f_Lambda_f_u = d->K_u_f*Lambda_K_f_u;

	MatrixXd Sigma = d->K_u_u + K_u_f_Lambda_f_u;
	for (int i = 0; i < nb_u_nodes; ++i) {
		//Sigma(i,i) += nugget;
	}
	d->L_eigen.compute(Sigma);
	double L12 = 0.0;//-log(d->K_u_u.determinant());
	for (int i = 0; i < d->u.rows(); ++i){
		L12 += log(d->LLTofK_u_u.vectorD()(i));
    	//std::cout << L1 << "L11-" << i << std::endl;
	}

	double det_Leigen = 0.0;
	for (int i = 0; i < d->u.rows(); ++i){
		det_Leigen += log(d->L_eigen.vectorD()(i));
    	//std::cout << d->L_eigen.matrixL()(i,i) << " " << log(d->L_eigen.matrixL()(i,i)) << "det_Leigen-" << i << std::endl;
	}

	double L11 = 0.0;
    //std::cout << L1 << "L12 " << std::endl;
	for (int i = 0; i < d->nb_gp_nodes; ++i){
		L11 += log(d->Lambda(i));
		//std::cout << "Lambda: " << i << " " << d->Lambda(i) <<" "<<log(d->Lambda(i)) << std::endl;
	}
    double L1 = 0.0;
	//L1 = 0.5*L11 + (2*0.5)*det_Leigen + 0.5*det_LeigenD - (2*0.5)*L12 - 0.5*L12D;
	L1 = 0.5*L11 + 0.5*det_Leigen - 0.5*L12;

	MatrixXd Q_f_f = K_f_u*K_u_u_u_f;
	for(int i = 0; i < nb_gp_nodes; i++){
		Q_f_f(i,i) += d->Lambda(i)+ d->Qff_opt_nugget;
	}
	LLT<MatrixXd> LLTofQ_f_f(Q_f_f);
	double L2 = 0.5*d->scaled_function_values_eigen.dot(LLTofQ_f_f.solve(d->scaled_function_values_eigen));
  	
  	//std::cout << d->scaled_function_values_eigen << std::endl;
  double result = L1 + L2;
 
  if (std::isinf(result) || std::isnan(result)){
  	std::cout << "Result is inf or nan" << std::endl;
  	std::cout << "L11 " << L11 << " " << 'x' << std::endl;
    std::cout << "L12 " << L12 << std::endl; //<< " " << log(d->K_u_u.determinant()) << std::endl;
    std::cout << "L13 " << det_Leigen << std::endl;// << " " << log(Sigma.determinant()) << std::endl;
    std::cout << L1 << ' ' << L2 << std::endl;
  	result = std::numeric_limits<double>::infinity();
  	if(d->Kuu_opt_nugget < d->nugget_max){
  		d->Kuu_opt_nugget *= 10;
  		d->Lambda_opt_nugget *= 10;
  		d->Qff_opt_nugget *= 10;
  	}
  }else{
  	if(d->Kuu_opt_nugget > d->nugget_min){
  		d->Kuu_opt_nugget *= 0.1;
  		d->Lambda_opt_nugget *= 0.1;
  		d->Qff_opt_nugget *= 0.1;
  	}
  }
  if ((d->print%1000)==0){
	  //for ( int i = 0; i < d->dim + 1; ++i )
	  //  std::cout << "gp_param = " << x[i] << std::endl;
	  //for(int j = offset; j < offset + d->u.rows(); ++j)
	//		std::cout << "gp_param = " << x[j] <<","<<x[j+ d->u.rows()]<< std::endl;
  	
  	//std::cout << d->print <<" Objective: " << L1 << " " << L2 << " "<< result<< std::endl;
   }

  d->print++;
  return result;

}

double FullyIndependentTrainingConditional::parameter_estimation_objective_w_gradients(std::vector<double> const &x,
                                                       std::vector<double> &grad,
                                                       void *data)
{

  FullyIndependentTrainingConditional *d = reinterpret_cast<FullyIndependentTrainingConditional*>(data);
  int offset;
  std::vector<double> local_params(d->dim+1);
  if(x.size()==1+d->dim){
  	  offset = 0;
  	  for(int i = 0; i < local_params.size(); ++i){
	  		local_params[i] = x[i];
  	  }
  }else{
	  if(x.size() > d->u.rows()*d->dim){
	  	offset = 1+d->dim;
	  	for(int i = 0; i < local_params.size(); ++i){
	  		local_params[i] = x[i];
	  	}
	  }else{
	  	offset = 0;
	  	for(int i = 0; i < local_params.size(); ++i){
	  		local_params[i] = d->gp_parameters[i];
	  	}
	  }
	  int u_counter;
	  for (int i = 0; i < d->dim; ++i) {
	  	  u_counter = 0;
	  	  for(int j = offset + i*d->u.rows(); j < offset + (i+1)*d->u.rows(); ++j){
	            d->u(u_counter,i) = x[j];
	            u_counter++;
	  	  }
	  }
  }

  //Compute Kuf, Kuu
  //Set up matrix K_u_f and K_f_u
	int nb_gp_nodes = d->gp_nodes.size();
	int nb_u_nodes = d->u.rows();
	d->K_u_f.resize(nb_u_nodes, nb_gp_nodes);
	for (int i = 0; i < nb_u_nodes; ++i) {
		for (int j = 0; j < nb_gp_nodes; ++j) {
			d->K_u_f(i,j) = d->evaluate_kernel(d->u.row(i), d->gp_nodes_eigen.row(j), local_params);
		}
	}
	MatrixXd K_f_u = d->K_u_f.transpose();
	//std::cout << "Kuf\n" << d->K_u_f << std::endl;

	//Set up matrix K_u_u
	d->K_u_u.resize(nb_u_nodes, nb_u_nodes);
	for (int i = 0; i < nb_u_nodes; ++i) {
		for (int j = 0; j < nb_u_nodes; ++j) {
			d->K_u_u(i,j) = d->evaluate_kernel(d->u.row(i), d->u.row(j), local_params);
			if(i==j)
				d->K_u_u(i,j) += d->Kuu_opt_nugget;
		}
	}
	//std::cout << "Kuu\n" << d->K_u_u << std::endl;
	d->LLTofK_u_u.compute(d->K_u_u);
	MatrixXd K_u_u_u_f = d->LLTofK_u_u.solve(d->K_u_f);

	VectorXd diag_Q_f_f;
	diag_Q_f_f.resize(nb_gp_nodes);

	for (int i = 0; i < nb_gp_nodes; ++i) {
		diag_Q_f_f(i) = 0.0;
		for (int j = 0; j < nb_u_nodes; ++j) {
			diag_Q_f_f(i) += (K_f_u(i,j) * K_u_u_u_f(j,i));
		}
	}
	VectorXd diag_K_f_f;
	diag_K_f_f.resize(nb_gp_nodes);
	for (int i = 0; i < nb_gp_nodes; ++i) {
		diag_K_f_f(i) = d->evaluate_kernel(d->gp_nodes_eigen.row(i),
									 d->gp_nodes_eigen.row(i), local_params);
	}
	/*std::cout << "diag_K_f_f\n" << diag_K_f_f << std::endl;
	std::cout << "diag_Q_f_f\n" << diag_Q_f_f << std::endl;*/

	d->Lambda.resize(nb_gp_nodes);
	//std::cout << "noise\n" << std::endl;
	for (int i = 0; i < nb_gp_nodes; ++i) {
		d->Lambda(i) = (diag_K_f_f(i) - diag_Q_f_f(i) + pow( d->gp_noise.at(i) / 2e0 + d->noise_regularization, 2e0 ));
	}
	//std::cout << std::endl;

	MatrixXd Lambda_K_f_u;
	Lambda_K_f_u.resize(nb_gp_nodes, nb_u_nodes);
	for(int i = 0; i < nb_gp_nodes; i++){
		for(int j = 0; j < nb_u_nodes; j++){
			Lambda_K_f_u(i,j) = ((1.0/(d->Lambda(i) + d->Lambda_opt_nugget)) * K_f_u(i,j));
		}
	}
	MatrixXd K_u_f_Lambda_f_u;
	K_u_f_Lambda_f_u.resize(nb_u_nodes, nb_u_nodes);
	K_u_f_Lambda_f_u = d->K_u_f*Lambda_K_f_u;

	MatrixXd Sigma = d->K_u_u + K_u_f_Lambda_f_u;
	for (int i = 0; i < nb_u_nodes; ++i) {
		//Sigma(i,i) += nugget;
	}
	d->L_eigen.compute(Sigma);
	double L12 = 0.0;//-log(d->K_u_u.determinant());
	for (int i = 0; i < d->u.rows(); ++i){
		L12 += log(d->LLTofK_u_u.vectorD()(i));
    	//std::cout << L1 << "L11-" << i << std::endl;
	}

	double det_Leigen = 0.0;
	for (int i = 0; i < d->u.rows(); ++i){
		det_Leigen += log(d->L_eigen.vectorD()(i));
    	//std::cout << d->L_eigen.matrixL()(i,i) << " " << log(d->L_eigen.matrixL()(i,i)) << "det_Leigen-" << i << std::endl;
	}

	double L11 = 0.0;
    //std::cout << L1 << "L12 " << std::endl;
	for (int i = 0; i < d->nb_gp_nodes; ++i){
		L11 += log(d->Lambda(i));
		//std::cout << "Lambda: " << i << " " << d->Lambda(i) <<" "<<log(d->Lambda(i)) << std::endl;
	}
    double L1 = 0.0;
	//L1 = 0.5*L11 + (2*0.5)*det_Leigen + 0.5*det_LeigenD - (2*0.5)*L12 - 0.5*L12D;
	L1 = 0.5*L11 + 0.5*det_Leigen - 0.5*L12;
	MatrixXd Q_f_f = K_f_u*K_u_u_u_f;
	for(int i = 0; i < nb_gp_nodes; i++){
		Q_f_f(i,i) += d->Lambda(i)+ d->Qff_opt_nugget;
	}
	LLT<MatrixXd> LLTofQ_f_f(Q_f_f);
	double L2 = 0.5*d->scaled_function_values_eigen.dot(LLTofQ_f_f.solve(d->scaled_function_values_eigen));
  	
  	//std::cout << d->scaled_function_values_eigen << std::endl;
  double result = L1 + L2;
 
  if (std::isinf(result) || std::isnan(result)){
  	std::cout << "Result is inf or nan" << std::endl;
  	std::cout << "L11 " << L11 << " " << 'x' << std::endl;
    std::cout << "L12 " << L12 << std::endl; //<< " " << log(d->K_u_u.determinant()) << std::endl;
    std::cout << "L13 " << det_Leigen << std::endl;// << " " << log(Sigma.determinant()) << std::endl;
    std::cout << L1 << ' ' << L2 << std::endl;
  	result = std::numeric_limits<double>::infinity();
  	if(d->Kuu_opt_nugget < d->nugget_max){
  		d->Kuu_opt_nugget *= 10;
  		d->Lambda_opt_nugget *= 10;
  		d->Qff_opt_nugget *= 10;
  	}
  }else{
  	if(d->Kuu_opt_nugget > d->nugget_min){
  		d->Kuu_opt_nugget *= 0.1;
  		d->Lambda_opt_nugget *= 0.1;
  		d->Qff_opt_nugget *= 0.1;
  	}
  }
  if ((d->print%1)==0){
	  /*for ( int i = 0; i < d->dim + 1; ++i )
	    std::cout << "gp_param = " << x[i] << std::endl;
	  for(int j = offset; j < offset + d->u.rows(); ++j)
			std::cout << "gp_param = " << x[j] <<","<<x[j+ d->u.rows()]<< std::endl;*/
  	
  	//std::cout << d->print <<" Objective: " << L1 << " " << L2 << " "<< result<< std::endl;
  }

	//Gradient computation:
	int dim_grad = x.size();
	grad.resize(dim_grad);

	for(int i = 0; i < grad.size(); i++){

		MatrixXd Kffdot;
		MatrixXd Kufdot;
		MatrixXd Kfudot;
		MatrixXd Kuudot;
		VectorXd LambdaDot;
		if(dim_grad == d->dim + 1){
			if(i == 0){//grad sigma_f
				d->derivate_K_u_u_wrt_sigmaf(local_params, Kuudot);
				d->derivate_K_f_f_wrt_sigmaf(local_params, Kffdot);
				d->derivate_K_u_f_wrt_sigmaf(local_params, Kufdot);
				Kfudot = Kufdot.transpose();
			}else{//grad length parameter
				d->derivate_K_u_u_wrt_l(local_params, i-1, Kuudot);
				d->derivate_K_f_f_wrt_l(local_params, i-1, Kffdot);
				d->derivate_K_u_f_wrt_l(local_params, i-1, Kufdot);
				Kfudot = Kufdot.transpose();
			}
		}else{
			if( dim_grad > d->u.rows()*d->dim){//if we optimize also for sigma_f and lengthscales
				if(i == 0){//grad sigma_f
					d->derivate_K_u_u_wrt_sigmaf(local_params, Kuudot);
					d->derivate_K_f_f_wrt_sigmaf(local_params, Kffdot);
					d->derivate_K_u_f_wrt_sigmaf(local_params, Kufdot);
					Kfudot = Kufdot.transpose();
				}else if(i > 0 && i < 1 + d->dim){//grad length parameter
					d->derivate_K_u_u_wrt_l(local_params, i-1, Kuudot);
					d->derivate_K_f_f_wrt_l(local_params, i-1, Kffdot);
					d->derivate_K_u_f_wrt_l(local_params, i-1, Kufdot);
					Kfudot = Kufdot.transpose();
				}else{//grad u
					int uidx = (i-offset)%d->u.rows();
					int udim = (int) ((i-offset)/d->u.rows());
					d->derivate_K_u_u_wrt_uik(local_params, uidx, udim, Kuudot);
					d->derivate_K_u_f_wrt_uik(local_params, uidx, udim, Kufdot);
					Kffdot.resize(nb_gp_nodes, nb_gp_nodes);
					Kffdot.setZero();
					Kfudot = Kufdot.transpose(); 
				}
		    }else{
		    	int uidx = (i-offset)%d->u.rows();
				int udim = (int) ((i-offset)/d->u.rows());
				d->derivate_K_u_u_wrt_uik(local_params, uidx, udim, Kuudot);
				d->derivate_K_u_f_wrt_uik(local_params, uidx, udim, Kufdot);
				Kffdot.resize(nb_gp_nodes, nb_gp_nodes);
				Kffdot.setZero();
				Kfudot = Kufdot.transpose(); 
		    }
	    }
		d->compute_LambdaDot(Kffdot, Kufdot, Kfudot, Kuudot, LambdaDot);

		//L1-term: dL3/dtheta = d(Lambda+noise^2)/dtheta = tr((Lambda+noise^2)^(-1)*LambdaDot)
		double dL1 = 0.0;
		VectorXd LambdaInv(nb_gp_nodes);
		for(int j = 0; j < nb_gp_nodes; ++j){
			LambdaInv(j) = 1.0/d->Lambda(j);
			dL1 += LambdaInv(j) * LambdaDot(j);
		}

		//L2-term: dL2/dtheta= tr(Kuu^(-1) * Kuudot)
		double dL2 = d->LLTofK_u_u.solve(Kuudot).trace();

		//L3-term: d(log(det(Kuu+Kuf*(Lambda+noise^2)^(-1)*Kfu)))/dtheta
		MatrixXd LambdaInvDiag = LambdaInv.asDiagonal();
		MatrixXd LambdaDotDiag = LambdaDot.asDiagonal();
		MatrixXd dSigma = Kuudot 
							+ 2*Kufdot*LambdaInvDiag*K_f_u 
							- d->K_u_f*LambdaInvDiag*LambdaDotDiag*LambdaInvDiag*K_f_u; 
							//+ d->K_u_f*LambdaInvDiag*Kfudot;
		double dL3 = (d->L_eigen.solve(dSigma)).trace();

		//L4-term: d(yT*Q*y)/dtheta
		MatrixXd KfudotKuinvKuf = Kfudot * d->LLTofK_u_u.solve(d->K_u_f);
		MatrixXd KfuKuinvKufdot = K_f_u * d->LLTofK_u_u.solve(Kufdot);
		MatrixXd KfuKuinfKudotKuinvKuf = K_f_u*(d->LLTofK_u_u.solve(Kuudot*(d->LLTofK_u_u.solve(d->K_u_f))));
		MatrixXd dQ = KfudotKuinvKuf - KfuKuinfKudotKuinvKuf + KfuKuinvKufdot + LambdaDotDiag;
		double dL4 = (-1)*d->scaled_function_values_eigen.dot(
						LLTofQ_f_f.solve(dQ*LLTofQ_f_f.solve(
							d->scaled_function_values_eigen)));

		grad[i] = 0.5*(dL1 - dL2 + dL3 + dL4);


		/*JacobiSVD<MatrixXd> svdKuuInv(d->K_u_u.inverse());
		double condKuu = svdKuuInv.singularValues()(0) 
		    / svdKuuInv.singularValues()(svdKuuInv.singularValues().size()-1);
		JacobiSVD<MatrixXd> svdQ(Q_f_f.inverse());
		double condQff = svdQ.singularValues()(0) 
		    / svdQ.singularValues()(svdQ.singularValues().size()-1);
		JacobiSVD<MatrixXd> svdLambdaInv(LambdaInvDiag);
		double condLambda = svdLambdaInv.singularValues()(0) 
		    / svdLambdaInv.singularValues()(svdLambdaInv.singularValues().size()-1);
		JacobiSVD<MatrixXd> svddQ(dQ);
		double conddQ = svddQ.singularValues()(0) 
		    / svddQ.singularValues()(svddQ.singularValues().size()-1);
		std::cout << "Condition numbers: \n";
		std::cout << "Kuuinv: " << condKuu << " Qff: " << condQff 
			<< " LambdaInv: " << condLambda << " dQ: " << conddQ << std::endl;
		*/
		//if ((d->print%1)==0){
		// 	std::cout << "Grad: " << i << "|" << 0.5*dL1 << " + "<< 0.5*dL2 << " + "<< 0.5*dL3 << " + "<< 0.5*dL4 << " = " << grad[i] << std::endl;
		//}
		//exit(-1);
	}

  d->print++;

  return result;

}

void FullyIndependentTrainingConditional::set_constraint_ball_radius(const double& radius){
	constraint_ball_radius = radius;
}

void FullyIndependentTrainingConditional::set_constraint_ball_center(const std::vector<double>& center){
	constraint_ball_center.resize(center.size());
	for(int i = 0; i < center.size(); ++i){
		constraint_ball_center(i) = center[i];
	}
}

void FullyIndependentTrainingConditional::trust_region_constraint(unsigned int m, double* c, unsigned int n, const double* x, double* grad,
                                                     			void *data){
	FullyIndependentTrainingConditional *d = reinterpret_cast<FullyIndependentTrainingConditional*>(data);

	int offset;
	if(m == d->dim + 1){
		for (int i = 0; i < d->dim+1; ++i) {
			c[i] = -1;
		}
	}else{
		if(m > d->u.rows()*d->dim){
		    offset = 1+d->dim;
		}else{
			offset = 0;
		}
		int u_counter;

		for (int i = 0; i < offset; ++i) {
			c[i] = -1;
		}
		MatrixXd u_intern(d->u.rows(), d->u.cols());
		for (int i = 0; i < d->dim; ++i) {
		  u_counter = 0;
		  for(int j = offset + i*d->u.rows(); j < offset + (i+1)*d->u.rows(); ++j){
	        u_intern(u_counter, i) = x[j];
	        u_counter++;
		  }
		}
		VectorXd c_intern(u_intern.rows());
		VectorXd dist(d->dim);
		for (int i = 0; i < u_intern.rows(); ++i) {
			for (int j = 0; j < d->dim; ++j) {
				dist(j) = (u_intern(i, j)-d->constraint_ball_center(j));
			}
	    	c_intern(i) = sqrt( dist.dot(dist) ) - d->constraint_ball_radius;
		}
		for (int i = 0; i < d->dim; ++i) {
		    u_counter = 0;
		  	for(int j = offset + i*d->u.rows(); j < offset + (i+1)*d->u.rows(); ++j){
		    c[j] = c_intern(u_counter);
	        u_counter++;
		  	}
		}
	}
  	return;
}

void FullyIndependentTrainingConditional::decrease_nugget(){
	if(K_u_u_nugget > K_u_u_nugget_min){
		K_u_u_nugget *= 0.1;
	}
	std::cout << "FITC: Decrease nugget to " << K_u_u_nugget << std::endl;
  return;
}
bool FullyIndependentTrainingConditional::increase_nugget(){
	if(K_u_u_nugget <= K_u_u_nugget_max){
		K_u_u_nugget *= 10;
	}
	std::cout << "FITC: Increase nugget to " << K_u_u_nugget << std::endl;
	if(K_u_u_nugget > K_u_u_nugget_max){
		return true;
	}
  return false;
}