//
// Created by friedrich on 16.08.16.
//

#include <DeterministicTrainingConditional.hpp>
#include <assert.h>
#include <random>
#include <algorithm>
#include <iostream>

//--------------------------------------------------------------------------------
DeterministicTrainingConditional::DeterministicTrainingConditional(int n, double &delta_input) :
		SubsetOfRegressors(n, delta_input) {
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void DeterministicTrainingConditional::build(std::vector<std::vector<double> > const &nodes,
		std::vector<double> const &values, std::vector<double> const &noise) {
	SubsetOfRegressors::build(nodes, values, noise);
}
//--------------------------------------------------------------------------------
void DeterministicTrainingConditional::evaluate(std::vector<double> const &x, double &mean,
		double &variance) {
	FullyIndependentTrainingConditional::evaluate(x, mean, variance);
}

//--------------------------------------------------------------------------------
double DeterministicTrainingConditional::parameter_estimation_objective(std::vector<double> const &x,
                                                       std::vector<double> &grad,
                                                       void *data){
  DeterministicTrainingConditional *d = reinterpret_cast<DeterministicTrainingConditional*>(data);
  int offset = 1+d->dim;
  int u_counter;
  double nugget = 0.0001;
  double nugget2 = 0.0;
  for (int i = 0; i < d->dim; ++i) {
  	  u_counter = 0;
  	  for(int j = offset + i*d->u.rows(); j < offset + (i+1)*d->u.rows(); ++j){
            d->u(u_counter,i) = x[j];
            u_counter++;
  	  }
  }

  //Compute Kuf, Kuu
  //Set up matrix K_u_f and K_f_u
	int nb_gp_nodes = d->gp_nodes.size();
	int nb_u_nodes = d->u.rows();
	d->K_u_f.resize(nb_u_nodes, nb_gp_nodes);
	for (int i = 0; i < nb_u_nodes; ++i) {
		for (int j = 0; j < nb_gp_nodes; ++j) {
			d->K_u_f(i,j) = d->evaluate_kernel(d->u.row(i), d->gp_nodes_eigen.row(j), x);
		}
	}
	MatrixXd K_f_u = d->K_u_f.transpose();
	//std::cout << "Kuf\n" << d->K_u_f << std::endl;

	//Set up matrix K_u_u
	d->K_u_u.resize(nb_u_nodes, nb_u_nodes);
	for (int i = 0; i < nb_u_nodes; ++i) {
		for (int j = 0; j < nb_u_nodes; ++j) {
			d->K_u_u(i,j) = d->evaluate_kernel(d->u.row(i), d->u.row(j), x);
			if(i==j)
				d->K_u_u(i,j) += nugget;
		}
	}
	//std::cout << "Kuu\n" << d->K_u_u << std::endl;
	LLT<MatrixXd> LLTofK_u_u(d->K_u_u);
	MatrixXd K_u_u_u_f = LLTofK_u_u.solve(d->K_u_f);

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
									 d->gp_nodes_eigen.row(i), x);
	}
	/*std::cout << "diag_K_f_f\n" << diag_K_f_f << std::endl;
	std::cout << "diag_Q_f_f\n" << diag_Q_f_f << std::endl;*/

	d->Lambda.resize(nb_gp_nodes);
	//std::cout << "noise\n" << std::endl;
	for (int i = 0; i < nb_gp_nodes; ++i) {
		d->Lambda(i) = (pow( d->gp_noise.at(i) / 2e0 + d->noise_regularization, 2e0 ));
	}
	//std::cout << std::endl;

	MatrixXd Lambda_K_f_u;
	Lambda_K_f_u.resize(nb_gp_nodes, nb_u_nodes);
	for(int i = 0; i < nb_gp_nodes; i++){
		for(int j = 0; j < nb_u_nodes; j++){
			Lambda_K_f_u(i,j) = ((1.0/(d->Lambda(i) + nugget2)) * K_f_u(i,j));
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
		L12 += log(LLTofK_u_u.matrixL()(i,i));
    	//std::cout << L1 << "L11-" << i << std::endl;
	}

	double det_Leigen = 0.0;
	for (int i = 0; i < d->u.rows(); ++i){
		det_Leigen += log(d->L_eigen.matrixL()(i,i));
    	//std::cout << d->L_eigen.matrixL()(i,i) << " " << log(d->L_eigen.matrixL()(i,i)) << "det_Leigen-" << i << std::endl;
	}

	double L11 = 0.0;
    //std::cout << L1 << "L12 " << std::endl;
	for (int i = 0; i < d->nb_gp_nodes; ++i){
		L11 += log(d->Lambda(i));
		//std::cout << "Lambda: " << i << " " << d->Lambda(i) <<" "<<log(d->Lambda(i)) << std::endl;
	}
    double L1 = 0.0;
	L1 = 0.5*L11 + (2*0.5)*det_Leigen - (2*0.5)*L12;

	MatrixXd Q_f_f = K_f_u*K_u_u_u_f;
	for(int i = 0; i < nb_gp_nodes; i++){
		Q_f_f(i,i) += d->Lambda(i);
	}
	LLT<MatrixXd> LLTofQ_f_f(Q_f_f);
	double L2 = 0.5*d->scaled_function_values_eigen.dot(LLTofQ_f_f.solve(d->scaled_function_values_eigen));
  	
  	//std::cout << d->scaled_function_values_eigen << std::endl;
  double result = L1 + L2;
 
  if (isinf(result) || isnan(result)){
  	std::cout << "Result is inf or nan" << std::endl;
  	std::cout << "L11 " << L11 << " " << 'x' << std::endl;
    std::cout << "L12 " << L12 << " "  << log(d->K_u_u.determinant()) << std::endl;
    std::cout << "L13 " << det_Leigen << " " << log(Sigma.determinant()) << std::endl;
    std::cout << L1 << ' ' << L2 << std::endl;
  	result = std::numeric_limits<double>::infinity();
  }
  if ((d->print%1000)==0){
	  /*for ( int i = 0; i < d->dim + 1; ++i )
	    std::cout << "gp_param = " << x[i] << std::endl;
	  for(int j = offset; j < offset + d->u.rows(); ++j)
			std::cout << "gp_param = " << x[j] <<","<<x[j+ d->u.rows()]<< std::endl;*/
  	
  	std::cout << d->print <<" Objective: " << L1 << " " << L2 << " "<< result<< std::endl;
  }

  d->print++;

  return result;
}
