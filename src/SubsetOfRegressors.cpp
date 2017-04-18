//
// Created by friedrich on 16.08.16.
//

#include <SubsetOfRegressors.hpp>
#include <assert.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>

//--------------------------------------------------------------------------------
SubsetOfRegressors::SubsetOfRegressors(int n, double &delta_input) :
		FullyIndependentTrainingConditional(n, delta_input) {
}
//--------------------------------------------------------------------------------
SubsetOfRegressors::SubsetOfRegressors(int n, double &delta_input, std::vector<double> gp_parameters_input) :
		FullyIndependentTrainingConditional(n, delta_input, gp_parameters_input) {
}
//--------------------------------------------------------------------------------
void SubsetOfRegressors::build(std::vector<std::vector<double> > const &nodes,
		std::vector<double> const &values, std::vector<double> const &noise) {

	int nb_u_nodes = u.rows();
	
	if (nb_u_nodes > 0) {
		std::cout << "SoR build with [" << nodes.size() << "," << nb_u_nodes
			<< "]" << std::endl;
		std::cout << "With Parameters: " << std::endl;
	    for ( int i = 0; i < dim+1; ++i )
	      std::cout << "gp_param = " << gp_parameters[i] << std::endl;
	    std::cout << std::endl;
		//nb_u_nodes++;
		
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
		/*
		if (resample_u){
			this->sample_u(nb_u_nodes);
			if (do_hp_estimation)
				this->estimate_hyper_parameters(nodes, values, noise);
  			resample_u = false;
		}
		*/

		//Set up matrix K_u_f and K_f_u
		compute_Kuf_and_Kuu();
		MatrixXd K_f_u = K_u_f.transpose();

		VectorXd diag_Q_f_f;
		compute_Qff(K_f_u, diag_Q_f_f);

		VectorXd diag_K_f_f;
		compute_Kff(diag_K_f_f);

		VectorXd diff_Kff_Qff(nb_gp_nodes); 
		diff_Kff_Qff.setZero();

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
void SubsetOfRegressors::evaluate(std::vector<double> const &x, double &mean,
		double &variance) {
	int nb_u_nodes = u.rows();
	if (nb_u_nodes > 0) {
		int nb_u_nodes = u.rows();
		VectorXd x_eigen;
		x_eigen.resize(x.size());
		for(int i = 0; i < x.size(); ++i){
			x_eigen(i) = x[i];
		}
		K0_eigen.resize(nb_u_nodes);
		for (int i = 0; i < nb_u_nodes; i++) {
			K0_eigen(i) = evaluate_kernel(x_eigen, u.row(i));
		}

//		std::cout << "K0" << std::endl;
//		VectorOperations::print_vector(K0);
//		std::cout << "alpha=(K_u_u + K_u_f*sigmaI*K_f_u)*K_u_f*noise*y:" << std::endl;
//		VectorOperations::print_vector(alpha);
		mean = K0_eigen.dot(alpha_eigen);
//		std::cout << "mean:" << mean << std::endl;

		variance = K0_eigen.dot(L_eigen.solve(K0_eigen));
		/*
		 std::cout << "Variance: " << variance << std::endl;
		 std::cout << "######################################" << std::endl;
		 evaluate_counter++;
		 assert(evaluate_counter<20*3);
		 */
	} else {
		GaussianProcess::evaluate(x, mean, variance);
	}
	return;

}

void SubsetOfRegressors::run_optimizer(std::vector<double> const &values){
	double optval;

  int exitflag;
  
  nlopt::opt* local_opt;
  nlopt::opt* global_opt;
  int dimp1 = gp_parameters_hp.size();
  set_optimizer(values, local_opt, global_opt);
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
	  //  std::cout << "gp_param = " << gp_parameters_hp[i] << std::endl;
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

void SubsetOfRegressors::estimate_hyper_parameters_all ( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise )
{
  //std::cout << "in sor1" << std::endl;
  if(u.rows() > 0){
	  copy_data_to_members(nodes, values, noise);

	  gp_pointer = this;

	  sample_u(u.rows());

	  set_hyperparameters();

	  run_optimizer(values);

	  copy_hyperparameters();

	  //this->build(nodes, values, noise);
	  for ( int i = 0; i < 1+dim; ++i )
	    std::cout << "gp_param = " << gp_parameters[i] << std::endl;
  }else{
  	GaussianProcess::estimate_hyper_parameters(nodes, values, noise);
  }

  return;
}

void SubsetOfRegressors::estimate_hyper_parameters_induced_only ( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise )
{
  //std::cout << "in sor1" << "u size:" << u.rows() << std::endl;
  if(u.rows() > 0){
	  copy_data_to_members(nodes, values, noise);

	  gp_pointer = this;

	  sample_u(u.rows());

	  set_hyperparameters_induced_only();

	  run_optimizer(values);

	  copy_hyperparameters();

	  //this->build(nodes, values, noise);
	  for ( int i = 0; i < 1+dim; ++i )
	    std::cout << "gp_param = " << gp_parameters[i] << std::endl;
  }else{
  	GaussianProcess::estimate_hyper_parameters(nodes, values, noise);
  }

  return;
}

void SubsetOfRegressors::estimate_hyper_parameters_ls_only ( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise )
{
  if(u.rows() > 0){
	  copy_data_to_members(nodes, values, noise);

	  gp_pointer = this;

	  sample_u(u.rows());

	  set_hyperparameters_ls_only();

	  run_optimizer(values);

	  copy_hyperparameters();

	  //this->build(nodes, values, noise);
	  for ( int i = 0; i < 1+dim; ++i )
	    std::cout << "gp_param = " << gp_parameters[i] << std::endl;
  }else{
  	GaussianProcess::estimate_hyper_parameters(nodes, values, noise);
  }

  return;
}

//--------------------------------------------------------------------------------
double SubsetOfRegressors::parameter_estimation_objective(std::vector<double> const &x,
                                                       std::vector<double> &grad,
                                                       void *data)
{

  SubsetOfRegressors *d = reinterpret_cast<SubsetOfRegressors*>(data);
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
		d->Lambda(i) = (pow( d->gp_noise.at(i) / 2e0 + d->noise_regularization, 2e0 ));
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
		Q_f_f(i,i) += d->Lambda(i) + d->Qff_opt_nugget;
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
	  /*for ( int i = 0; i < d->dim + 1; ++i )
	    std::cout << "gp_param = " << x[i] << std::endl;
	  for(int j = offset; j < offset + d->u.rows(); ++j)
			std::cout << "gp_param = " << x[j] <<","<<x[j+ d->u.rows()]<< std::endl;*/
  	
  	//std::cout << d->print <<" Objective: " << L1 << " " << L2 << " "<< result<< std::endl;
  }

  d->print++;

  return result;

}

double SubsetOfRegressors::parameter_estimation_objective_w_gradients(std::vector<double> const &x,
                                                       std::vector<double> &grad,
                                                       void *data)
{
	
  //std::cout << "in sor2" << std::endl;
 
  SubsetOfRegressors *d = reinterpret_cast<SubsetOfRegressors*>(data);
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
		d->Lambda(i) = (pow( d->gp_noise.at(i) / 2e0 + d->noise_regularization, 2e0 ));
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
		Q_f_f(i,i) += d->Lambda(i) + d->Qff_opt_nugget;
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
		VectorXd LambdaInv(nb_gp_nodes);
		for(int j = 0; j < nb_gp_nodes; ++j){
			LambdaInv(j) = 1.0/d->Lambda(j);
		}

		/*
		IOFormat HeavyFmt(FullPrecision, 0, ", ", ";\n", "", "", "[", "]");
		std::cout << "Kuudot\n" << Kuudot.format(HeavyFmt) << std::endl;
		std::cout << "Kffdot\n" << Kffdot.format(HeavyFmt) << std::endl;
		std::cout << "Kufdot\n" << Kufdot.format(HeavyFmt) << std::endl;
		std::cout << "Kuu\n" << d->K_u_u.format(HeavyFmt) << std::endl;
		std::cout << "Kuf\n" << d->K_u_f.format(HeavyFmt) << std::endl;
		std::cout << "Qff\n" << Q_f_f.format(HeavyFmt) << std::endl;
		std::cout << "noise:\n" << std::endl;
		for (int i = 0; i < nb_gp_nodes; ++i) {
			std::cout << (pow( d->gp_noise.at(i) / 2e0 + d->noise_regularization, 2e0 )) << std::endl;
		}		
		std::cout << std::endl;
		std::cout << "y:\n" << d->scaled_function_values_eigen << std::endl;
		*/

		//L1-term: dL1/dtheta= tr(Kuu^(-1) * Kuudot)
		double dL1 = (d->LLTofK_u_u.solve(Kuudot)).trace();

		//L2-term: d(log(det(Kuu+Kuf*(noise^2)^(-1)*Kfu)))/dtheta
		MatrixXd LambdaInvDiag = LambdaInv.asDiagonal();
		MatrixXd dSigma = Kuudot 
							+ 2*Kufdot*LambdaInvDiag*K_f_u;
							//+ d->K_u_f*LambdaInvDiag*Kfudot;
		//std::cout << "LambdaInv:\n" << LambdaInv.format(HeavyFmt) << std::endl;
		//std::cout << "LambdaInvDiag:\n" << LambdaInvDiag.format(HeavyFmt) << std::endl;
		//std::cout << "dSigma:\n" << dSigma.format(HeavyFmt) << std::endl;
		double dL2 = (d->L_eigen.solve(dSigma)).trace();

		//L3-term: d(yT*Q*y)/dtheta
		MatrixXd KfudotKuinvKuf = Kfudot * d->LLTofK_u_u.solve(d->K_u_f);
		MatrixXd KfuKuinvKufdot = K_f_u * d->LLTofK_u_u.solve(Kufdot);
		MatrixXd KfuKuinfKudotKuinvKuf = K_f_u*(d->LLTofK_u_u.solve(Kuudot*(d->LLTofK_u_u.solve(d->K_u_f))));
		MatrixXd dQ = KfudotKuinvKuf - KfuKuinfKudotKuinvKuf + KfuKuinvKufdot;
		//std::cout << "dQ:\n" << dQ.format(HeavyFmt) << std::endl;
		double dL3 = (-1)*d->scaled_function_values_eigen.dot(
						LLTofQ_f_f.solve(dQ*LLTofQ_f_f.solve(
							d->scaled_function_values_eigen)));

		grad[i] = 0.5*(-dL1 + dL2 + dL3);

		if ((d->print%1)==0){
	 		//std::cout << "Grad: " << i << "|" << 0.5*dL1 << " + "<< 0.5*dL2 << " + "<< 0.5*dL3 << " = " << grad[i] << std::endl;
		}
		//exit(-1);
		
	}

  d->print++;

  return result;

}

void SubsetOfRegressors::trust_region_constraint(unsigned int m, double* c, unsigned int n, const double* x, double* grad,
                                                     			void *data){
	SubsetOfRegressors *d = reinterpret_cast<SubsetOfRegressors*>(data);

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