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

void DeterministicTrainingConditional::run_optimizer(){
	double optval;

  int exitflag;
  
  nlopt::opt* local_opt;
  nlopt::opt* global_opt;

  int dimp1 = 1+dim+u.rows()*dim;
  set_optimizer(local_opt, global_opt);

  if (optimize_global){
  	  print = 0;
 	  std::cout << "Global optimization" << std::endl;
	  exitflag=-20;
	  global_opt->set_min_objective( parameter_estimation_objective, gp_pointer);
	  exitflag = global_opt->optimize(gp_parameters, optval);

	  std::cout << "exitflag = "<< exitflag<<std::endl;
  	  std::cout << "Function calls: " << print << std::endl;
	  std::cout << "OPTVAL .... " << optval << std::endl;
	  for ( int i = 0; i < dimp1; ++i )
	    std::cout << "gp_param = " << gp_parameters[i] << std::endl;
	  std::cout << std::endl;
  }
  if (optimize_local){
  	  print = 0;
  	  std::cout << "Local optimization" << std::endl;
	  exitflag=-20;
	  //try {
	  local_opt->set_min_objective( parameter_estimation_objective_w_gradients, gp_pointer);
	  exitflag = local_opt->optimize(gp_parameters, optval);

	  std::cout << "exitflag = "<< exitflag<<std::endl;
  	  std::cout << "Function calls: " << print << std::endl;
	  std::cout << "OPTVAL .... " << optval << std::endl;
	  for ( int i = 0; i < dimp1; ++i )
	    std::cout << "gp_param = " << gp_parameters[i] << std::endl;
	  std::cout << std::endl;
  }
  
  delete local_opt;
  delete global_opt;

  return;
}

void DeterministicTrainingConditional::estimate_hyper_parameters ( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise )
{
  //std::cout << "in dtc1" << std::endl;
  copy_data_to_members(nodes, values, noise);

  gp_pointer = this;

  run_optimizer();

  update_induced_points();
  
  resample_u = false;
  this->build(nodes, values, noise);

  return;
}

//--------------------------------------------------------------------------------
double DeterministicTrainingConditional::parameter_estimation_objective(std::vector<double> const &x,
                                                       std::vector<double> &grad,
                                                       void *data){
  DeterministicTrainingConditional *d = reinterpret_cast<DeterministicTrainingConditional*>(data);
  int offset = 1+d->dim;
  int u_counter;
  double nugget = 0.00001;
  double nugget2 = 0.0;
  double nuggetQff = 0.00001;
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
		Q_f_f(i,i) += d->Lambda(i)+ nuggetQff;
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
  	
  	//std::cout << d->print <<" Objective: " << L1 << " " << L2 << " "<< result<< std::endl;
  }

  d->print++;

  return result;
}

double DeterministicTrainingConditional::parameter_estimation_objective_w_gradients(std::vector<double> const &x,
                                                       std::vector<double> &grad,
                                                       void *data)
{

  DeterministicTrainingConditional *d = reinterpret_cast<DeterministicTrainingConditional*>(data);
  int offset = 1+d->dim;
  int u_counter;
  double nugget = 0.00001;
  double nugget2 = 0.0;
  double nuggetQff = 0.00001;
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
		Q_f_f(i,i) += d->Lambda(i)+ nuggetQff;
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
  	
  	//std::cout << d->print <<" Objective: " << L1 << " " << L2 << " "<< result<< std::endl;
  }


	//Gradient computation:
	int dim_grad = 1+d->dim+d->u.rows()*d->dim;
	grad.resize(dim_grad);

	for(int i = 0; i < grad.size(); i++){

		MatrixXd Kffdot;
		MatrixXd Kufdot;
		MatrixXd Kfudot;
		MatrixXd Kuudot;
		VectorXd LambdaDot;
		if(i == 0){//grad sigma_f
			d->derivate_K_u_u_wrt_sigmaf(x, Kuudot);
			d->derivate_K_f_f_wrt_sigmaf(x, Kffdot);
			d->derivate_K_u_f_wrt_sigmaf(x, Kufdot);
			Kfudot = Kufdot.transpose();
		}else if(i > 0 && i < 1 + d->dim){//grad length parameter
			d->derivate_K_u_u_wrt_l(x, i-1, Kuudot);
			d->derivate_K_f_f_wrt_l(x, i-1, Kffdot);
			d->derivate_K_u_f_wrt_l(x, i-1, Kufdot);
			Kfudot = Kufdot.transpose();
		}else{//grad u
			int uidx = (i-offset)%d->u.rows();
			int udim = (int) ((i-offset)/d->u.rows());
			d->derivate_K_u_u_wrt_uik(x, uidx, udim, Kuudot);
			d->derivate_K_u_f_wrt_uik(x, uidx, udim, Kufdot);
			Kffdot.resize(nb_gp_nodes, nb_gp_nodes);
			Kffdot.setZero();
			Kfudot = Kufdot.transpose(); 
		}
		VectorXd LambdaInv(nb_gp_nodes);
		for(int j = 0; j < nb_gp_nodes; ++j){
			LambdaInv(j) = 1.0/d->Lambda(j);
		}
		//L1-term: dL1/dtheta= tr(Kuu^(-1) * Kuudot)
		double dL1 = d->LLTofK_u_u.solve(Kuudot).trace();

		//L2-term: d(log(det(Kuu+Kuf*(noise^2)^(-1)*Kfu)))/dtheta
		MatrixXd LambdaInvDiag = LambdaInv.asDiagonal();
		MatrixXd dSigma = Kuudot 
							+ 2*Kufdot*LambdaInvDiag*K_f_u;
							//+ d->K_u_f*LambdaInvDiag*Kfudot;
		double dL2 = (d->L_eigen.solve(dSigma)).trace();

		//L3-term: d(yT*Q*y)/dtheta
		MatrixXd KfudotKuinvKuf = Kfudot * d->LLTofK_u_u.solve(d->K_u_f);
		MatrixXd KfuKuinvKufdot = K_f_u * d->LLTofK_u_u.solve(Kufdot);
		MatrixXd KfuKuinfKudotKuinvKuf = K_f_u*(d->LLTofK_u_u.solve(Kuudot*(d->LLTofK_u_u.solve(d->K_u_f))));
		MatrixXd dQ = KfudotKuinvKuf - KfuKuinfKudotKuinvKuf + KfuKuinvKufdot;
		double dL3 = (-1)*d->scaled_function_values_eigen.dot(
						LLTofQ_f_f.solve(dQ*LLTofQ_f_f.solve(
							d->scaled_function_values_eigen)));

		grad[i] = 0.5*(-dL1 + dL2 + dL3);

		if ((d->print%1)==0){
		 	//std::cout << "Grad: " << i << "|" << 0.5*dL1 << " + "<< 0.5*dL2 << " + "<< 0.5*dL3 << " = " << grad[i] << std::endl;
		}
	}

  d->print++;

  return result;

}