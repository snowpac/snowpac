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

 #include <Eigen/Dense>

//--------------------------------------------------------------------------------
FullyIndependentTrainingConditional::FullyIndependentTrainingConditional(int n,
		double &delta_input) :
		GaussianProcess(n, delta_input) {

}
//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------
double FullyIndependentTrainingConditional::evaluate_kernel ( VectorXd const &x,
                                          VectorXd const &y )
{
  return evaluate_kernel ( x, y, gp_parameters );
/*
  dist = 0e0;
  for ( int i = 0; i < dim; ++i )
    dist += pow( (x.at(i) - y.at(i)), 2e0) /  gp_parameters.at( i+1 );
  kernel_evaluation = exp(-dist / 2e0 );
  
  return kernel_evaluation * gp_parameters.at( 0 ) ;
*/
}
//--------------------------------------------------------------------------------
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
				K_u_u(i, j) += 0.000001;
		}
	}
}

 void FullyIndependentTrainingConditional::compute_Qff(const MatrixXd& K_f_u, VectorXd& diag_Q_f_f) {
	int nb_u_nodes = u.rows();
	LLT < MatrixXd > LLTofK_u_u(K_u_u);
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

//--------------------------------------------------------------------------------

void FullyIndependentTrainingConditional::build(
		std::vector<std::vector<double> > const &nodes,
		std::vector<double> const &values, std::vector<double> const &noise) {
	int nb_u_nodes;
	if (nodes.size() < evaluations.active_index.size()) {
		nb_u_nodes = (nodes.size());
	} else if ((nodes.size()) * u_ratio < evaluations.active_index.size()) {
		nb_u_nodes = evaluations.active_index.size();
	} else {
		nb_u_nodes = (nodes.size()) * u_ratio;
	}
	std::cout << "In Build Gaussian with [" << nodes.size() << "," << nb_u_nodes
			<< "]" << std::endl;
	if (nb_u_nodes >= min_nb_u_nodes) {
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

		if (resample_u)
			this->sample_u(nb_u_nodes);
		resample_u = true;

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
	int nb_u_nodes;
	if (gp_nodes.size() < evaluations.active_index.size()) {
		nb_u_nodes = (gp_nodes.size() + 1);
	} else if ((gp_nodes.size() + 1) * u_ratio
			< evaluations.active_index.size()) {
		nb_u_nodes = evaluations.active_index.size();
	} else {
		nb_u_nodes = (gp_nodes.size() + 1) * u_ratio;
	}
	//std::cout << "In update Gaussian with [" << gp_nodes.size()+1 << "," << nb_u_nodes <<"]" << std::endl;
	if (nb_u_nodes >= min_nb_u_nodes) {
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
	if (u.size() > 0) {
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

//		std::cout << "alpha= (K_u_u + K_u_f Lambda_inv K_f_u)\(K_u_f*LambdaInv*f):" << std::endl;
//		VectorOperations::print_vector(alpha);
		mean = K0_eigen.dot(alpha_eigen);
//		std::cout << "mean:" << mean << std::endl;
		//std::cout << "Mean: " << mean << std::endl;

		double variance_term3 = K0_eigen.dot((L_eigen.solve(K0_eigen)));
//		std::cout << "variance_term3:" << variance_term3 << std::endl;


		double variance_term2 = K0_eigen.dot(K_u_u.inverse()*K0_eigen);
//		std::cout << "variance_term2:" << variance_term2 << std::endl;
		/*
		 std::cout << "Variance: " << variance << std::endl;
		 std::cout << "######################################" << std::endl;
		 evaluate_counter++;
		 assert(evaluate_counter<20*3);
//		 */
//		std::cout << "K**:" << evaluate_kernel(x,x) << std::endl;
		variance = evaluate_kernel(x_eigen,x_eigen)-variance_term2+variance_term3;
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

void FullyIndependentTrainingConditional::sample_u(const int &nb_u_nodes) {

	u.setConstant(-1);
	std::vector<int> u_idx_left_over;
	std::vector<int> u_idx_from_active;
	//Set distribution for sampling the indices for the u samples
	std::random_device rd;
	int random_seed = 1;//rd();
	std::mt19937 random_generator(random_seed);
	std::vector<double> nodes_weights_vector;

	/*double prefactor = 200.0/( (nb_gp_nodes + 1)*nb_gp_nodes);
	 for(int i = 1; i <= nb_gp_nodes; ++i){
	 nodes_weights_vector.push_back(prefactor * i);
	 }*/

	int nb_left_over_u = nb_u_nodes - evaluations.active_index.size();

	int random_draw = -1;
	std::vector<int>::iterator it;
	if (nb_left_over_u > 0) {
		u_idx_from_active = evaluations.active_index;

		nodes_weights_vector.resize(gp_nodes.size());
		double nodes_weights = 100.0 / gp_nodes.size();
		std::fill(nodes_weights_vector.begin(), nodes_weights_vector.end(),
				nodes_weights);
		std::discrete_distribution<> d(nodes_weights_vector.begin(),
				nodes_weights_vector.end());

		//Sample the indices
		random_draw = d(random_generator);
		u_idx_left_over.push_back(random_draw);
		std::cout << "Drawing";
		int no_draws = 0;
		while (u_idx_left_over.size() < nb_left_over_u) {
			random_draw = d(random_generator);
			no_draws++;
			it = std::find(u_idx_left_over.begin(), u_idx_left_over.end(),
					random_draw);
			if (it == u_idx_left_over.end()) {
				u_idx_left_over.push_back(random_draw);
				no_draws = 0;
			}
		}
	} else {
		nodes_weights_vector.resize(evaluations.active_index.size());
		double nodes_weights = 100.0 / evaluations.active_index.size();
		std::fill(nodes_weights_vector.begin(), nodes_weights_vector.end(),
				nodes_weights);

		std::discrete_distribution<> d(nodes_weights_vector.begin(),
				nodes_weights_vector.end());

		//Sample the indices
		random_draw = d(random_generator);
		u_idx_from_active.push_back(evaluations.active_index[random_draw]);
		std::cout << "Drawing";
		int no_draws = 0;
		while (u_idx_from_active.size() < nb_u_nodes) {
			random_draw = d(random_generator);
			no_draws++;
			it = std::find(u_idx_from_active.begin(), u_idx_from_active.end(),
					evaluations.active_index[random_draw]);
			if (it == u_idx_from_active.end()) {
				u_idx_from_active.push_back(evaluations.active_index[random_draw]);
				no_draws = 0;
			}
		}
		/*
		 //New Change: Take all active nodes always
		 for(int i = 0; i < evaluations.active_index.size(); ++i){
		 u_idx_from_active.push_back(evaluations.active_index[i]);
		 }
		 */
	}
	std::cout << std::endl;
	std::sort(u_idx_from_active.begin(), u_idx_from_active.end());
	std::sort(u_idx_left_over.begin(), u_idx_left_over.end());
	std::cout << "Sampling " << nb_u_nodes << " idx done" << std::endl;
	std::cout << "From active: ";
	for (int i = 0; i < u_idx_from_active.size(); ++i) {
		std::cout << " " << u_idx_from_active.at(i);
	}
	std::cout << std::endl;
	std::cout << "From left over: ";
	for (int i = 0; i < u_idx_left_over.size(); ++i) {
		std::cout << " " << u_idx_left_over.at(i);
	}
	std::cout << std::endl;

	//Create u vector
	u.resize(nb_u_nodes, dim);
	for (int i = 0; i < u_idx_from_active.size(); ++i) {
		for (int j = 0; j < evaluations.nodes.at(u_idx_from_active.at(i)).size();
				++j) {
			u(i, j) = (evaluations.nodes.at(u_idx_from_active.at(i)).at(j));
		}
	}
	for (int i = 0; i < u_idx_left_over.size(); ++i) {
		for (int j = 0; j < gp_nodes.at(u_idx_left_over.at(i)).size(); ++j) {
			u(u_idx_from_active.size() + i, j) = gp_nodes.at(u_idx_left_over.at(i)).at(j);
		}
	}
	return;
}


void FullyIndependentTrainingConditional::estimate_hyper_parameters ( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise )
{
  nb_gp_nodes = nodes.size();
  gp_nodes.clear();
  gp_noise.clear();
  for ( int i = 0; i < nb_gp_nodes; ++i ) {
    gp_nodes.push_back ( nodes.at(i) );
    gp_noise.push_back ( noise.at(i) );
  }

  gp_pointer = this;

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

  double optval;
  //adjust those settings to optimize GP approximation
  //--------------------------------------------------
//  double max_noise = 0e0;
//  for (int i = 0; i < nb_gp_nodes; i++) {
//    if (gp_noise.at( i ) > max_noise)
//      max_noise = gp_noise.at( i );
//  }
  lb.resize(1+dim+u.rows()*dim);
  ub.resize(1+dim+u.rows()*dim);
  lb[0] = 1e-1; // * pow(1000e0 * max_noise / 2e0, 2e0);
  ub[0] = 1e1;// * pow(1000e0 * max_noise / 2e0, 2e0);
  double delta_threshold = *delta;
  if (delta_threshold < 1e-2) delta_threshold = 1e-2;
  for (int i = 0; i < dim; ++i) {
      lb[i+1] = 1e-1 * delta_threshold;
      ub[i+1] = 1e2 * delta_threshold;
  }
  int offset = 1+dim;
  std::vector<double> lb_u(dim);
  lb_u[0] = -1; //TODO 
  lb_u[1] = -1;
  std::vector<double> ub_u(dim);
  ub_u[0] = 5;
  ub_u[1] = 5;
  for (int i = 0; i < dim; ++i) {
	  for(int j = offset + i*u.rows(); j < offset + (i+1)*u.rows(); ++j){
          lb[j] = lb_u[i];
          ub[j] = ub_u[i];
	  }
  }

  gp_parameters.resize(1+dim+u.rows()*dim);
  if (gp_parameters[0] < 0e0) {
    gp_parameters[0] = lb[0]*5e-1 + 5e-1*ub[0];
    for (int i = 1; i < dim+1; ++i) {
      gp_parameters[i] = (lb[i]*5e-1 + 5e-1*ub[i]);
    }
  } else {
    for (int i = 0; i < dim+1; ++i) {
      if ( gp_parameters[i] <= lb[i] ) gp_parameters[i] = 1.1 * lb[i];
      if ( gp_parameters[i] >= ub[i] ) gp_parameters[i] = 0.9 * ub[i];
    }
  }

  for ( int i = 0; i < dim + 1; ++i )
    std::cout << "gp_param = " << gp_parameters[i] << std::endl;
  int u_counter;
  for (int i = 0; i < dim; ++i) {
  	  u_counter = 0;
  	  for(int j = offset + i*u.rows(); j < offset + (i+1)*u.rows(); ++j){
            gp_parameters[j] = u(u_counter,i);    		
            std::cout << "gp_param = " << gp_parameters[j] << std::endl;
            u_counter++;
  	  }
  } 
  //--------------------------------------------------

  //initialize optimizer from NLopt library
  int dimp1 = 1+dim+u.rows()*dim;
//  nlopt::opt opt(nlopt::LD_CCSAQ, dimp1);
//  nlopt::opt opt(nlopt::LN_BOBYQA, dimp1);
//
  nlopt::opt opt(nlopt::GN_CRS2_LM, dimp1);

  //opt = nlopt_create(NLOPT_LN_COBYLA, dim+1);
  opt.set_lower_bounds( lb );
  opt.set_upper_bounds( ub );

  opt.set_min_objective( parameter_estimation_objective, gp_pointer);

 // opt.set_xtol_abs(1e-2);
//  opt.set_xtol_rel(1e-2);
//set timeout to NLOPT_TIMEOUT seconds
  opt.set_maxtime(1800.0);
  //perform optimization to get correction factors

  int exitflag=-20;
  //try {
    exitflag = opt.optimize(gp_parameters, optval);
  //} catch (...) {
  //  gp_parameters[0] = lb[0]*5e-1 + 5e-1*ub[0];
  //  for (int i = 1; i < dim+1; ++i) {
  //    gp_parameters[i] = (lb[i]*5e-1 + 5e-1*ub[i]);
  //  }
  //}

  std::cout << "exitflag = "<< exitflag<<std::endl;
  std::cout << "OPTVAL .... " << optval << std::endl;
  for ( int i = 0; i < dim + 1; ++i )
    std::cout << "gp_param = " << gp_parameters[i] << std::endl;
  std::cout << std::endl;

  resample_u = false;
  this->build(nodes, values, noise);

  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double FullyIndependentTrainingConditional::parameter_estimation_objective(std::vector<double> const &x,
                                                       std::vector<double> &grad,
                                                       void *data)
{

  FullyIndependentTrainingConditional *d = reinterpret_cast<FullyIndependentTrainingConditional*>(data);
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
		d->Lambda(i) = (diag_K_f_f(i) - diag_Q_f_f(i) + pow( d->gp_noise.at(i) / 2e0 + d->noise_regularization, 2e0 ));
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
  if ((d->print%100)==0){
	  for ( int i = 0; i < d->dim + 1; ++i )
	    std::cout << "gp_param = " << x[i] << std::endl;
	  for(int j = offset; j < offset + d->u.rows(); ++j)
			std::cout << "gp_param = " << x[j] <<","<<x[j+ d->u.rows()]<< std::endl;
  	
    std::cout << L1 << ' ' << L2 << std::endl;
  	std::cout << "Objective: "  << result<< std::endl;
  }

  d->print++;

  return result;

}

