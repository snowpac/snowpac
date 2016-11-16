//
// Created by friedrich on 16.08.16.
//

#include <SubsetOfRegressors.hpp>
#include <assert.h>
#include <random>
#include <algorithm>
#include <iostream>

//--------------------------------------------------------------------------------
SubsetOfRegressors::SubsetOfRegressors(int n, double &delta_input) :
		GaussianProcess(n, delta_input) {
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void SubsetOfRegressors::build(std::vector<std::vector<double> > const &nodes,
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
		for (int i = 0; i < nb_gp_nodes; ++i) {
			gp_nodes.push_back(nodes.at(i));
			gp_noise.push_back(noise.at(i));
		}

		this->sample_u(nb_u_nodes);

		std::cout << "Creating u done!" << std::endl;
		VectorOperations::print_matrix(u);

		//std::cout << "Noise: " << std::endl;
		//VectorOperations::print_vector(gp_noise);

		//Set up matrix K_u_f and K_f_u
		K_u_f.clear();
		K_u_f.resize(nb_u_nodes);
		for (int i = 0; i < nb_u_nodes; ++i) {
			for (int j = 0; j < nb_gp_nodes; ++j) {
				K_u_f.at(i).push_back(evaluate_kernel(u.at(i), gp_nodes.at(j)));
			}
		}
		K_f_u.clear();
		K_f_u.resize(nb_gp_nodes);
		for (int i = 0; i < nb_gp_nodes; ++i) {
			K_f_u.at(i).resize(nb_u_nodes);
		}
		VectorOperations::mat_transpose(K_u_f, K_f_u);
		//std::cout << "K_u_f and K_f_u done!" << std::endl;
		//VectorOperations::print_matrix(K_u_f);

		//Set up matrix K_u_u
		L.clear();
		L.resize(nb_u_nodes);
		for (int i = 0; i < nb_u_nodes; ++i) {
			for (int j = 0; j <= i; ++j) {
				L.at(i).push_back(evaluate_kernel(u.at(i), u.at(j)));
			}
		}
		//std::cout << "K_u_u done!" << std::endl;
		//VectorOperations::print_matrix(L);

		//Compute 1/noise^2*eye(length(f))*K_f_u (we save it directly in K_f_u since we do not change it later)
		for (int i = 0; i < nb_gp_nodes; ++i) {
			for (int j = 0; j < nb_u_nodes; ++j) {
				K_f_u[i][j] *= 1.0
						/ pow(gp_noise.at(i) / 2e0 + noise_regularization, 2e0);
			}
		}
		//std::cout << "eye(noise)*K_f_u" << std::endl;
		//VectorOperations::print_matrix(K_f_u);

		//Compute K_u_f*K_f_u = K_u_f_f_u
		std::vector<std::vector<double> > K_u_f_f_u;
		K_u_f_f_u.resize(nb_u_nodes);
		for (int i = 0; i < nb_u_nodes; ++i) {
			K_u_f_f_u[i].resize(nb_u_nodes);
		}
		VectorOperations::mat_product(K_u_f, K_f_u, K_u_f_f_u);

		//Add K_u_f_f_u + K_u_u = Sigma_not_inv
		for (int i = 0; i < nb_u_nodes; ++i) {
			for (int j = 0; j <= i; ++j) {
				L[i][j] += K_u_f_f_u[i][j];
				//Nugget
				if (i == j) {
					L[i][j] += 0.0001;
				}
			}
		}

		//Compute Cholesky of K_u_u
		CholeskyFactorization::compute(L, pos, rho, nb_u_nodes);
		assert(pos == 0);

		//std::cout << "f:" << std::endl;
		//VectorOperations::print_vector(values);

		//Set f and compute 1/noise^2*eye(length(f)) * f
		scaled_function_values.clear();
		scaled_function_values.resize(nb_gp_nodes);
		for (int i = 0; i < nb_gp_nodes; i++) {
			scaled_function_values.at(i) = values.at(i);
			//      scaled_function_values.at(i) = values.at(i) - min_function_value;
			//      scaled_function_values.at(i) /= 5e-1*( max_function_value-min_function_value );
			//      scaled_function_values.at(i) -= 1e0;
		}
		std::vector<double> noisy_values(scaled_function_values.size());
		for (int i = 0; i < nb_gp_nodes; ++i) {
			noisy_values[i] = scaled_function_values[i] * 1.0
					/ pow(gp_noise.at(i) / 2e0 + noise_regularization, 2e0);
		}
		//std::cout << "eye(noise) * f:" << std::endl;
		//VectorOperations::print_vector(noisy_values);

		//Compute K_u_f * f = alpha
		alpha.clear();
		alpha.resize(nb_u_nodes);
		VectorOperations::mat_vec_product(K_u_f, noisy_values, alpha);

		//std::cout << "Alpha:" << std::endl;
		//VectorOperations::print_vector(alpha);

		//Solve Sigma_not_inv^(-1)*alpha
		forward_substitution(L, alpha);
		backward_substitution(L, alpha);

		//std::cout << "Sigma*K_u_f*noise*y:" << std::endl;
		//VectorOperations::print_vector(alpha);

	} else {
		GaussianProcess::build(nodes, values, noise);
	}
	return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void SubsetOfRegressors::update(std::vector<double> const &x, double &value,
		double &noise) {
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
void SubsetOfRegressors::evaluate(std::vector<double> const &x, double &mean,
		double &variance) {
	if (u.size() > 0) {
		int nb_u_nodes = u.size();

		K0.clear();
		K0.resize(nb_u_nodes);
		for (int i = 0; i < nb_u_nodes; i++) {
			K0.at(i) = evaluate_kernel(x, u.at(i));
		}
		/*
		 std::cout << "##############Evaluate################" << std::endl;
		 std::cout << "At x:"<< std::endl;
		 VectorOperations::print_vector(x);
		 std::cout << "K_star_u:" << std::endl;
		 VectorOperations::print_vector(K0);
		 */
		mean = VectorOperations::dot_product(K0, alpha);
		//std::cout << "Mean: " << mean << std::endl;

		TriangularMatrixOperations::forward_substitution(L, K0);
		variance = VectorOperations::dot_product(K0, K0);
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

void SubsetOfRegressors::get_induced_nodes(
		std::vector<std::vector<double> > &induced_nodes) const {
	for (int i = 0; i < u.size(); ++i) {
		induced_nodes.push_back(u[i]);
	}
	return;
}

void SubsetOfRegressors::sample_u(const int &nb_u_nodes) {

	u.clear();

	std::vector<int> u_idx_left_over;
	std::vector<int> u_idx_from_active;
	//Set distribution for sampling the indices for the u samples
	std::random_device rd;
	int random_seed = rd();
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
	u.resize(nb_u_nodes);
	for (int i = 0; i < u_idx_from_active.size(); ++i) {
		for (int j = 0; j < evaluations.nodes.at(u_idx_from_active.at(i)).size();
				++j) {
			u.at(i).push_back(evaluations.nodes.at(u_idx_from_active.at(i)).at(j));
		}
	}
	for (int i = 0; i < u_idx_left_over.size(); ++i) {
		for (int j = 0; j < gp_nodes.at(u_idx_left_over.at(i)).size(); ++j) {
			u.at(u_idx_from_active.size() + i).push_back(
					gp_nodes.at(u_idx_left_over.at(i)).at(j));
		}
	}
	return;
}