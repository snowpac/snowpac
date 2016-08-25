//
// Created by friedrich on 16.08.16.
//

#include <ApproxGaussianProcessSoR.hpp>
#include <assert.h>
#include <random>
#include <algorithm>
#include <iostream>

//--------------------------------------------------------------------------------
ApproxGaussianProcessSoR::ApproxGaussianProcessSoR(int n, double &delta_input):GaussianProcess(n, delta_input) {}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void ApproxGaussianProcessSoR::build ( std::vector< std::vector<double> > const &nodes,
                              std::vector<double> const &values,
                              std::vector<double> const &noise )
{

    int nb_u_nodes = nodes.size()*u_ratio;
    std::cout << "In Build Gaussian with [" << nodes.size() << "," << nb_u_nodes <<"]" << std::endl;
    if(nb_u_nodes >= min_nb_u_nodes) {
        nb_u_nodes++;

        nb_gp_nodes = nodes.size();
        gp_nodes.clear();
        gp_noise.clear();
        for ( int i = 0; i < nb_gp_nodes; ++i ) {
            gp_nodes.push_back(nodes.at(i));
            gp_noise.push_back(noise.at(i));
        }
        u_idx.clear();
        //Set distribution for sampling the indices for the u samples
        std::random_device rd;
        int random_seed = rd();
        std::mt19937 random_generator(random_seed);
        std::vector<double> nodes_weights_vector;
        nodes_weights_vector.resize(nodes.size());
        double nodes_weights = 100.0 / nodes.size();
        std::fill(nodes_weights_vector.begin(), nodes_weights_vector.end(), nodes_weights);
        /*double prefactor = 200.0/( (nb_gp_nodes + 1)*nb_gp_nodes);
        for(int i = 1; i <= nb_gp_nodes; ++i){
            nodes_weights_vector.push_back(prefactor * i);
        }*/
        std::discrete_distribution<> d(nodes_weights_vector.begin(), nodes_weights_vector.end());

        //Sample the indices
        int random_draw = -1;
        std::vector<int>::iterator it;
        random_draw = d(random_generator);
        u_idx.push_back(random_draw);
        std::cout << "Drawing";
        int no_draws = 0;
        while (u_idx.size() < nb_u_nodes) {
            random_draw = d(random_generator);
            no_draws++;
            it = std::find(u_idx.begin(), u_idx.end(), random_draw);
            if (it == u_idx.end()) {
                u_idx.push_back(random_draw);
                no_draws=0;
            }
        }
        std::cout << std::endl;
        std::sort(u_idx.begin(), u_idx.end());
        std::cout << "Sampling done with";
        for(int i = 0; i < u_idx.size(); ++i){
            std::cout << " " << u_idx.at(i);
        }
        std::cout << std::endl;

        //Create u vector
        std::vector< std::vector<double> > u;
        u.resize(nb_u_nodes);
        for (int i = 0; i < nb_u_nodes; ++i) {
            for (int j = 0; j < nodes.at(u_idx.at(i)).size(); ++j) {
                u.at(i).push_back(nodes.at(u_idx.at(i)).at(j));
            }
        }
        //std::cout << "Creating u done!" << std::endl;

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
        for (int i = 0; i < nb_gp_nodes; ++i){
            for (int j = 0; j < nb_u_nodes; ++j){
                K_f_u[i][j] *=  1.0/pow( gp_noise.at(i) / 2e0 + noise_regularization, 2e0 );
            }
        }
        //std::cout << "eye(noise)*K_f_u" << std::endl;
        //VectorOperations::print_matrix(K_f_u);

        //Compute K_u_f*K_f_u = K_u_f_f_u
        std::vector< std::vector<double> > K_u_f_f_u;
        K_u_f_f_u.resize(nb_u_nodes);
        for (int i = 0; i < nb_u_nodes; ++i){
            K_u_f_f_u[i].resize(nb_u_nodes);
        }
        VectorOperations::mat_product(K_u_f, K_f_u, K_u_f_f_u);

        //Add K_u_f_f_u + K_u_u = Sigma_not_inv
        for (int i = 0; i < nb_u_nodes; ++i){
            for (int j = 0; j <= i; ++j){
                L[i][j] +=  K_u_f_f_u[i][j];
                //Nugget
                if(i==j){
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
        for (int i = 0; i < nb_gp_nodes; ++i){
            noisy_values[i] = scaled_function_values[i] * 1.0/pow( gp_noise.at(i) / 2e0 + noise_regularization, 2e0 );
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

    }else{
        GaussianProcess::build(nodes, values, noise);
    }
    return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void ApproxGaussianProcessSoR::update ( std::vector<double> const &x,
                               double &value,
                               double &noise )
{

    int nb_u_nodes = (gp_nodes.size())*u_ratio;
    //std::cout << "In update Gaussian with [" << gp_nodes.size()+1 << "," << nb_u_nodes <<"]" << std::endl;
    if(nb_u_nodes >= min_nb_u_nodes) {
        //std::cout << "#Update" << std::endl;
        std::vector< std::vector<double> > temp_nodes;
        temp_nodes.resize(gp_nodes.size());
        std::vector< double > temp_values;
        std::vector< double > temp_noise;
        for(int i = 0; i < gp_nodes.size(); ++i){
            for(int j = 0; j < gp_nodes.at(i).size(); ++j){
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
    }else{
        GaussianProcess::update(x, value, noise);
    }
    return;
}
//--------------------------------------------------------------------------------
void ApproxGaussianProcessSoR::evaluate ( std::vector<double> const &x,
                                 double &mean, double &variance )
{
    if(u_idx.size()>0) {

        int nb_u_nodes = u_idx.size();

        K0.clear();
        K0.resize(nb_u_nodes);
        for (int i = 0; i < nb_u_nodes; i++){
            K0.at(i) = evaluate_kernel(x, gp_nodes[u_idx[i]]);
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
    }else{
        GaussianProcess::evaluate(x, mean, variance);
    }
    return;

}

const std::vector<int> ApproxGaussianProcessSoR::get_induced_indices() const {
    return u_idx;
}

void ApproxGaussianProcessSoR::get_induced_nodes(std::vector< std::vector<double> > &induced_nodes) const {
    //assert(induced_nodes.size()==u_idx.size());
    for(int i = 0; i < u_idx.size(); ++i){
        //for(int j = 0; j < gp_nodes[u_idx[i]].size(); ++j){
        induced_nodes.push_back(gp_nodes[u_idx[i]]);
        //}
    }
    return;
}
