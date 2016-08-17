//
// Created by friedrich on 16.08.16.
//

#include <ApproximatedGaussianProcess.hpp>
#include <assert.h>
#include <random>
#include <algorithm>
#include <iostream>

//--------------------------------------------------------------------------------
ApproximatedGaussianProcess::ApproximatedGaussianProcess(int n, double &delta_input):GaussianProcess(n, delta_input) {}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void ApproximatedGaussianProcess::build ( std::vector< std::vector<double> > const &nodes,
                              std::vector<double> const &values,
                              std::vector<double> const &noise )
{

    int nb_u_nodes = nodes.size()*u_ratio;
    std::cout << "In Approximatin Gaussian with [" << nodes.size() << "," << nb_u_nodes <<"]" << std::endl;
    if(nb_u_nodes > 2) {
        nb_gp_nodes = nodes.size();
        gp_nodes.clear();
        gp_noise.clear();
        for ( int i = 0; i < nb_gp_nodes; ++i ) {
            gp_nodes.push_back(nodes.at(i));
            gp_noise.push_back(noise.at(i));
        }

        std::cout << "APPROXIMATING GAUSSIAN WITH " << nb_u_nodes << " POINTS!! Total points: "<< nb_gp_nodes << std::endl;
        //Set distribution for sampling the indices for the u samples
        std::random_device rd;
        int random_seed = rd(); //rd();
        std::mt19937 random_generator(random_seed);
        std::vector<double> nodes_weights_vector;
        //equal weights : double nodes_weights = 100.0 / nodes.size();
        //std::fill(nodes_weights_vector.begin(), nodes_weights_vector.end(), nodes_weights);
        double prefactor = 200.0/( (nb_gp_nodes + 1)*nb_gp_nodes);
        for(int i = 1; i <= nb_gp_nodes; ++i){
            nodes_weights_vector.push_back(prefactor * i);
        }
        std::discrete_distribution<> d(nodes_weights_vector.begin(), nodes_weights_vector.end());

        //Sample the indices
        std::vector<int> u_idx;
        int random_draw = -1;
        std::vector<int>::iterator it;
        random_draw = d(random_generator);
        u_idx.push_back(random_draw);
        std::cout << "Drawing";
        while (u_idx.size() < nb_u_nodes) {
            random_draw = d(random_generator);
            it = std::find(u_idx.begin(), u_idx.end(), random_draw);
            if (it == u_idx.end()) {
                u_idx.push_back(random_draw);
            }
            std::cout << ".";
        }
        std::cout << std::endl;
        std::sort(u_idx.begin(), u_idx.end());
        std::cout << "Sampling done with";
        for(int i = 0; i < u_idx.size(); ++i){
            std::cout << " " << u_idx.at(i);
        }
        std::cout << std::endl;

        //Create u vector
        u.resize(nb_u_nodes);
        for (int i = 0; i < nb_u_nodes; ++i) {
            for (int j = 0; j < nodes.at(u_idx.at(i)).size(); ++j) {
                u.at(i).push_back(nodes.at(u_idx.at(i)).at(j));
            }
        }
        std::cout << "Creating u done!" << std::endl;

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
        std::cout << "K_u_f and K_f_u done!" << std::endl;

        //Set up matrix K_u_u
        K_u_u.clear();
        K_u_u.resize(nb_u_nodes);
        for (int i = 0; i < nb_u_nodes; ++i) {
            for (int j = 0; j < nb_u_nodes; ++j) {
                K_u_u.at(i).push_back(evaluate_kernel(u.at(i), u.at(j)));
            }
        }
        std::cout << "K_u_u done!" << std::endl;

        //Pick noise values from noise vector using u_idx
        std::vector<double> noise_u_minus_squared(u_idx.size());
        for (int i = 0; i < u_idx.size(); ++i) {
            noise_u_minus_squared[i] = 1;//1 / (pow(noise.at(u_idx[i]) / 2e0 + noise_regularization, 2e0));
        }
        std::cout << "Set up Noise done!" << std::endl;

        //Multiply K_u_f and K_f_u
        std::vector<std::vector<double> > K_u_f_f_u;
        K_u_f_f_u.resize(nb_u_nodes);
        for (int i = 0; i < nb_u_nodes; ++i) {
            K_u_f_f_u.at(i).resize(nb_u_nodes);
        }
        VectorOperations::mat_product(K_u_f, K_f_u, K_u_f_f_u);
        std::cout << "K_u_f*K_f_u done!" << std::endl;

        //Multiply noise to K_u_f_f_u
        for (int i = 0; i < nb_u_nodes; ++i) {
            K_u_f_f_u[i][i] *= noise_u_minus_squared.at(i);
        }
        std::cout << "sigma*K_u_f_f_u done!" << std::endl;

        //Add K_u_u to K_u_f_f_u which is L
        //for(int i = 0; i < L.size(); ++i){
        //    for(int j = 0; j < L.at(i).size(); ++j){
        //        std::cout <<  "[" << i << "," << j << "]" << ":" << L[i][j];
        //    }
        //    std::cout << std::endl;
        //}
        L.clear();
        L.resize(nb_u_nodes);
        std::cout << "Resize L done!" << std::endl;
        for (int i = 0; i < nb_u_nodes; ++i) {
            for (int j = 0; j <= i; ++j) {
                L.at(i).push_back(K_u_f_f_u.at(i).at(j) + K_u_u.at(i).at(j));
            }
        }
        std::cout << " L = K_u_u + K_u_f_f_u done!" << std::endl;

        for (int i = 0; i < L.size(); ++i) {
            for (int j = 0; j < L.at(i).size(); ++j) {
                std::cout << L.at(i).at(j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Print L done!" << std::endl;

        CholeskyFactorization::compute(L, pos, rho, nb_u_nodes);
        assert(pos == 0);
        std::cout << "Cholesky done!" << std::endl;

        scaled_function_values.resize(nb_gp_nodes);
        for (int i = 0; i < nb_gp_nodes; i++) {
            scaled_function_values.at(i) = values.at(i);
            //      scaled_function_values.at(i) = values.at(i) - min_function_value;
            //      scaled_function_values.at(i) /= 5e-1*( max_function_value-min_function_value );
            //      scaled_function_values.at(i) -= 1e0;
        }

        //Compute K_u_f * y = alpha
        alpha.clear();
        alpha.resize(nb_u_nodes);
        VectorOperations::mat_vec_product(K_u_f, scaled_function_values, alpha);
        std::cout << "K_u_f * y  = alpha done!" << std::endl;

        //Compute Sigma*alpha
        forward_substitution(L, alpha);
        backward_substitution(L, alpha);
        std::cout << "Forward-Backward Substitution done!" << std::endl;
    }else{
        GaussianProcess::build(nodes, values, noise);
    }
    return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void ApproximatedGaussianProcess::update ( std::vector<double> const &x,
                               double &value,
                               double &noise )
{
    if(u.size() > 0) {
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
void ApproximatedGaussianProcess::evaluate ( std::vector<double> const &x,
                                 double &mean, double &variance )
{
    if(u.size()>0){
        int no_u_nodes = u.size();
        K0.resize( no_u_nodes );

        for (int i = 0; i < no_u_nodes; i++)
            K0.at(i) = evaluate_kernel( x, u[i] );

        double noise = 0.02; //TODO this is not correct!
        mean = 1/noise * VectorOperations::dot_product(K0, alpha);
    //  mean = VectorOperations::dot_product(K0, alpha) + 1e0;
    //  mean *= 5e-1*( max_function_value-min_function_value );
    //  mean += min_function_value;

        forward_substitution( L, K0 );

        variance = VectorOperations::dot_product(K0, K0);
    }else{
        GaussianProcess::evaluate(x, mean, variance);
    }
    return;

}