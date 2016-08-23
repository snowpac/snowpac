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
    std::cout << "In Build Gaussian with [" << nodes.size() << "," << nb_u_nodes <<"]" << std::endl;
    if(nb_u_nodes >= min_nb_u_nodes) {
        nb_gp_nodes = nodes.size();
        gp_nodes.clear();
        gp_noise.clear();
        for ( int i = 0; i < nb_gp_nodes; ++i ) {
            gp_nodes.push_back(nodes.at(i));
            gp_noise.push_back(noise.at(i));
        }
        u_idx.clear();
        std::cout << "APPROXIMATING GAUSSIAN WITH " << nb_u_nodes << " POINTS!! Total points: "<< nb_gp_nodes << std::endl;
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
        std::vector< std::vector<double> > u;
        u.resize(nb_u_nodes);
        for (int i = 0; i < nb_u_nodes; ++i) {
            for (int j = 0; j < nodes.at(u_idx.at(i)).size(); ++j) {
                u.at(i).push_back(nodes.at(u_idx.at(i)).at(j));
            }
        }
        //std::cout << "Creating u done!" << std::endl;

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

        //Set up matrix K_u_u
        K_u_u.clear();
        K_u_u.resize(nb_u_nodes);
        for (int i = 0; i < nb_u_nodes; ++i) {
            for (int j = 0; j <= i; ++j) {
                K_u_u.at(i).push_back(evaluate_kernel(u.at(i), u.at(j)));
                if(i==j){
                    K_u_u[i][j] += 0.0001;
                }
            }
        }
        //std::cout << "K_u_u done!" << std::endl;
        //VectorOperations::print_matrix(K_u_u);

        //Compute Cholesky of K_u_u
        CholeskyFactorization::compute(K_u_u, pos, rho, nb_u_nodes);
        assert(pos == 0);
        //Compute L^-1 K_u_f <=> L gamma = K_u_f <=> gamma = L_inv_K_u_f_transpose^T
        std::vector< std::vector<double> > L_inv_K_u_f_transpose;
        L_inv_K_u_f_transpose.clear();
        L_inv_K_u_f_transpose.resize(nb_gp_nodes);
        for(int i = 0; i < nb_gp_nodes; ++i){
            L_inv_K_u_f_transpose.at(i).resize(nb_u_nodes);
        }
        for(int i = 0; i < nb_gp_nodes; ++i){
            TriangularMatrixOperations::forward_substitution_not_inplace(K_u_u, K_f_u.at(i), L_inv_K_u_f_transpose.at(i));
        }

        //std::cout << "K_u_u:" << std::endl;
        //VectorOperations::print_matrix(K_u_u);
        //std::cout << "K_u_f:" << std::endl;
        //VectorOperations::print_matrix(K_u_f);
        //std::cout << "L_inv_K_u_f_transpose_after_forward:" << std::endl;
        //VectorOperations::print_matrix(L_inv_K_u_f_transpose);

        //Transpose L => L^T
        std::vector< std::vector<double> > L_transpose;
        L_transpose.clear();
        L_transpose.resize(nb_u_nodes);
        for(int i = 0; i < nb_u_nodes; ++i){
            L_transpose.at(i).resize(nb_u_nodes);
        }
        //Compute L^T beta = gamma <=> gamma = (L^T^(-1) * L^(-1) * K_u_f)^T
        VectorOperations::mat_transpose(K_u_u, L_transpose);
        //std::cout << "L_tranpose:" << std::endl;
        //VectorOperations::print_matrix(L_transpose);

        for(int i = 0; i < nb_gp_nodes; ++i){
            TriangularMatrixOperations::backward_substitution(K_u_u, L_inv_K_u_f_transpose.at(i));
        }
        //std::cout << "L_inv_K_u_f_transpose_after_backward:" << std::endl;
        //VectorOperations::print_matrix(L_inv_K_u_f_transpose);

        L_inv_T_L_inv_K_u_f.clear();
        L_inv_T_L_inv_K_u_f.resize(nb_u_nodes);
        for(int i = 0; i < nb_u_nodes; ++i){
            L_inv_T_L_inv_K_u_f.at(i).resize(nb_gp_nodes);
        }
        VectorOperations::mat_transpose(L_inv_K_u_f_transpose, L_inv_T_L_inv_K_u_f);

        //Finally compute Q_f_f =: L
        L.clear();
        L.resize(nb_gp_nodes);
        for(int i = 0; i < nb_gp_nodes; ++i){
            L.at(i).resize(nb_gp_nodes);
        }
        VectorOperations::mat_product(K_f_u, L_inv_T_L_inv_K_u_f, L);

        //std::cout << "K_f_u:" << std::endl;
        //VectorOperations::print_matrix(K_f_u);
        //std::cout << "L_inv_T_L_inv_K_u_f:" << std::endl;
        //VectorOperations::print_matrix(L_inv_T_L_inv_K_u_f);
        //std::cout << "L:" << std::endl;
        //VectorOperations::print_matrix(L);

        //Add noise to L := Q_f_f
        for (int i = 0; i < nb_gp_nodes; ++i) {
            L[i][i] += pow( gp_noise.at(i) / 2e0 + noise_regularization, 2e0 );
        }
        //std::cout << "L := Q_f_f + sigma^2*I done!" << std::endl;
        //VectorOperations::print_matrix(L);

        CholeskyFactorization::compute(L, pos, rho, nb_gp_nodes);
        for(int i = 0; i < L.size(); ++i){
            for(int j = i+1; j < L[i].size(); ++j){
                L[i][j] = 0;
            }
        }
        assert(pos == 0);
        //std::cout << "Cholesky done!" << std::endl;
        //VectorOperations::print_matrix(L);

        scaled_function_values.resize(nb_gp_nodes);
        for (int i = 0; i < nb_gp_nodes; i++) {
            scaled_function_values.at(i) = values.at(i);
            //      scaled_function_values.at(i) = values.at(i) - min_function_value;
            //      scaled_function_values.at(i) /= 5e-1*( max_function_value-min_function_value );
            //      scaled_function_values.at(i) -= 1e0;
        }

        //Compute Q_f_f * y = alpha
        alpha = scaled_function_values;
        //VectorOperations::print_vector(alpha);
        forward_substitution(L, alpha);
        backward_substitution(L, alpha);
        //VectorOperations::print_vector(alpha);
        //std::cout << "Forward-Backward Substitution done!" << std::endl;

        //std::cout << "Final L_inv_T_L_inv_K_u_f:" << std::endl;
        //VectorOperations::print_matrix(L_inv_T_L_inv_K_u_f);
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

    int nb_u_nodes = (gp_nodes.size()+1)*u_ratio;
    std::cout << "In update Gaussian with [" << gp_nodes.size()+1 << "," << nb_u_nodes <<"]" << std::endl;
    if(nb_u_nodes >= min_nb_u_nodes) {
        std::cout << "#Update" << std::endl;
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
    if(u_idx.size()>0) {

        int nb_u_nodes = u_idx.size();

        K0.clear();
        K0.resize(nb_u_nodes);

        std::vector< double > Q_star_star;
        Q_star_star.resize(nb_u_nodes);
        for (int i = 0; i < nb_u_nodes; i++){
            K0.at(i) = evaluate_kernel(x, gp_nodes[u_idx[i]]);
            Q_star_star.at(i) = K0.at(i);
        }

        //std::cout << "K0" << std::endl;
        //VectorOperations::print_vector(K0);
        std::vector< double > Q_star_f;
        Q_star_f.resize(nb_gp_nodes);
        //std::cout << "L_inv_T_L_inv_K_u_f:" << std::endl;
        //VectorOperations::print_matrix(L_inv_T_L_inv_K_u_f);

        VectorOperations::vec_mat_product(L_inv_T_L_inv_K_u_f, K0, Q_star_f);
        //std::cout << "Q_star_f:" << std::endl;
        //VectorOperations::print_vector(Q_star_f);

        mean = VectorOperations::dot_product(Q_star_f, alpha);
        //std::cout << "Mean: " << mean << std::endl;
    //  mean = VectorOperations::dot_product(K0, alpha) + 1e0;
    //  mean *= 5e-1*( max_function_value-min_function_value );
    //  mean += min_function_value;
        //std::cout << "L:" << std::endl;
        std::vector<double> beta(Q_star_f.size());
        for(int i = 0; i < Q_star_f.size(); ++i){
            beta[i] = Q_star_f[i];
        }
        //VectorOperations::print_matrix(L);
        forward_substitution( L, beta );
        //std::cout << "Forward L*beta = Q_star_f" << std::endl;
        //VectorOperations::print_vector(beta);
        backward_substitution( L, beta );
        //std::cout << "Backward L^T*beta = beta (inplace)" << std::endl;
        //VectorOperations::print_vector(beta);
        double second_term_variance = VectorOperations::dot_product(Q_star_f, beta);
        //std::cout << "Second term: " << second_term_variance << std::endl;
        //Compute Q_*_*
        TriangularMatrixOperations::forward_substitution(K_u_u, K0);
        //std::cout << "forward K_u_u" << std::endl;
        //VectorOperations::print_matrix(K_u_u);
        //std::cout << "forward K0" << std::endl;
        //VectorOperations::print_vector(K0);
        TriangularMatrixOperations::backward_substitution(K_u_u, K0);
        //std::cout << "backward K_u_u" << std::endl;
        //VectorOperations::print_matrix(K_u_u);
        //std::cout << "backward K0" << std::endl;
        //VectorOperations::print_vector(K0);
        //std::cout << "FirstTerm: " << VectorOperations::dot_product(Q_star_star, K0) << std::endl;
        variance = VectorOperations::dot_product(Q_star_star, K0) - second_term_variance;
        //std::cout << "Variance: " << variance << std::endl;
    }else{
        GaussianProcess::evaluate(x, mean, variance);
    }
    return;

}

const std::vector<int> &ApproximatedGaussianProcess::getU_idx() const {
    return u_idx;
}
