#include "GaussianProcess.hpp"
#include "math.h"
#include <iostream>
#include <cassert>

//--------------------------------------------------------------------------------
GaussianProcess::GaussianProcess ( int n ) : 
  TriangularMatrixOperations ( n ),
  dim( n )
{
  lb.reserve( n+1 );
  ub.reserve( n+1 );
  gp_parameters.reserve(n+1);
  for (int i = 0; i < n+1; i++) {
    lb.push_back( 0e0  );
    ub.push_back( 10e0 );
    gp_parameters.push_back( -1e0 );
  }
  nb_gp_nodes = 0;
  gp_pointer = NULL;
} 
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double GaussianProcess::evaluate_kernel ( std::vector<double> const &x, 
                                          std::vector<double> const &y )
{
  dist = 0e0;
  for ( int i = 0; i < dim; ++i )
    dist += pow( (x.at(i) - y.at(i)), 2e0) /  gp_parameters.at( i+1 );
  kernel_evaluation = exp(-dist / 2e0 );
  
  return kernel_evaluation * gp_parameters.at( 0 ) ;
}
//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------
double GaussianProcess::evaluate_kernel ( std::vector<double> const &x, 
                                          std::vector<double> const &y,
                                          std::vector<double> const &p )
{
  dist = 0e0;
  for ( int i = 0; i < dim; ++i )
    dist += pow( (x.at(i) - y.at(i)), 2e0) /  p.at( i+1 );
  kernel_evaluation = exp(-dist / 2e0 );
  
  return kernel_evaluation * p.at( 0 ) ;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double GaussianProcess::d_evaluate_kernel ( std::vector<double> const &x, 
                                            std::vector<double> const &y, 
                                            std::vector<double> const &p, int j )
{
  dist = 0e0;
  for ( int i = 0; i < dim; ++i )
    dist += pow( (x.at(i) - y.at(i)), 2e0) /  p.at(i+1);
  kernel_evaluation = exp(-dist / 2e0 );
  if ( j == 0 ) {
    return kernel_evaluation;
  } else {
    kernel_evaluation *= 2e0*pow( (x.at(j-1) - y.at(j-1)), 2e0) / pow( p.at(j-1), 2e0);
    return kernel_evaluation * p.at(0) ;
  }

}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void GaussianProcess::build ( std::vector< std::vector<double> > const &nodes,
                              Eigen::VectorXd const &values,
                              Eigen::VectorXd const &noise ) 
{
    nb_gp_nodes = nodes.size();
    gp_nodes.clear();
    for ( int i = 0; i < nb_gp_nodes; ++i )
      gp_nodes.push_back ( nodes.at(i) );
    gp_noise = noise;
    gp_function_values = values;

    auto minmax = std::minmax_element(values.data(), values.data()+nb_gp_nodes-1);

    min_function_value = values((minmax.first - values.data()));
    max_function_value = values((minmax.second - values.data()));

    L.clear();
    L.resize( nb_gp_nodes );

    for (int i = 0; i < nb_gp_nodes; i++) {
      for (int j = 0; j <= i; j++)
        L.at(i).push_back (evaluate_kernel( gp_nodes[i], gp_nodes[j] ) );
      L.at(i).at(i) += pow( gp_noise(i) / 2e0, 2e0 );
    }
  
    CholeskyFactorization::compute( L, pos, rho, nb_gp_nodes );
    assert( pos == 0 );
    
    scaled_function_values.resize(nb_gp_nodes);
    for (int i = 0; i < nb_gp_nodes; i++) {
      scaled_function_values.at(i) = values(i) - min_function_value;
      scaled_function_values.at(i) /= 5e-1*( max_function_value-min_function_value );
      scaled_function_values.at(i) -= 1e0;
    }

    alpha = scaled_function_values;
    forward_substitution( L, alpha );
    backward_substitution( L, alpha );
  
    return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void GaussianProcess::update ( std::vector<double> const &x, 
                               double &value, 
                               double &noise )
{
  K0.resize( nb_gp_nodes );
  nb_gp_nodes += 1;
  gp_function_values.conservativeResize( nb_gp_nodes );
  gp_noise.conservativeResize( nb_gp_nodes );
  gp_nodes.push_back( x );
  gp_function_values( nb_gp_nodes-1 ) = value;
  scaled_function_values.push_back ( ( value -  min_function_value ) / 
                                     ( 5e-1*( max_function_value-min_function_value ) ) - 1e0 );
  gp_noise( nb_gp_nodes-1 ) = noise;

   
  for (int i = 0; i < nb_gp_nodes-1; i++) 
    K0.at(i) = evaluate_kernel( gp_nodes[i], x );

  forward_substitution( L, K0 );

  L.push_back ( K0 );
  L.at(L.size()-1).push_back(
    sqrt( evaluate_kernel( x, x ) + pow( gp_noise(nb_gp_nodes-1) / 2e0, 2e0 ) - 
          VectorOperations::dot_product(K0, K0) ) );
        
  alpha = scaled_function_values;
  forward_substitution( L, alpha );
  backward_substitution( L, alpha );
    
  return;
}
//--------------------------------------------------------------------------------
 
//--------------------------------------------------------------------------------
void GaussianProcess::evaluate ( std::vector<double> const &x,
                                 double &mean, double &variance ) 
{

  K0.resize( nb_gp_nodes );
    
  for (int i = 0; i < nb_gp_nodes; i++)
    K0.at(i) = evaluate_kernel( gp_nodes[i], x );

  mean = VectorOperations::dot_product(K0, alpha) + 1e0;
  mean *= 5e-1*( max_function_value-min_function_value );
  mean += min_function_value;

/*
  mean = K0.dot(alpha) + 1e0;
  mean *= 5e-1*( max_function_value-min_function_value );
  mean += min_function_value;
*/        
  forward_substitution( L, K0 );
  
  variance = evaluate_kernel( x, x ) - VectorOperations::dot_product(K0, K0);

  return;

}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void GaussianProcess::estimate_hyper_parameters ( std::vector< std::vector<double> > const &nodes,
                                                  Eigen::VectorXd const &values,
                                                  Eigen::VectorXd const &noise ) 
{
  nb_gp_nodes = nodes.size();
  gp_nodes.clear();
  for ( int i = 0; i < nb_gp_nodes; ++i )
    gp_nodes.push_back ( nodes.at(i) );
  gp_noise = noise;
  gp_function_values = values;

//  dK.resize( nb_gp_nodes, nb_gp_nodes);
  gp_pointer = this;

  auto minmax = std::minmax_element(values.data(), values.data()+nb_gp_nodes-1);
 // auto minmax = std::minmax_element(values.begin(), values.end());
  min_function_value = values((minmax.first - values.data()));
  max_function_value = values((minmax.second - values.data()));

  L.clear();
  L.resize( nb_gp_nodes );
  for (int i = 0; i < nb_gp_nodes; i++) 
    L.at(i).resize( i+1 ); 
 
  scaled_function_values.resize(nb_gp_nodes);
  for (int i = 0; i < nb_gp_nodes; i++) {
    scaled_function_values.at(i) = values(i) - min_function_value;
    scaled_function_values.at(i) /= 5e-1*( max_function_value-min_function_value );
    scaled_function_values.at(i) -= 1e0;
  }

  double optval;    
  //adjust those settings to optimize GP approximation
  //--------------------------------------------------
  double max_noise = 0e0;
  for (int i = 0; i < nb_gp_nodes; i++) {
    if (gp_noise( i ) > max_noise)
      max_noise = gp_noise( i );
  }
  lb[0] = 1e-1; // * pow(1000e0 * max_noise / 2e0, 2e0);
  ub[0] = 1e1;// * pow(1000e0 * max_noise / 2e0, 2e0);
  for (int i = 0; i < dim; i++) {
    lb[i+1] = 1e-2;// * delta;
    ub[i+1] = 1e2;// * delta;   
  }

  if (gp_parameters[0] < 0e0) {
    gp_parameters[0] = 2e0;//(lb[0]*5e-1 + 5e-1*ub[0]);
    for (int i = 1; i < dim+1; i++) {
      gp_parameters[i] = 4e0;//(lb[i]*5e-1 + 5e-1*ub[i]);
    }
  } else {
    for (int i = 0; i < dim+1; i++) {
      if ( gp_parameters[i] <= lb[i] ) gp_parameters[i] = 1e0;//1.1 * lb[i];
      if ( gp_parameters[i] >= ub[i] ) gp_parameters[i] = 0.9 * ub[i];
    }
  }
  //--------------------------------------------------

  //initialize optimizer from NLopt library
//  nlopt::opt opt(nlopt::LD_CCSAQ, dim+1);
  nlopt::opt opt(nlopt::LN_BOBYQA, dim+1);
//  nlopt::opt opt(nlopt::GN_DIRECT, dim+1);

  //opt = nlopt_create(NLOPT_LN_COBYLA, dim+1);
  opt.set_lower_bounds( lb );
  opt.set_upper_bounds( ub );
    
  opt.set_max_objective( parameter_estimation_objective, gp_pointer);

  opt.set_xtol_abs(1e-6);
  opt.set_xtol_rel(1e-6);
//set timeout to NLOPT_TIMEOUT seconds
  opt.set_maxtime(0.5);
  //perform optimization to get correction factors
  int exitflag = opt.optimize(gp_parameters, optval);

  //std::cout << "exitflag = "<< exitflag<<std::endl;
  std::cout << "OPTVAL .... " << optval << std::endl;
  for ( int i = 0; i < gp_parameters.size(); ++i )
    std::cout << "gp_param = " << gp_parameters[i] << std::endl;
  std::cout << std::endl;
 
      
  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
double GaussianProcess::parameter_estimation_objective(std::vector<double> const &x, 
                                                       std::vector<double> &grad, 
                                                       void *data) 
{

  GaussianProcess *d = reinterpret_cast<GaussianProcess*>(data);

  double noise_regularization = 0e-2;

  for (int i = 0; i < d->nb_gp_nodes; i++) {
    for (int j = 0; j <= i; j++)
      d->L.at(i).at(j) = d->evaluate_kernel( d->gp_nodes[i], d->gp_nodes[j], x );
    d->L.at(i).at(i) += pow( d->gp_noise(i) / 2e0 + noise_regularization, 2e0 );
  }


  d->CholeskyFactorization::compute( d->L, d->pos, d->rho, d->nb_gp_nodes );
  assert(d->pos == 0);
  d->alpha = d->scaled_function_values;
  d->forward_substitution( d->L, d->alpha );
  d->backward_substitution( d->L, d->alpha );
  double result = -0.5*d->VectorOperations::dot_product(d->scaled_function_values, d->alpha) + 
                  -0.5*((double)d->nb_gp_nodes)*log(6.28);
  for (int i = 0; i < d->nb_gp_nodes; ++i)
    result -= 0.5*log(d->L.at(i).at(i)); 


/*
  //Eigen::ColPivHouseholderQR<Eigen::MatrixXd> S;
  d->S = (d->L).fullPivHouseholderQr();
  //d->S = (d->L).colPivHouseholderQr();
  d->K0 = (d->S).solve(d->scaled_function_values);


  double result = -0.5*(d->scaled_function_values).dot(d->K0) -
                  0.5*log((d->S).absDeterminant() + 1e-16) -
                  0.5*((double)d->nb_gp_nodes)*log(6.28); 


  if ( !grad.empty() ) {
    for (int k = 0; k < d->dim+1; ++k) {
      for (int i = 0; i < d->nb_gp_nodes; i++) {
        for (int j = 0; j <= i; j++) {
          d->dK(i, j) = d->d_evaluate_kernel( d->gp_nodes[i], d->gp_nodes[j], x, k);
          if (i != j) d->dK(j, i) = d->dK(i, j);
        }
      }
      grad[k] = 0.5*((d->K0*(d->K0.transpose()))*d->dK - d->S.solve(d->dK)).trace() - 
                0.5*((double)d->nb_gp_nodes)*0e-6;
    }
  }
*/
  return result;
    
}
//--------------------------------------------------------------------------------
