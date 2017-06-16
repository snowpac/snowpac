#include "GaussianProcess.hpp"
#include "math.h"
#include <iostream>
#include <cassert>

//--------------------------------------------------------------------------------
GaussianProcess::GaussianProcess ( int n, double &delta_input ) : 
  TriangularMatrixOperations ( n ),
  dim( n )
{
  lb.reserve( n+1 );
  ub.reserve( n+1 );
  gp_parameters.reserve(n+1);
  for (int i = 0; i < n+1; i++) {
    lb.push_back( 0e0  );
    ub.push_back( 10e0 );
    gp_parameters.push_back( 1e0 );
  }
  nb_gp_nodes = 0;
  gp_pointer = NULL;
  delta = &delta_input;
} 
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
GaussianProcess::GaussianProcess ( int n, double &delta_input, std::vector<double> gp_parameters_input ) : 
  TriangularMatrixOperations ( n ),
  dim( n )
{
  if(gp_parameters_input.size() != n+1){
    std::cout << "gp_parameters_input wrong size! Should be: " << n+1 << ", is: " << gp_parameters_input.size() << std::endl;
    exit(-1);
  }

  lb.reserve( n+1 );
  ub.reserve( n+1 );
  gp_parameters.reserve(n+1);
  for (int i = 0; i < n+1; i++) {
    lb.push_back( 0e0  );
    ub.push_back( 10e0 );
    gp_parameters.push_back( gp_parameters_input[i] );
  }
  nb_gp_nodes = 0;
  gp_pointer = NULL;
  delta = &delta_input;
} 
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
double GaussianProcess::evaluate_kernel ( std::vector<double> const &x,
                                          std::vector<double> const &y )
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

bool GaussianProcess::test_for_parameter_estimation(const int& nb_values,
                                                const int& update_interval_length,
                                                const int& next_update,
                                                const std::vector<int>& update_at_evaluations){
  bool do_parameter_estimation = false;

  if ( nb_values >= next_update && update_interval_length > 0 ) {
    do_parameter_estimation = true;
    return do_parameter_estimation;
  } 
  if ( update_at_evaluations.size( ) > 0 ) {
    if ( nb_values >= update_at_evaluations[0] ) {
      do_parameter_estimation = true;
    }
  }

  return do_parameter_estimation;
}


//--------------------------------------------------------------------------------
void GaussianProcess::build ( std::vector< std::vector<double> > const &nodes,
                              std::vector<double> const &values,
                              std::vector<double> const &noise ) 
{
    //std::cout << "GP build [" << nodes.size() << "]" << std::endl;
    //std::cout << "With Parameters: " << std::endl;
    //for ( int i = 0; i < dim+1; ++i )
    // std::cout << "gp_param = " << gp_parameters[i] << std::endl;
    //std::cout << std::endl;

    nb_gp_nodes = nodes.size();
    gp_nodes.clear();
    //gp_noise.clear();
    for ( int i = 0; i < nb_gp_nodes; ++i ) {
      gp_nodes.push_back ( nodes.at(i) );
      //gp_noise.push_back( noise.at(i) );
    }


//    auto minmax = std::minmax_element(values.begin(), values.end());
//    min_function_value = values.at((minmax.first - values.begin()));
//    max_function_value = values.at((minmax.second - values.begin()));

//    std::cout << "[ " << min_function_value << ", " << max_function_value << " ]" << std::endl;

    L.clear();
    L.resize( nb_gp_nodes );

    for (int i = 0; i < nb_gp_nodes; i++) {
      for (int j = 0; j <= i; j++)
        L.at(i).push_back (evaluate_kernel( gp_nodes[i], gp_nodes[j] ) );
      L.at(i).at(i) += pow( noise.at(i) / 2e0 + noise_regularization, 2e0 );
    }
  
    CholeskyFactorization::compute( L, pos, rho, nb_gp_nodes );
    assert( pos == 0 );
    
    scaled_function_values.resize(nb_gp_nodes);
    for (int i = 0; i < nb_gp_nodes; i++) {
      scaled_function_values.at(i) = values.at(i);
//      scaled_function_values.at(i) = values.at(i) - min_function_value;
//      scaled_function_values.at(i) /= 5e-1*( max_function_value-min_function_value );
//      scaled_function_values.at(i) -= 1e0;
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
  //std::cout << "GP update [" << nb_gp_nodes+1 << "]" << std::endl;
  K0.resize( nb_gp_nodes );
  nb_gp_nodes += 1;
  gp_nodes.push_back( x );
  //gp_noise.push_back( noise );
  scaled_function_values.push_back ( value );
//  scaled_function_values.push_back ( ( value -  min_function_value ) / 
//                                     ( 5e-1*( max_function_value-min_function_value ) ) - 1e0 );

  
   
  for (int i = 0; i < nb_gp_nodes-1; i++) 
    K0.at(i) = evaluate_kernel( gp_nodes[i], x );

  forward_substitution( L, K0 );

  L.push_back ( K0 );

  L.at(L.size()-1).push_back(
    sqrt( evaluate_kernel( x, x ) + pow( noise / 2e0 + noise_regularization, 2e0 ) - 
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

  mean = VectorOperations::dot_product(K0, alpha);
//  mean = VectorOperations::dot_product(K0, alpha) + 1e0;
//  mean *= 5e-1*( max_function_value-min_function_value );
//  mean += min_function_value;

  forward_substitution( L, K0 );
  
  variance = evaluate_kernel( x, x ) - VectorOperations::dot_product(K0, K0);

  //std::cout << "GP evalute [" << gp_nodes.size() <<"] mean,variance " << mean << ", " << variance << std::endl;
  return;

}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void GaussianProcess::estimate_hyper_parameters ( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise ) 
{
  //std::cout << "GP Estimator" << std::endl;
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
  /* 
  auto minmax = std::minmax_element(values.begin(), values.end());
  min_function_value = values.at((minmax.first - values.begin()));
  max_function_value = fabs(values.at((minmax.second - values.begin())));
  if ( fabs(min_function_value) > max_function_value )
    max_function_value = fabs( min_function_value );
  */
  L.clear();
  L.resize( nb_gp_nodes );
  for ( int i = 0; i < nb_gp_nodes; ++i)
    L.at(i).resize( i+1 ); 
 
  scaled_function_values.resize(nb_gp_nodes);
  for ( int i = 0; i < nb_gp_nodes; ++i) {
    scaled_function_values.at(i) = values.at(i);
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
  
  //My Version
  /*lb[0] = 1e-3; 
      ub[0] = 1e3;
      lb[0] = max_function_value - 1e2;
      if ( lb[0] < 1e-3 ) lb[0] = 1e-3;
      ub[0] = max_function_value + 1e2; 
      if ( ub[0] > 1e3 ) ub[0] = 1e3;
      if ( ub[0] <= lb[0]) lb[0] = 1e-3;
  double delta_threshold = *delta;
  if (delta_threshold < 1e-2) delta_threshold = 1e-2;
  for (int i = 0; i < dim; ++i) {
      lb[i+1] = 1e-2 * delta_threshold; // 1e1
      ub[i+1] = 2.0 * delta_threshold; // 1e2
  }*/
  

  //Florians old version1:
  /*lb[0] = 1e-1; 
  ub[0] = 1e1;
  lb[0] = max_function_value - 1e2;
  if ( lb[0] < 1e-2 ) lb[0] = 1e-2;
  ub[0] = max_function_value + 1e2;
  double delta_threshold = *delta;
  if (delta_threshold < 1e-2) delta_threshold = 1e-2;
  for (int i = 0; i < dim; ++i) {
      lb[i+1] = 1e-2 * delta_threshold; // 1e1
      ub[i+1] = 2.0 * delta_threshold; // 1e2
  }*/
  //Florians old version2:
  lb[0] = 1e-1; // * pow(1000e0 * max_noise / 2e0, 2e0);
  ub[0] = 1e1;// * pow(1000e0 * max_noise / 2e0, 2e0);
  double delta_threshold = *delta;
  if (delta_threshold < 1e-2) delta_threshold = 1e-2;
  for (int i = 0; i < dim; ++i) {
      lb[i+1] = 1e1 * delta_threshold;
      ub[i+1] = 1e2 * delta_threshold;
  }

  if (gp_parameters[0] < 0e0) {
    //gp_parameters[0] = max_function_value;
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
  //--------------------------------------------------

  //initialize optimizer from NLopt library
  int dimp1 = dim+1;
//  nlopt::opt opt(nlopt::LD_CCSAQ, dimp1);
//  nlopt::opt opt(nlopt::LN_BOBYQA, dimp1);
//
  nlopt::opt opt(nlopt::GN_DIRECT, dimp1);

  //opt = nlopt_create(NLOPT_LN_COBYLA, dim+1);
  opt.set_lower_bounds( lb );
  opt.set_upper_bounds( ub );
    
  opt.set_max_objective( GaussianProcess::parameter_estimation_objective, gp_pointer);

 // opt.set_xtol_abs(1e-2);
//  opt.set_xtol_rel(1e-2);
//set timeout to NLOPT_TIMEOUT seconds
  opt.set_maxtime(1.0);
  //opt.set_maxeval(10000);
  //perform optimization to get correction factors

    int exitflag=-20;
  try {
    exitflag = opt.optimize(gp_parameters, optval);
  } catch (...) {
    gp_parameters[0] = lb[0]*5e-1 + 5e-1*ub[0];
    for (int i = 1; i < dim+1; ++i) {
      gp_parameters[i] = (lb[i]*5e-1 + 5e-1*ub[i]);
    }
  }

  //std::cout << "exitflag = "<< exitflag<<std::endl;
  //std::cout << "OPTVAL .... " << optval << std::endl;
  //for ( int i = 0; i < gp_parameters.size(); ++i )
  //  std::cout << "gp_param = " << gp_parameters[i] << std::endl;
  //std::cout << std::endl;
 
      
  return;
}
//--------------------------------------------------------------------------------
void GaussianProcess::estimate_hyper_parameters_induced_only ( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise ) 
{ this->estimate_hyper_parameters(nodes, values, noise);}

void GaussianProcess::estimate_hyper_parameters_ls_only ( std::vector< std::vector<double> > const &nodes,
                                                  std::vector<double> const &values,
                                                  std::vector<double> const &noise ) 
{ this->estimate_hyper_parameters(nodes, values, noise);}

//--------------------------------------------------------------------------------
double GaussianProcess::parameter_estimation_objective(std::vector<double> const &x, 
                                                       std::vector<double> &grad, 
                                                       void *data) 
{

  GaussianProcess *d = reinterpret_cast<GaussianProcess*>(data);



  for (int i = 0; i < d->nb_gp_nodes; ++i) {
    for (int j = 0; j <= i; ++j)
      d->L.at(i).at(j) = d->evaluate_kernel( d->gp_nodes[i], d->gp_nodes[j], x );
    d->L.at(i).at(i) += pow( d->gp_noise.at(i) / 2e0 + d->noise_regularization, 2e0 );
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
    
  // std::cout << result << std::endl;


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

const std::vector<std::vector<double>> &GaussianProcess::getGp_nodes() const {
    return gp_nodes;
}

void GaussianProcess::get_induced_nodes(std::vector<std::vector<double> > &) const {
    return;
}

std::vector<double> GaussianProcess::get_hyperparameters(){
    return this->gp_parameters;
}
void GaussianProcess::decrease_nugget(){
  return;
}
bool GaussianProcess::increase_nugget(){
  return true;
}
//--------------------------------------------------------------------------------
