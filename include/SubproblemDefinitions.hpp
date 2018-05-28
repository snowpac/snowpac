#ifndef HSubproblemDefinitions
#define HSubproblemDefinitions

#include "VectorOperations.hpp"
#include <Eigen/Core>
#include <vector>


template<class TSurrogateModel, template<class TSurrogateModel1> class TSubproblemOptimization>
struct SubproblemData {
    TSubproblemOptimization<TSurrogateModel> *me;
    VectorOperations *vo;
//    std::vector<double> vector;
    int constraint_number;
};


template<class TSurrogateModel, template<class TSurrogateModel1> class TSubproblemOptimization>
class SubproblemDefinitions : public VectorOperations {
  public:
    static double opt_trial_point_obj ( std::vector<double> const&, std::vector<double>&, void*);
    static double opt_criticality_measure_obj ( std::vector<double> const &, 
                                                std::vector<double>&, void*); 
    static double opt_restore_feasibility_obj ( std::vector<double> const &, 
                                                std::vector<double>&, void*); 
    static double trustregion_constraint ( std::vector<double> const &, 
                                     std::vector<double>&, void*); 
    static double constraints_for_subproblems ( std::vector<double> const &, 
                                          std::vector<double>&, void*); 
};


//--------------------------------------------------------------------------------
template<class TSurrogateModel, template<class TSurrogateModel1> class TSubproblemOptimization>
double SubproblemDefinitions<TSurrogateModel, TSubproblemOptimization>::opt_trial_point_obj (
  std::vector<double> const &x, std::vector<double> &grad, void* data) 
{
  SubproblemData<TSurrogateModel, TSubproblemOptimization> *d =
    reinterpret_cast<  SubproblemData<TSurrogateModel, TSubproblemOptimization>*>(data);

  if (!grad.empty( )) {
    d->vo->mat_vec_product( (*(d->me->surrogate_models))[0].hessian( ), x, grad );
    d->vo->add ( (*(d->me->surrogate_models))[0].gradient( ), grad );
  }
  return (*(d->me->surrogate_models))[0].evaluate( x );
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
template<class TSurrogateModel, template<class TSurrogateModel1> class TSubproblemOptimization>
double SubproblemDefinitions<TSurrogateModel, TSubproblemOptimization>::opt_criticality_measure_obj (
  std::vector<double> const &x, std::vector<double> &grad, void* data) 
{

  SubproblemData<TSurrogateModel, TSubproblemOptimization> *d =
    reinterpret_cast<  SubproblemData<TSurrogateModel, TSubproblemOptimization>*>(data);

  if (!grad.empty( )) {
    grad = d->me->criticality_gradient;
  }  

  return d->vo->dot_product( x, d->me->criticality_gradient );
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
template<class TSurrogateModel, template<class TSurrogateModel1> class TSubproblemOptimization>
double SubproblemDefinitions<TSurrogateModel, TSubproblemOptimization>::opt_restore_feasibility_obj (
  std::vector<double> const &x, std::vector<double> &grad, void* data) 
{

  SubproblemData<TSurrogateModel, TSubproblemOptimization> *d =
    reinterpret_cast<  SubproblemData<TSurrogateModel, TSubproblemOptimization>*>(data);

//  d->vo->minus( x, d->me->best_point, d->vector );
  std::vector<double> gradient (x.size());
  double objective_value = 0e0;
  double tmp[3];
  double lambda_g = *(d->me->delta);
  if (!grad.empty( )) {
    for (int j = 0; j < x.size( ); ++j)
      grad[ j ] = 0e0;
  }
  tmp[1] = d->vo->dot_product( x, x );
  tmp[2] = pow( *(d->me->delta), 2e0 );

  for ( int i = 0; i < (d->me->surrogate_models)->size()-1; ++i ) {
    tmp[0] = (*(d->me->surrogate_models))[i+1].evaluate( x );
    if ( tmp[0] > 0e0 ) {
      //tmp[0] += d->me->inner_boundary_constant->at(i) * tmp[1] * tmp[2];
      d->vo->mat_vec_product( (*(d->me->surrogate_models))[i+1].hessian( ), x, gradient );
      d->vo->add ((*(d->me->surrogate_models))[i+1].gradient( ), gradient );
      //gradient = (*(d->me->surrogate_models))[i+1].gradient( x );
      objective_value += pow( tmp[0] , 2e0) + lambda_g * tmp[0];
      if (!grad.empty( )) {
        for (int j = 0; j < x.size( ); j++){ 
          grad[j] += 2e0 * tmp[0] * ( gradient[j] ) + lambda_g * gradient[j]; 
			// + d->me->inner_boundary_constant->at(i) * x.at(j) * tmp[2]);
      	}
      }
    }
  }

  return objective_value;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
//unsigned m, double *result, unsigned n, const double* x, double* grad, void* data)
template<class TSurrogateModel, template<class TSurrogateModel1> class TSubproblemOptimization>
double SubproblemDefinitions<TSurrogateModel, TSubproblemOptimization>::constraints_for_subproblems (
  std::vector<double> const &x, std::vector<double> &grad, void* data)  
{

  SubproblemData<TSurrogateModel, TSubproblemOptimization> *d =
    reinterpret_cast<  SubproblemData<TSurrogateModel, TSubproblemOptimization>*>(data);

  double result;
  result = (*(d->me->surrogate_models))[d->constraint_number+1].evaluate( x );
  if (!grad.empty()) {
    d->vo->mat_vec_product( (*(d->me->surrogate_models))[d->constraint_number+1].hessian( ), x, grad );
    d->vo->add ( (*(d->me->surrogate_models))[d->constraint_number+1].gradient( ), grad );
//    grad = (*(d->me->surrogate_models))[d->constraint_number+1].gradient( x );
  }

  double tmpdbl;
  //d->vo->minus ( x, d->me->best_point, d->vector );
  tmpdbl = d->vo->dot_product( x, x );

  double tmpdbl1 = pow(*(d->me->delta), 2e0);
//  if ( d->me->inner_boundary_constant->at( d->constraint_number ) < 9.9e0 ) tmpdbl1 = 1e0;

  result += d->me->inner_boundary_constant->at( d->constraint_number ) * tmpdbl *
            tmpdbl1 -
            d->me->feasibility_thresholds.at( d->constraint_number );
  if (!grad.empty()) {
    tmpdbl = d->me->inner_boundary_constant->at( d->constraint_number ) * 2e0 *
             tmpdbl1;
    for (int j = 0; j < x.size(); ++j) 
      grad[j] += tmpdbl * x.at(j);
  }

/*
  result += d->me->inner_boundary_constant->at( d->constraint_number ) * tmpdbl - 
            d->me->feasibility_thresholds.at( d->constraint_number );
  if (!grad.empty()) {
    tmpdbl = d->me->inner_boundary_constant->at( d->constraint_number ) * 2e0;
    for (int j = 0; j < x.size(); ++j ) 
      grad[j] += tmpdbl * x.at(j);
  }
*/

   return result;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
template<class TSurrogateModel, template<class TSurrogateModel1> class TSubproblemOptimization>
double SubproblemDefinitions<TSurrogateModel, TSubproblemOptimization>::trustregion_constraint (
  std::vector<double> const &x, std::vector<double> &grad, void* data)  
{

  SubproblemData<TSurrogateModel, TSubproblemOptimization> *d =
    reinterpret_cast<  SubproblemData<TSurrogateModel, TSubproblemOptimization>*>(data);

  double result;
//  d->vo->minus ( x, d->me->best_point, d->vector );
  result = 1e0*(d->vo->dot_product( x, x ) - 1e0);//pow( *(d->me->delta), 2e0 );


  if (!grad.empty()) {
    for (int i = 0; i < x.size(); ++i) 
      grad[i] = 2e0 * x[i];
  }

  return result;
}
//--------------------------------------------------------------------------------




#endif
