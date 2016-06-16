#ifndef HSubproblemDefinitions
#define HSubproblemDefinitions

#include "VectorOperations.hpp"
#include <Eigen/Core>
#include <vector>


template<class TSurrogateModel, template<class TSurrogateModel1> class TSubproblemOptimization>
struct SubproblemData {
    TSubproblemOptimization<TSurrogateModel> *me;
    VectorOperations *vo;
    std::vector<double> vector;
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

  TSubproblemOptimization<TSurrogateModel> *d = 
    reinterpret_cast<TSubproblemOptimization<TSurrogateModel>*>(data);

  if (!grad.empty( ))
    grad = (*(d->surrogate_models))[0].gradient( x );

  return (*(d->surrogate_models))[0].evaluate( x );
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
template<class TSurrogateModel, template<class TSurrogateModel1> class TSubproblemOptimization>
double SubproblemDefinitions<TSurrogateModel, TSubproblemOptimization>::opt_criticality_measure_obj (
  std::vector<double> const &x, std::vector<double> &grad, void* data) 
{

  SubproblemData<TSurrogateModel, TSubproblemOptimization> *d =
    reinterpret_cast<  SubproblemData<TSurrogateModel, TSubproblemOptimization>*>(data);

  d->vo->minus( x, d->me->best_point, d->vector );
 
 if (!grad.empty( )) {
    for (unsigned int i = 0; i < x.size( ); i++) 
      grad[i] = d->me->criticality_gradient.at ( i );
  }  

  return d->vo->dot_product( d->vector, d->me->criticality_gradient );
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
template<class TSurrogateModel, template<class TSurrogateModel1> class TSubproblemOptimization>
double SubproblemDefinitions<TSurrogateModel, TSubproblemOptimization>::opt_restore_feasibility_obj (
  std::vector<double> const &x, std::vector<double> &grad, void* data) 
{

  SubproblemData<TSurrogateModel, TSubproblemOptimization> *d =
    reinterpret_cast<  SubproblemData<TSurrogateModel, TSubproblemOptimization>*>(data);

  d->vo->minus( x, d->me->best_point, d->vector );
  std::vector<double> gradient (x.size());
  double objective_value = 0e0;
  double tmp[3];
  if (!grad.empty( )) {
    for (unsigned int j = 0; j < x.size( ); j++)
      grad[ j ] = 0e0;
  }
  tmp[1] = d->vo->dot_product( d->vector, d->vector );
  tmp[2] = pow( *(d->me->delta), 2e0 );

  for ( int i = 0; i < (d->me->surrogate_models)->size()-1; i++ ) {
    tmp[0] = (*(d->me->surrogate_models))[i+1].evaluate( x );
    if ( tmp[0] > 0e0 ) {
      tmp[0] += d->me->inner_boundary_constant->at(i) * tmp[1] / tmp[2];
      gradient = (*(d->me->surrogate_models))[i+1].gradient( x );
      objective_value += pow( tmp[0] , 2e0);
      if (!grad.empty( )) {
        for (unsigned int j = 0; j < x.size( ); j++) 
          grad[j] += 2e0 * tmp[0] * ( gradient.at( j ) + 
                     d->me->inner_boundary_constant->at(i) * d->vector.at(j) / tmp[2]);
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
  if (!grad.empty())
    grad = (*(d->me->surrogate_models))[d->constraint_number+1].gradient( x );


  double tmpdbl;
  d->vo->minus ( x, d->me->best_point, d->vector );
  tmpdbl = d->vo->dot_product( d->vector, d->vector );

/*
  result += d->me->inner_boundary_constant->at( d->constraint_number ) * tmpdbl / 
            pow(*(d->me->delta), 2e0) -
            d->me->feasibility_thresholds( d->constraint_number );
  if (!grad.empty()) {
    tmpdbl = d->me->inner_boundary_constant->at( d->constraint_number ) * 2e0 / 
             pow(*(d->me->delta), 2e0) ;
    for ( int j = 0; j < x.size(); j++) 
      grad[j] += tmpdbl * d->vector.at(j);
  }
*/
  result += d->me->inner_boundary_constant->at( d->constraint_number ) * tmpdbl - 
            d->me->feasibility_thresholds( d->constraint_number );
  if (!grad.empty()) {
    tmpdbl = d->me->inner_boundary_constant->at( d->constraint_number ) * 2e0;
    for ( int j = 0; j < x.size(); j++) 
      grad[j] += tmpdbl * d->vector.at(j);
  }


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
  d->vo->minus ( x, d->me->best_point, grad );
  result = d->vo->dot_product( grad, grad ) - pow( *(d->me->delta), 2e0 );

  if (!grad.empty()) {
    for (int i = 0; i < x.size(); i++) 
      grad[i] = 2e0 * grad[i];
  }

  return result;
}
//--------------------------------------------------------------------------------




#endif
