#ifndef HSubproblemOptimization
#define HSubproblemOptimization

#include "SubproblemDefinitions.hpp"
#include "VectorOperations.hpp"
#include <Eigen/Core>
#include <vector>
#include <nlopt.hpp>
#include <cassert>
#include <iomanip>

#define NLOPT_ALG LN_COBYLA
//#define NLOPT_ALG LD_CCSAQ
//#define NLOPT_ALG LD_SLSQP


template <class TSurrogateModel>
class SubproblemOptimization : public VectorOperations {
  private:
    int dim;
    int number_constraints;
    nlopt::opt opt_criticality_measure;
    nlopt::opt opt_trial_point;
    nlopt::opt opt_restore_feasibility;
    std::vector<double> lb, ub;
    std::vector<double> lower_bounds, upper_bounds;
    double constraint_tolerance;
    std::vector<double> abs_tol;
    double rel_tol;
    std::vector<double> x;
    double optimization_result;
    SubproblemOptimization *me;
    std::vector< SubproblemData<TSurrogateModel, SubproblemOptimization> > SpD;
    SubproblemDefinitions<TSurrogateModel, SubproblemOptimization> subproblems;
    void set_feasibility_thresholds ( std::vector<double> const& );
    bool point_is_feasible;
    void set_local_lower_bounds (std::vector<double> const& );
    void set_local_upper_bounds (std::vector<double> const& );
  public:
    VectorOperations vo;
    std::vector<double> best_point;
    std::vector<double> *inner_boundary_constant;
    double *delta;
    std::vector<double> feasibility_thresholds;
    std::vector<TSurrogateModel> *surrogate_models;
    std::vector<double> criticality_gradient;
    SubproblemOptimization ( std::vector<TSurrogateModel>&, double&, std::vector<double>&); 
    double compute_criticality_measure ( std::vector<double>& );
    double compute_trial_point ( std::vector<double>& );
    double restore_feasibility ( std::vector<double>& );
    void set_lower_bounds ( std::vector<double> const&); 
    void set_upper_bounds ( std::vector<double> const&); 
};



//--------------------------------------------------------------------------------
template<class TSurrogateModel>
SubproblemOptimization<TSurrogateModel>::SubproblemOptimization ( 
                             std::vector<TSurrogateModel> &surrogate_models_input,
                             double &delta_input, std::vector<double> &inner_boundary_path_input) :
                             delta ( &delta_input ),
                             inner_boundary_constant ( &inner_boundary_path_input ),
                             surrogate_models ( &surrogate_models_input ),
                             opt_criticality_measure (nlopt::NLOPT_ALG, 
                               surrogate_models_input[0].dimension( ) ),
                             opt_trial_point (nlopt::NLOPT_ALG, 
                               surrogate_models_input[0].dimension( ) ),
                             opt_restore_feasibility (nlopt::NLOPT_ALG, 
                               surrogate_models_input[0].dimension( ) )
{
  me = this;
  dim = (*surrogate_models)[0].dimension( );
  number_constraints = (*surrogate_models).size( )-1;

  feasibility_thresholds.resize( number_constraints ); 
  criticality_gradient.resize( dim );
  best_point.resize( dim );

  constraint_tolerance = 1e-12;

  lb.resize( dim );
  ub.resize( dim );

  double opt_time = 1e-1;
  int opt_eval = 1000;
  rel_tol = 1e-11; //1e-11
  double abs_ftol = 1e-5;
  for (int i = 0; i < dim; ++i) {
    abs_tol.push_back( 1e-5 ); //1e-5
    x.push_back ( 0e0 );    
    lb[i] = -1e0;
    ub[i] = 1e0;
  }

  SubproblemData<TSurrogateModel, SubproblemOptimization> SpD_prototype;
  SpD_prototype.me = this;
  SpD_prototype.vo = &vo;

//  SpD_prototype.vector.resize( dim );
//  SpD_prototype.constraint_number = 0;
//  SpD.push_back( SpD_prototype );

  int SpDlen = (number_constraints==0)?1:number_constraints;
  for (int i = 0; i < SpDlen; ++i ) {
//    SpD_prototype.constraint_number = i;
    SpD.push_back( SpD_prototype );
    SpD[ i ].constraint_number = i;
  }

  //opt_criticality_measure.set_ftol_abs( abs_ftol );
  //opt_criticality_measure.set_ftol_rel( rel_tol );
  //opt_criticality_measure.set_xtol_abs( abs_tol );
  opt_criticality_measure.set_xtol_rel( rel_tol );
  //opt_criticality_measure.set_maxtime( opt_time );
  opt_criticality_measure.set_maxeval( opt_eval );
  opt_criticality_measure.set_min_objective ( subproblems.opt_criticality_measure_obj, &SpD[0] );
  opt_criticality_measure.add_inequality_constraint( 
    subproblems.trustregion_constraint, &SpD[0], constraint_tolerance );
  for (int i = 0; i < number_constraints; ++i)
    opt_criticality_measure.add_inequality_constraint( 
      subproblems.constraints_for_subproblems, &SpD[i], constraint_tolerance );

  //opt_trial_point.set_ftol_abs( abs_ftol );
  //opt_trial_point.set_ftol_rel( rel_tol );
  opt_trial_point.set_xtol_abs( abs_tol );
  opt_trial_point.set_xtol_rel( rel_tol );
  //opt_trial_point.set_maxtime( opt_time );
  opt_trial_point.set_maxeval( opt_eval );
  opt_trial_point.set_min_objective ( subproblems.opt_trial_point_obj, &SpD[0] );
  opt_trial_point.add_inequality_constraint( 
    subproblems.trustregion_constraint, &SpD[0], constraint_tolerance );
  for (int i = 0; i < number_constraints; ++i)
    opt_trial_point.add_inequality_constraint( 
      subproblems.constraints_for_subproblems, &SpD[i], constraint_tolerance );

  //opt_restore_feasibility.set_ftol_abs( abs_ftol );
  //opt_restore_feasibility.set_ftol_rel( rel_tol );
  opt_restore_feasibility.set_xtol_abs( abs_tol );
  opt_restore_feasibility.set_xtol_rel( rel_tol );
  //opt_restore_feasibility.set_maxtime( opt_time );  
  opt_restore_feasibility.set_maxeval( opt_eval );
  opt_restore_feasibility.set_min_objective ( subproblems.opt_restore_feasibility_obj, &SpD[0] );
  opt_restore_feasibility.add_inequality_constraint( 
    subproblems.trustregion_constraint, &SpD[0], constraint_tolerance );
  for (int i = 0; i < number_constraints; ++i)
    opt_restore_feasibility.add_inequality_constraint( 
      subproblems.constraints_for_subproblems, &SpD[i], constraint_tolerance );
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel>
void SubproblemOptimization<TSurrogateModel>::set_lower_bounds ( 
  std::vector<double> const &lower_bounds_input )
{
  lower_bounds = lower_bounds_input;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel>
void SubproblemOptimization<TSurrogateModel>::set_upper_bounds ( 
  std::vector<double> const &upper_bounds_input )
{
  upper_bounds = upper_bounds_input;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel>
void SubproblemOptimization<TSurrogateModel>::set_feasibility_thresholds ( 
  std::vector<double> const &x )
{ 
  point_is_feasible = true;
  for ( int i = 0; i < number_constraints; i++ ) {
    feasibility_thresholds.at( i ) = (*surrogate_models)[i+1].evaluate ( x );
    if ( feasibility_thresholds.at( i ) < 0e0 ) feasibility_thresholds.at( i ) = 0e0; 
    else point_is_feasible = false;
  }
  /*
  std::cout << "#M2# Subproblem Testing feasiblity: " << std::endl;
  std::cout << "#M2#      at point: [";
  for(int i = 0; i < x.size(); ++i)
    std::cout << x[i] << ', ';
  std::cout << std::endl;
  std::cout << "#M2#      Feasiblity thresholds:";
  for ( int i = 0; i < number_constraints; i++ ) {
    std::cout << feasibility_thresholds.at( i ) << ", ";
  }
  std::cout << std::endl;
  std::cout << "#M2#      feasible: " << point_is_feasible << std::endl;
  */
  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
template<class TSurrogateModel>
void SubproblemOptimization<TSurrogateModel>::set_local_lower_bounds ( 
  std::vector<double> const &x )
{
  if ( !lower_bounds.empty() ) {
    for ( int i = 0; i < dim; ++i ) {
      lb[i] = -(x[i] - lower_bounds[i])/(*delta);
      if ( lb[i] > 0e0 ) lb[i] = 0e0;
      if ( lb[i] < -1e0 ) lb[i] = -1e0;
    }
  }
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel>
void SubproblemOptimization<TSurrogateModel>::set_local_upper_bounds ( 
  std::vector<double> const &x )
{
  if ( !upper_bounds.empty() ) {
    for ( int i = 0; i < dim; ++i ) {
      ub[i] = (upper_bounds[i] - x[i])/(*delta);
      if ( ub[i] < 0e0 ) ub[i] = 0e0;
      if ( ub[i] > 1e0 ) ub[i] = 1e0;
    }
  }
  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
template<class TSurrogateModel>
double SubproblemOptimization<TSurrogateModel>::compute_criticality_measure ( 
                                                std::vector<double> &x )
{
  double lambda_g = *delta;
  best_point = x;

  for ( int i = 0; i < dim; ++i )
    x[i] = 0e0;

  set_local_lower_bounds( best_point );
  opt_criticality_measure.set_lower_bounds ( lb );

  set_local_upper_bounds( best_point );
  opt_criticality_measure.set_upper_bounds ( ub );

  set_feasibility_thresholds ( x );

  if ( point_is_feasible ) {
    criticality_gradient = (*surrogate_models)[0].gradient ( );
  } else {
    set_zero( criticality_gradient );
    for ( int i = 1; i < number_constraints+1; ++i )
      add( 2e0*feasibility_thresholds.at(i-1) + lambda_g,   
           (*surrogate_models)[i].gradient ( ), 
           criticality_gradient );
  }
  int errmess = -2541981;
  try {
   errmess =  opt_criticality_measure.optimize ( x, optimization_result );
  } catch ( ... ) {};


  add( *delta, x, best_point );
  x = best_point;
 
  return fabs( optimization_result ) / *delta ;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
template<class TSurrogateModel>
double SubproblemOptimization<TSurrogateModel>::compute_trial_point ( 
                                                std::vector<double> &x )
{
  int errmess = -2541981;

  best_point = x;
  //std::cout << "-----------------------------1-----------------------" << std::endl;
  for ( int i = 0; i < dim; ++i ) 
    x[i] = 0e0;

  set_local_lower_bounds( best_point );
  opt_trial_point.set_lower_bounds ( lb );
  set_local_upper_bounds( best_point );
  opt_trial_point.set_upper_bounds ( ub );

  /*std::cout << "TrialPointOpt: " << std::endl;
  std::cout << "LowerBound: " << std::endl;
  for (int i = 0; i < dim; ++i)
    std::cout << lb[i] << " ";
  std::cout << "\nUpperBound: " << std::endl;
  for (int i = 0; i < dim; ++i)
    std::cout << ub[i] << " ";
  std::cout << "\nbest_point: " << std::endl;
  for (int i = 0; i < dim; ++i)
    std::cout << best_point[i] << " ";
  std::cout << "\n x: " << std::endl;
  for (int i = 0; i < dim; ++i)
    std::cout << x[i] << " ";
  std::cout << std::endl;*/

  set_feasibility_thresholds ( x );

  //std::cout << "#M3: Point is feasible: " << point_is_feasible << std::endl;
  if ( !point_is_feasible ) {
    //std::cout << "#M3: FEASIBILITY RESTORARION: " << std::endl;
    //opt_restore_feasibility.optimize ( x, optimization_result );for ( int i = 0; i < dim; ++i ) 

    opt_restore_feasibility.set_lower_bounds ( lb );
    opt_restore_feasibility.set_upper_bounds ( ub );

    try{
      errmess = opt_restore_feasibility.optimize ( x, optimization_result );
    } catch ( ... ) { };

    set_feasibility_thresholds ( x );    
    //std::cout << "#M4: Point is feasible: " << point_is_feasible << std::endl;
    if ( point_is_feasible ) {
      /*std::cout << "      LowerBound: " << std::endl;
      for (int i = 0; i < dim; ++i)
        std::cout << std::setprecision(16) << "      " << lb[i] << " ";
      std::cout << "\n      UpperBound: " << std::endl;
      for (int i = 0; i < dim; ++i)
        std::cout << std::setprecision(16) << "      " << ub[i] << " ";
      std::cout << "\n      best_point: " << std::endl;
      for (int i = 0; i < dim; ++i)
        std::cout << std::setprecision(16) << "      " << best_point[i] << " ";
      std::cout << "\n      x: " << std::endl;
      for (int i = 0; i < dim; ++i)
        std::cout << std::setprecision(16) << "      " << x[i] << " ";
      std::cout << std::endl;*/
      for (int i = 0; i < dim; ++i)
        if(x[i] > ub[i]){
          //std::cout << std::setprecision(16) << "Setting x[i] to upper bound: " << ub[i] << std::endl;
          x[i] = ub[i];
        }else if(x[i] < lb[i]){
          //std::cout << std::setprecision(16) << "Setting x[i] to lower bound: " << lb[i] << std::endl;
          x[i] = lb[i];
        }

      //std::cout << "#M3 Inner: Classic optimization!" << std::endl;
      try{
        errmess = opt_trial_point.optimize ( x, optimization_result );
      } catch ( ... ) { };
    }
   // assert( false );
  } else {
    //std::cout << "#M3: Classic optimization!" << std::endl;
    try{
      errmess = opt_trial_point.optimize ( x, optimization_result );
    } catch ( ... ) { };
  }

  add( *delta, x, best_point );
  x = best_point;
  //std::cout << "-----------------------------2-----------------------" << std::endl;

  optimization_result = (*surrogate_models)[0].evaluate(x);

  return optimization_result;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel>
double SubproblemOptimization<TSurrogateModel>::restore_feasibility ( 
                                                std::vector<double> &x )
{

  best_point = x;

  for ( int i = 0; i < dim; ++i ) 
    x[i] = 0e0;

  set_local_lower_bounds( best_point );
  opt_restore_feasibility.set_lower_bounds ( lb );

  set_local_upper_bounds( best_point );
  opt_restore_feasibility.set_upper_bounds ( ub );

  set_feasibility_thresholds ( x );

  int errmess;
  try{
    errmess = opt_restore_feasibility.optimize ( x, optimization_result );
  } catch ( ... ) { };

  add( *delta, x, best_point );
  x = best_point;

  return optimization_result;
}
//--------------------------------------------------------------------------------


#endif
