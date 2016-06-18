#ifndef HSubproblemOptimization
#define HSubproblemOptimization

#include "SubproblemDefinitions.hpp"
#include "VectorOperations.hpp"
#include <Eigen/Core>
#include <vector>
#include <nlopt.hpp>

#define NLOPT_ALG LD_CCSAQ
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
    double constraint_tolerance;
    std::vector<double> abs_tol;
    double rel_tol;
    std::vector<double> x;
    double optimization_result;
    SubproblemOptimization *me;
//    std::vector< SubproblemData<TSurrogateModel, SubproblemOptimization::template SubproblemOptimization> > SpD;
//    SubproblemDefinitions<TSurrogateModel, SubproblemOptimization::template SubproblemOptimization> subproblems;
    std::vector< SubproblemData<TSurrogateModel, SubproblemOptimization> > SpD;
    SubproblemDefinitions<TSurrogateModel, SubproblemOptimization> subproblems;
    void set_feasibility_thresholds ( std::vector<double> const& );
    bool point_is_feasible;
  public:
    VectorOperations vo;
    std::vector<double> best_point;
    std::vector<double> *inner_boundary_constant;
    double *delta;
    Eigen::VectorXd feasibility_thresholds;
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
//LD_CCSAQ 
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

  feasibility_thresholds = Eigen::VectorXd::Zero ( number_constraints ); 
  criticality_gradient.resize( dim );
  best_point.resize( dim );

  constraint_tolerance = 1e-12;

  rel_tol = 1e-11; //1e-11
  for (unsigned int i = 0; i < dim; ++i) {
    abs_tol.push_back( 1e-7 ); //1e-5
    x.push_back ( 0e0 );    
  }

  SubproblemData<TSurrogateModel, SubproblemOptimization> SpD_prototype;
//  SubproblemData<TSurrogateModel, SubproblemOptimization::template SubproblemOptimization> SpD_prototype;
  SpD_prototype.me = this;
  SpD_prototype.vo = &vo;
  SpD_prototype.vector.resize( dim );
  SpD_prototype.constraint_number = 0;
  SpD.push_back( SpD_prototype );
  for ( unsigned int i = 1; i < number_constraints; ++i ) {
    SpD_prototype.constraint_number = i;
    SpD.push_back( SpD_prototype );
  }

  opt_criticality_measure.set_xtol_abs( abs_tol );
  opt_criticality_measure.set_xtol_rel( rel_tol );
  opt_criticality_measure.set_maxtime( 1e0 );
  opt_criticality_measure.set_min_objective ( subproblems.opt_criticality_measure_obj, &SpD[0] );
  opt_criticality_measure.add_inequality_constraint( 
    subproblems.trustregion_constraint, &SpD[0], constraint_tolerance );
  for (unsigned int i = 0; i < number_constraints; ++i)
    opt_criticality_measure.add_inequality_constraint( 
      subproblems.constraints_for_subproblems, &SpD[i], constraint_tolerance );
 // opt_criticality_measure.add_inequality_mconstraint( 
 //   subproblems.constraints_for_subproblems<1>, me, constraint_tolerance );

  opt_trial_point.set_xtol_abs( abs_tol );
  opt_trial_point.set_xtol_rel( rel_tol );
  opt_trial_point.set_maxtime( 1e0 );
  opt_trial_point.set_min_objective ( subproblems.opt_trial_point_obj, me );
  opt_trial_point.add_inequality_constraint( 
    subproblems.trustregion_constraint, &SpD[0], constraint_tolerance );
  for (unsigned int i = 0; i < number_constraints; ++i)
    opt_trial_point.add_inequality_constraint( 
      subproblems.constraints_for_subproblems, &SpD[i], constraint_tolerance );
//  opt_trial_point.add_inequality_mconstraint( 
//    subproblems.constraints_for_subproblems, me, constraint_tolerance );


  opt_restore_feasibility.set_xtol_abs( abs_tol );
  opt_restore_feasibility.set_xtol_rel( rel_tol );
  opt_restore_feasibility.set_maxtime( 1e0 );  
  opt_restore_feasibility.set_min_objective ( subproblems.opt_restore_feasibility_obj, &SpD[0] );
  opt_restore_feasibility.add_inequality_constraint( 
    subproblems.trustregion_constraint, &SpD[0], constraint_tolerance );
  for (unsigned int i = 0; i < number_constraints; ++i)
    opt_restore_feasibility.add_inequality_constraint( 
      subproblems.constraints_for_subproblems, &SpD[i], constraint_tolerance );
//  opt_restore_feasibility.add_inequality_mconstraint( 
//    subproblems.constraints_for_subproblems, me, constraint_tolerance );


  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel>
void SubproblemOptimization<TSurrogateModel>::set_lower_bounds ( 
  std::vector<double> const &lower_bounds )
{
  lb = lower_bounds;
  opt_criticality_measure.set_lower_bounds ( lb );
  opt_trial_point.set_lower_bounds ( lb );
  opt_restore_feasibility.set_lower_bounds ( lb );
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel>
void SubproblemOptimization<TSurrogateModel>::set_upper_bounds ( 
  std::vector<double> const &upper_bounds )
{
  ub = upper_bounds;
  opt_criticality_measure.set_upper_bounds ( ub );
  opt_trial_point.set_upper_bounds ( ub );
  opt_restore_feasibility.set_upper_bounds ( ub );
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
    feasibility_thresholds( i ) = (*surrogate_models)[i+1].evaluate ( x );
    if ( feasibility_thresholds( i ) < 0e0 ) feasibility_thresholds( i ) = 0e0; 
    else point_is_feasible = false;
  }
  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
template<class TSurrogateModel>
double SubproblemOptimization<TSurrogateModel>::compute_criticality_measure ( 
                                                std::vector<double> &x )
{
  set_feasibility_thresholds ( x );
  if ( point_is_feasible ) 
    criticality_gradient = (*surrogate_models)[0].gradient ( x );
  else {
    set_zero( criticality_gradient );
    for ( int i = 1; i < number_constraints+1; i++ )
      add( 2e0*feasibility_thresholds(i-1),   
           (*surrogate_models)[i].gradient ( x ), 
           criticality_gradient );
  }

  best_point = x;
  opt_criticality_measure.optimize ( x, optimization_result );
  return fabs( optimization_result ) / *delta ;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel>
double SubproblemOptimization<TSurrogateModel>::compute_trial_point ( 
                                                std::vector<double> &x )
{
  set_feasibility_thresholds ( x );

  best_point = x;

  if ( !point_is_feasible ) {
    opt_restore_feasibility.optimize ( x, optimization_result );
    set_feasibility_thresholds ( x );    
    if ( point_is_feasible )
      opt_trial_point.optimize ( x, optimization_result );
  } else 
    opt_trial_point.optimize ( x, optimization_result );

  return optimization_result;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel>
double SubproblemOptimization<TSurrogateModel>::restore_feasibility ( 
                                                std::vector<double> &x )
{
  set_feasibility_thresholds ( x );

  best_point = x;

  opt_restore_feasibility.optimize ( x, optimization_result );

  return optimization_result;
}
//--------------------------------------------------------------------------------




#endif
