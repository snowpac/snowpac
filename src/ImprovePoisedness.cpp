#include "ImprovePoisedness.hpp"
#include <fstream>
#include <iostream>

//--------------------------------------------------------------------------------
ImprovePoisedness::ImprovePoisedness ( BasisForSurrogateModelBaseClass &B,
                                       double poisedness_threshold,
                                       int m, double &rad, int verbose, 
                                       std::vector<double> upper_bound_constraints_in, 
                                       std::vector<double> lower_bound_constraints_in, 
                                       bool use_hard_box_constraints_in) :
                                       ImprovePoisednessBaseClass (
                                       poisedness_threshold, B ),
                                       QuadraticMinimization ( B.dimension() ),
                                       max_nb_nodes ( m ), use_hard_box_constraints(use_hard_box_constraints_in)
{ 
    dim = B.dimension ( );
    delta = &rad;
   
    //allocate memory for auxiliary variables in compute_poisedness_constant
    q1.resize ( dim );
    q2.resize ( dim );
    basis_gradient.resize ( dim );
    basis_hessian.resize ( dim );
    for ( int i = 0; i < dim; ++i )
      basis_hessian[i].resize( dim );

    if(!upper_bound_constraints_in.empty()){
      upper_bound_constraints.resize(upper_bound_constraints_in.size());
      for(int i = 0; i < upper_bound_constraints_in.size(); ++i){
        upper_bound_constraints[i] = upper_bound_constraints_in[i];
      }
    }
    if(!lower_bound_constraints_in.empty()){
      lower_bound_constraints.resize(lower_bound_constraints_in.size());
      for(int i = 0; i < lower_bound_constraints_in.size(); ++i){
        lower_bound_constraints[i] = lower_bound_constraints_in[i];
      }
    }

    tmp_node.resize ( dim );  

    if ( verbose > 2 ) print_output = true;
    else print_output = false;

}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
int ImprovePoisedness::replace_node ( int reference_node, 
                                      BlackBoxData &evaluations,
                                      std::vector<double> const &new_node )
{
  change_index = -1;

  nb_nodes = evaluations.active_index.size( );
  maxvalue = -1e0;

  // test for collinearity
  std::vector<double> v1 ( dim );
  std::vector<double> v2 ( dim );

  int checked_nodes_counter = 0;
  int check_node = 0;
  int tmp_int;

  bool ref_is_best = false;
  if ( reference_node >= 0 ) ref_is_best = true;

  reference_node = evaluations.best_index;

  std::vector<int> eval_act_ind =  evaluations.active_index;
  tmp_int = nb_nodes;

  while ( nb_nodes - 1 - checked_nodes_counter >= 0 ) {
    check_node = eval_act_ind[ nb_nodes - checked_nodes_counter - 1 ];
    v1 = evaluations.nodes[ check_node ];
    add ( -1e0, new_node, v1 );
    scale( 1e0/norm(v1), v1, v1);
    for ( int i = tmp_int-1; i >= 0; --i ) {
      if ( evaluations.active_index[ i ] != reference_node &&
           evaluations.active_index[ i ] != check_node ) {
        v2 = new_node;
//        v2 = evaluations.nodes[ reference_node ];
        add ( -1e0, evaluations.nodes[ evaluations.active_index[ i ] ], v2 );
        scale( 1e0/norm(v2), v2, v2);
        if ( diff_norm( v1, v2) < 1e-4 ) {
          evaluations.active_index.erase( evaluations.active_index.begin() + i );
          tmp_int--;
        } else {
          scale( -1e0, v2, v2);
          if ( diff_norm( v1, v2) < 1e-4 ) {
            evaluations.active_index.erase( evaluations.active_index.begin() + i );
            tmp_int--;
          }
        }
      }
    }
    checked_nodes_counter++;
  }

  nb_nodes = evaluations.active_index.size( );

  if ( !ref_is_best ) reference_node = -1;
  for ( int i = nb_nodes-1; i >= 0; --i ) {
    if ( evaluations.active_index[ i ] != reference_node ) {
      if ( diff_norm( evaluations.nodes[ evaluations.active_index[i]],
                      new_node) / (*delta) < 1e-4) { //&& evaluations.active_index[i] != evaluations.best_index
        change_index = i;
        return change_index;
      }
    }
  }

  reference_node = evaluations.best_index;

  basis_values = basis->evaluate ( evaluations.transform( new_node ) );

  for (int i = 0; i < nb_nodes; ++i) {
    if ( evaluations.active_index[ i ] != reference_node ) {
//      if ( diff_norm( evaluations.nodes[ evaluations.active_index[i]], 
//                      new_node) / (*delta) < 1e-3 ) {
//        change_index = i;
//        return change_index;
//      }
      LK = basis_values.at(i);
      if ( LK < 0e0 ) LK = -LK;
      norm_dbl = diff_norm( evaluations.nodes[ evaluations.active_index[i]],
               evaluations.nodes[ evaluations.best_index ] ) / (*delta);
      if ( norm_dbl <= 1e0 ) norm_dbl = 1e0;
      else norm_dbl = pow( norm_dbl, 3e0 );
      LK *= norm_dbl;
      if ( LK > maxvalue) { //&& evaluations.active_index[i] != evaluations.best_index
        change_index = i;
        maxvalue = LK;
      }
    }
  }
  if (print_output){
  std::cout << "##Change index: " << change_index;
  if (change_index != -1) {std::cout << ", " <<   evaluations.active_index[change_index] << ", " << evaluations.best_index;}
  std::cout << std::endl;
  }
  //assert ( evaluations.active_index[change_index] != evaluations.best_index);
  return change_index;
}
//--------------------------------------------------------------------------------



//--------------------------------------------------------------------------------
void ImprovePoisedness::improve_poisedness ( int reference_node, 
                                             BlackBoxData &evaluations )
{
  
  nb_nodes = evaluations.active_index.size( );

  //check quality of the geometrical distribution of the interpolation nodes
  std::vector<double> new_node( dim );
  
  //initialize all nodes to be not changed (changed_nodes_index = false)

  if ( print_output ) {
    std::cout << " Improving model ........ ";
    fflush( stdout );
  }

  compute_poisedness_constant ( reference_node, new_node, evaluations );
  int counter_max_improvement_steps = 0;
  int max_improvement_steps;
  max_improvement_steps = ((int) ceil( sqrt( (double)dim ) ));
  if ( max_improvement_steps < 2 ) max_improvement_steps = 2;
//  max_improvement_steps = 1000;
 
  model_has_been_improved = false;
  while (poisedness_constant > threshold_for_poisedness_constant 
         && counter_max_improvement_steps < max_improvement_steps) {

    counter_max_improvement_steps++;
        
    if ( print_output ) {
      if ( counter_max_improvement_steps == 1 ) std::cout << std::endl;
      std::cout << "   Current poisedness value: " << poisedness_constant << std::endl; 
      fflush( stdout );
    }

    evaluations.nodes.push_back ( new_node );

    if ( (evaluations.active_index.size( ) >= max_nb_nodes || 
         poisedness_constant > 1e2 * threshold_for_poisedness_constant) || true) {
      assert ( evaluations.active_index.at(change_index) != evaluations.best_index );
      evaluations.active_index.erase( 
        evaluations.active_index.begin() + change_index );
    }
    evaluations.active_index.push_back( evaluations.nodes.size()-1 );

    model_has_been_improved = true;
		
    //compute new basis for models
    basis->compute_basis_coefficients ( evaluations.get_scaled_active_nodes( *delta ) );

    //compute geometry value (poisedness_constant), worst point index (change_index) and 
    // new point to improve poisedness (new_node)
    compute_poisedness_constant ( reference_node,  new_node, evaluations );

  }

  //compute new basis for models of objective function and constraints
  if (model_has_been_improved && print_output ) 
    std::cout << "   Final poisedness value  : " <<  poisedness_constant << std::endl; 
  else if ( print_output )
    std::cout << "done" << std::endl; 


  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void ImprovePoisedness::compute_poisedness_constant ( int reference_node,
  std::vector<double> &new_node, BlackBoxData &evaluations )
{
  //initialize geometry value
  poisedness_constant     = -1e0;

  nb_nodes = evaluations.active_index.size ( );

  for ( int i = 0; i < nb_nodes; ++i ) {		
    if ( evaluations.active_index[ i ] == reference_node ) continue; 
    //compute norm-scaling of node i
    node_norm_scaling = 
      diff_norm( evaluations.nodes[ evaluations.active_index[i] ],
                 evaluations.nodes[ evaluations.best_index ] ) / (*delta);
    if (node_norm_scaling <= 1e0) { node_norm_scaling = 1e0; 
    } else {
      node_norm_scaling = pow(node_norm_scaling, 3e0);
    }

//    if ( node_norm_scaling > 1e2 && nb_nodes < max_nb_nodes ) continue;
  
    //compute gradient and hessian of the i-th basis function
    basis_gradient = basis->gradient( i );
    basis_hessian  = basis->hessian( i );

    if ( VectorOperations::norm( basis_gradient ) <= 1e-12 ) {
      continue;
    }

    assert ( VectorOperations::norm( basis_gradient ) > 1e-12 ); 

    //compute candidate for argmax -l_i(x)
    QuadraticMinimization::minimize( q1, basis_gradient, basis_hessian );
    poisedness_constant_tmp1 = basis->evaluate ( q1, i );
    if ( poisedness_constant_tmp1 < 0e0 ) poisedness_constant_tmp1 = -poisedness_constant_tmp1;
    poisedness_constant_tmp1 *= node_norm_scaling;
			
    //compute candidate for argmax l_i(x)
    VectorOperations::scale ( -1e0, basis_gradient, basis_gradient );
    for ( int k = 0; k < dim; ++k )
      VectorOperations::scale ( -1e0, basis_hessian[k], basis_hessian[k] );
    QuadraticMinimization::minimize ( q2, basis_gradient, basis_hessian ); 

    poisedness_constant_tmp2 = basis->evaluate ( q2, i );
    if ( poisedness_constant_tmp2 < 0e0 ) poisedness_constant_tmp2 = -poisedness_constant_tmp2;
    poisedness_constant_tmp2 *= node_norm_scaling;


    if (poisedness_constant_tmp1 >= poisedness_constant_tmp2 && 
        poisedness_constant_tmp1 > poisedness_constant) {
      for ( int k = 0; k < dim; ++k){
        new_node.at(k) = q1.at(k)*(*delta) + evaluations.nodes[ evaluations.best_index ].at(k);

        if(use_hard_box_constraints){
          if(!upper_bound_constraints.empty() && !lower_bound_constraints.empty()){
            if(new_node[k] > upper_bound_constraints[k]){
              new_node[k] = upper_bound_constraints[k];
            }else if(new_node[k] < lower_bound_constraints[k]){
              new_node[k] = lower_bound_constraints[k];
            }
          }else if(!upper_bound_constraints.empty()){
            if (new_node.at(k) > upper_bound_constraints[k])
            {
              new_node.at(k) = upper_bound_constraints[k];
            }
          }else if(!lower_bound_constraints.empty()){
            if (new_node.at(k) < lower_bound_constraints[k])
            {
              new_node.at(k) = lower_bound_constraints[k];
            }
          }
        }
      }
      poisedness_constant = poisedness_constant_tmp1;
      change_index = i;
    } else if (poisedness_constant_tmp2 > poisedness_constant_tmp1 && 
               poisedness_constant_tmp2 > poisedness_constant) {
      for ( int k = 0; k < dim; ++k){
        new_node.at(k) = q2.at(k)*(*delta) + evaluations.nodes[ evaluations.best_index ].at(k);

        if(use_hard_box_constraints){
          if( !upper_bound_constraints.empty() && !lower_bound_constraints.empty()){
            if(new_node[k] > upper_bound_constraints[k]){
              new_node[k] = upper_bound_constraints[k];
            }else if(new_node[k] < lower_bound_constraints[k]){
              new_node[k] = lower_bound_constraints[k];
            }
          }else if(!upper_bound_constraints.empty()){
            if (new_node.at(k) > upper_bound_constraints[k])
            {
              new_node.at(k) = upper_bound_constraints[k];
            }
          }else if(!lower_bound_constraints.empty()){
            if (new_node.at(k) < lower_bound_constraints[k])
            {
              new_node.at(k) = lower_bound_constraints[k];
            }
          }
        }
      }
      poisedness_constant = poisedness_constant_tmp2;
      change_index = i;
    }


  }

  return;
}
//--------------------------------------------------------------------------------
