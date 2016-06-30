#include "ImprovePoisedness.hpp"
#include <fstream>
#include <iostream>

//--------------------------------------------------------------------------------
ImprovePoisedness::ImprovePoisedness ( BasisForSurrogateModelBaseClass &B,
                                       double poisedness_threshold,
                                       int m, double &rad, int verbose ) :
                                       ImprovePoisednessBaseClass (
                                       poisedness_threshold, B ),
                                       QuadraticMinimization ( B.dimension() ),
                                       max_nb_nodes ( m )
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
  nb_nodes = evaluations.active_index.size( );
  maxvalue = -1e0;  

  // test for collinearity
  std::vector<double> v1 ( dim );
  std::vector<double> v2 ( dim );
  v1 = evaluations.nodes[ reference_node ];
  add ( -1e0, new_node, v1 );
  scale( 1e0/norm(v1), v1, v1);
  for ( int i = nb_nodes-1; i >= 0; --i ) {
    if ( evaluations.active_index[ i ] != reference_node ) {
      v2 = evaluations.nodes[ reference_node ];
      add ( -1e0, evaluations.nodes[ evaluations.active_index[ i ] ], v2 );    
      scale( 1e0/norm(v2), v2, v2);
      if ( diff_norm( v1, v2) < 1e-3 ) {
        //change_index = i;
        //return change_index;
        evaluations.active_index.erase( evaluations.active_index.begin() + i ); 
//        break;
      } else {
        scale( -1e0, v2, v2);
        if ( diff_norm( v1, v2) < 1e-3 ) {
          //change_index = i;
          //return change_index;
          evaluations.active_index.erase( evaluations.active_index.begin() + i ); 
  //        break;
        }
      }
    }
  }


  basis_values = basis->evaluate ( evaluations.transform( new_node ) );

  for (int i = 0; i < nb_nodes; ++i) {
    if ( evaluations.active_index[ i ] != reference_node ) {
      //if ( diff_norm( evaluations.nodes[ evaluations.active_index[i]], 
      //                new_node) / (*delta) < 1e-4 ) {
      //  change_index = i;
      //  return change_index;
      //}
      LK = basis_values.at(i);
      if ( LK < 0e0 ) LK = -LK;
      norm_dbl = diff_norm( evaluations.nodes[ evaluations.active_index[i]],
               evaluations.nodes[ evaluations.best_index ] ) / (*delta);
      norm_dbl /= sqrt( (double) dim );
      if ( norm_dbl <= 1e0 ) norm_dbl = 1e0;
      else norm_dbl = pow( norm_dbl, 6e0 );
   
      //if (norm_dbl > 1e0) LK = LK * norm;
      LK *= norm_dbl;
      if ( LK > maxvalue ) {
        change_index = i;
        maxvalue = LK;
      }
    }
  }

//  assert ( change_index != evaluations.best_index);

  return change_index;
}
//--------------------------------------------------------------------------------

/*
//--------------------------------------------------------------------------------
void ImprovePoisedness::replace_node ( BlackBoxData const &evaluations )
{
  nb_nodes = evaluations.surrogate_nodes_index.size( );
  maxvalue = -1e0;  


  for (int j = 0; j < nb_nodes; j++) {
    if ( evaluations.surrogate_nodes_index[ j ] == evaluations.best_index ) continue;
    basis_values = basis->evaluate ( evaluations.nodes.at( evaluations.surrogate_nodes_index[j]) );
    for (int i = 0; i < nb_nodes; i++) {
      if ( evaluations.surrogate_nodes_index[ i ] == evaluations.best_index ) continue;
      LK = fabs( basis_values.at(i) );
      norm_dbl = diff_norm( evaluations.nodes[ evaluations.surrogate_nodes_index[i]],
             evaluations.nodes[ evaluations.best_index ] ) / (*delta);
      if ( norm_dbl <= 1e0 ) norm_dbl = 1e0;
      else               norm_dbl /= sqrt( (double) dim );
      norm_dbl = pow( norm_dbl, 3e0 );
      //if (norm_dbl > 1e0) LK = LK * norm_dbl;
      LK *= norm_dbl;
      if ( LK > maxvalue ) {
        change_index = i;
        maxvalue = LK;
      }      
    }
  }

  return;
}
//--------------------------------------------------------------------------------
*/

//--------------------------------------------------------------------------------
void ImprovePoisedness::improve_poisedness ( int reference_node, 
                                             BlackBoxData &evaluations )
{
  
  nb_nodes = evaluations.active_index.size( );

//  if ( evaluations.active_index.size( ) < max_nb_nodes ) return;

  //check quality of the geometrical distribution of the interpolation nodes
  std::vector<double> new_node( dim );
  
  //initialize all nodes to be not changed (changed_nodes_index = false)
  index_of_changed_nodes.clear( );

//  if ( print_output && poisedness_constant > threshold_for_poisedness_constant )
  if ( print_output ) {
    std::cout << " Improving model ........ ";// << std::endl;
    fflush( stdout );
  }
//    std::cout << " Performing model improvement ... " << std::endl;

  compute_poisedness_constant ( reference_node, new_node, evaluations );
  int counter_max_improvement_steps = 0;
  int max_improvement_steps;
  max_improvement_steps = (int) ceil( sqrt( (double)dim ) );
  if ( max_improvement_steps < 2 ) max_improvement_steps = 2;
 
  model_has_been_improved = false;
  while (poisedness_constant > threshold_for_poisedness_constant 
         && counter_max_improvement_steps < max_improvement_steps) {

    counter_max_improvement_steps++;
        
    if ( print_output ) {
      if ( counter_max_improvement_steps == 1 ) std::cout << std::endl;
      std::cout << "   Current poisedness value: " << poisedness_constant << std::endl; 
    }

    evaluations.nodes.push_back ( new_node );

 //   for ( int i = 0; i < evaluations.active_index.size(); ++i ) 
 //     std::cout <<  i+1 << " : " << evaluations.active_index[i] << std::endl;

    if ( evaluations.active_index.size( ) >= max_nb_nodes || 
         poisedness_constant > 1e2 * threshold_for_poisedness_constant || true) {
      assert ( evaluations.active_index.at(change_index) != evaluations.best_index );
      evaluations.active_index.erase( 
        evaluations.active_index.begin() + change_index );
  //    std::cout << "------------" << std::endl;
    }
    evaluations.active_index.push_back( evaluations.nodes.size()-1 );

//   for ( int i = 0; i < evaluations.active_index.size(); ++i ) 
//     std::cout << i+1 << " : " << evaluations.active_index[i] << std::endl;
//    std::cout << "------------" << std::endl;
//    std::cout << evaluations.best_index << std::endl;

   // system("read");

    model_has_been_improved = true;
		

    //compute new basis for models
    basis->compute_basis_coefficients ( evaluations.get_scaled_active_nodes( 
                                        evaluations.nodes[ evaluations.best_index], *delta ) );
		


    //compute geometry value (poisedness_constant), worst point index (change_index) and 
    // new point to improve poisedness (new_node)
    compute_poisedness_constant ( reference_node,  new_node, evaluations );

  }

  //compute new basis for models of objective function and constraints
  if (model_has_been_improved && print_output ) 
//  if ( print_output )
  //  std::cout << "done" <<  std::endl; 
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
    node_norm_scaling /= sqrt((double) dim);    
    if (node_norm_scaling <= 1e0) { node_norm_scaling = 1e0; 
    } else {
      node_norm_scaling = pow(node_norm_scaling, 6e0);
    }

//    if ( node_norm_scaling > 1e2 && nb_nodes < max_nb_nodes ) continue;

  
    //compute gradient and hessian of the i-th basis function
    basis->get_mat_vec_representation( i, basis_gradient, basis_hessian );

    if ( VectorOperations::norm( basis_gradient ) <= 1e-12 ) {
/*
      std::cout << " STOP " << std::endl << std::endl;;
      for ( int kk = 0; kk < dim; ++kk) 
        std::cout << basis_gradient[kk] << std::endl;
      std::cout << "======" << std::endl;
      std::cout << " number of nodes = " << evaluations.nodes.size() << std::endl; 
      std::cout << "======" << std::endl;
      for (int k = 0; k < nb_nodes; ++k)  {
        for ( int kk = 0; kk < dim; ++kk) {
          std::cout << evaluations.active_index[k] << std::endl;
          std::cout << evaluations.nodes [ evaluations.active_index[k] ][ kk] << std::endl;
        }
        std::cout << "------" << std::endl;
      }
*/
      continue;
    }

    assert ( VectorOperations::norm( basis_gradient ) > 1e-12 ); 

    //compute candidate for argmax -l_i(x)
    QuadraticMinimization::minimize( q1, basis_gradient, basis_hessian );
  //  for ( int k = 0; k < dim; ++k )
  //    q1.at(k) = q1.at(k)*(*delta) + evaluations.nodes[ evaluations.best_index ].at(k);
    poisedness_constant_tmp1 = basis->evaluate ( q1, i );
    if ( poisedness_constant_tmp1 < 0e0 ) poisedness_constant_tmp1 = -poisedness_constant_tmp1;
    poisedness_constant_tmp1 *= node_norm_scaling;
			
    //compute candidate for argmax l_i(x)
    VectorOperations::scale ( -1e0, basis_gradient, basis_gradient );
    for ( int k = 0; k < dim; ++k )
      VectorOperations::scale ( -1e0, basis_hessian[k], basis_hessian[k] );
    QuadraticMinimization::minimize ( q2, basis_gradient, basis_hessian ); 

//    for ( int k = 0; k < dim; ++k)
//      q2.at(k) = q2.at(k)*(*delta) + evaluations.nodes[ evaluations.best_index ].at(k);
    poisedness_constant_tmp2 = basis->evaluate ( q2, i );
    if ( poisedness_constant_tmp2 < 0e0 ) poisedness_constant_tmp2 = -poisedness_constant_tmp2;
    poisedness_constant_tmp2 *= node_norm_scaling;

/*
    if ( node_norm_scaling >= 3e0 ) {
      std::cout << std::endl;
      std::cout << "----------------" << std::endl;
      std::cout << poisedness_constant_tmp1 << ", "
                << poisedness_constant_tmp2 << std::endl;
      std::cout << "----------------" << std::endl;
    }
*/

    if (poisedness_constant_tmp1 >= poisedness_constant_tmp2 && 
        poisedness_constant_tmp1 > poisedness_constant) {
      for ( int k = 0; k < dim; ++k)
        new_node.at(k) = q1.at(k)*(*delta) + evaluations.nodes[ evaluations.best_index ].at(k);
//      new_node = q1;
      poisedness_constant = poisedness_constant_tmp1;
      change_index = i;
    } else if (poisedness_constant_tmp2 > poisedness_constant_tmp1 && 
               poisedness_constant_tmp2 > poisedness_constant) {
      for ( int k = 0; k < dim; ++k)
        new_node.at(k) = q2.at(k)*(*delta) + evaluations.nodes[ evaluations.best_index ].at(k);
//      new_node = q2;
      poisedness_constant = poisedness_constant_tmp2;
      change_index = i;
    }

  }

  assert ( evaluations.best_index != evaluations.active_index[change_index] );

    	
  return;
}
//--------------------------------------------------------------------------------
