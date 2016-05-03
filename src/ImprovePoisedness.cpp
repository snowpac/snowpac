#include "ImprovePoisedness.hpp"
#include <fstream>

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
//    q1vec.resize ( dim );
//    q2vec.resize ( dim );
//    q.resize ( dim );
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
                                      BlackboxData const &evaluations,
                                      std::vector<double> const &new_node ) 
{
  nb_nodes = evaluations.surrogate_nodes_index.size( );
  maxvalue = -1e0;  
  basis_values = basis->evaluate ( new_node );

  for (int i = 0; i < nb_nodes; i++) {
    if ( diff_norm( evaluations.nodes[ evaluations.surrogate_nodes_index[i]], 
                    new_node) / (*delta) < 1e-6 ) {
      change_index = i;
      return change_index;
    }
    if ( evaluations.surrogate_nodes_index[ i ] != reference_node ) {
      LK = fabs( basis_values.at(i) );
      norm = diff_norm( evaluations.nodes[ evaluations.surrogate_nodes_index[i]],
               evaluations.nodes[ evaluations.best_index ] ) / (*delta);
      if ( norm <= 1e0 ) norm = 1e0;
      else { 
//        norm /= sqrt( (double) dim );
        norm = pow( norm, 3e0 );
      }
      //if (norm > 1e0) LK = LK * norm;
      LK *= norm;
      if ( LK > maxvalue ) {
        change_index = i;
        maxvalue = LK;
      }
    }
  }

  return change_index;
}
//--------------------------------------------------------------------------------

/*
//--------------------------------------------------------------------------------
void ImprovePoisedness::replace_node ( BlackboxData const &evaluations ) 
{
  nb_nodes = evaluations.surrogate_nodes_index.size( );
  maxvalue = -1e0;  


  for (int j = 0; j < nb_nodes; j++) {
    if ( evaluations.surrogate_nodes_index[ j ] == evaluations.best_index ) continue;
    basis_values = basis->evaluate ( evaluations.nodes.at( evaluations.surrogate_nodes_index[j]) );
    for (int i = 0; i < nb_nodes; i++) {
      if ( evaluations.surrogate_nodes_index[ i ] == evaluations.best_index ) continue;
      LK = fabs( basis_values.at(i) );
      norm = diff_norm( evaluations.nodes[ evaluations.surrogate_nodes_index[i]],
             evaluations.nodes[ evaluations.best_index ] ) / (*delta);
      if ( norm <= 1e0 ) norm = 1e0;
      else               norm /= sqrt( (double) dim );
      norm = pow( norm, 3e0 );
      //if (norm > 1e0) LK = LK * norm;
      LK *= norm;
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
                                             BlackboxData &evaluations ) 
{
  
  nb_nodes = evaluations.surrogate_nodes_index.size( );

//  if ( evaluations.surrogate_nodes_index.size( ) < max_nb_nodes ) return;

  //check quality of the geometrical distribution of the interpolation nodes
  std::vector<double> new_node( dim );
  
  //initialize all nodes to be not changed (changed_nodes_index = false)
  index_of_changed_nodes.clear( );

//  if ( print_output && poisedness_constant > threshold_for_poisedness_constant )
  if ( print_output )
    std::cout << " Improving model ........ ";// << std::endl;
//    std::cout << " Performing model improvement ... ";// << std::endl;

  compute_poisedness_constant ( reference_node, new_node, evaluations );
	
  int counter_max_improvement_steps = 0;
  int max_improvement_steps;
  max_improvement_steps = (int) ceil( sqrt( dim ) );
  if ( max_improvement_steps < 2 ) max_improvement_steps = 2;
 
  //XXX
  //double tmpdelete; 
  //int intdelete;

  model_has_been_improved = false;
  while (poisedness_constant > threshold_for_poisedness_constant 
         && counter_max_improvement_steps < max_improvement_steps) {

    counter_max_improvement_steps++;
        
//    if ( print_output )
//      std::cout << "   Current poisedness value: " << poisedness_constant << std::endl; 

  //  intdelete = evaluations.surrogate_nodes_index[ change_index ];
  //  tmpdelete = poisedness_constant;

    evaluations.nodes.push_back ( new_node );
    if ( evaluations.surrogate_nodes_index.size( ) >= max_nb_nodes ) {//|| model_has_been_improved ) {
      evaluations.surrogate_nodes_index.erase( 
        evaluations.surrogate_nodes_index.begin() + change_index );
 //    std::cout << "------------" << std::endl;
   }
    evaluations.surrogate_nodes_index.push_back( evaluations.nodes.size()-1 );
    model_has_been_improved = true;
		
    //compute new basis for models
    basis->compute_basis_coefficients ( evaluations );
		

    //compute geometry value (poisedness_constant), worst point index (change_index) and 
    // new point to improve poisedness (new_node)
    compute_poisedness_constant ( reference_node,  new_node, evaluations );

/*    
    if (tmpdelete < poisedness_constant) {

  std::ofstream outputfile ( "points.dat" );
  if ( outputfile.is_open( ) ) {
    for (int i = 0; i < evaluations.surrogate_nodes_index.size(); ++i) {
        outputfile << evaluations.nodes[evaluations.surrogate_nodes_index[i]][0] << "; " << 
                      evaluations.nodes[evaluations.surrogate_nodes_index[i]][1] << std::endl;
    } 
        outputfile << evaluations.nodes[intdelete][0] << "; " << 
                      evaluations.nodes[intdelete][1] << std::endl;
        outputfile << evaluations.nodes[evaluations.best_index][0] << "; " << 
                      evaluations.nodes[evaluations.best_index][1] << std::endl;
    
    outputfile << *delta << "; " << poisedness_constant<< std::endl; 
    outputfile.close( );
    assert(false);
  } else std::cout << "Unable to open file." << std::endl;
    


    }
*/

  }

  //compute new basis for models of objective function and constraints
//  if (model_has_been_improved && print_output )
  if ( print_output )
    std::cout << "done" <<  std::endl; 
//  std::cout << "   Final poisedness value  : " <<  poisedness_constant << std::endl; 

  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void ImprovePoisedness::compute_poisedness_constant ( int reference_node,
  std::vector<double> &new_node, BlackboxData &evaluations ) 
{
  //initialize geometry value
  poisedness_constant     = -1e0;

  nb_nodes = evaluations.surrogate_nodes_index.size ( );

  for (int i = 0; i < nb_nodes; i++) {		
    if ( evaluations.surrogate_nodes_index[ i ] == reference_node ) continue; 
    //compute norm-scaling of node i
    node_norm_scaling = 
      diff_norm( evaluations.nodes[ evaluations.surrogate_nodes_index[i] ],
                 evaluations.nodes[ evaluations.best_index ] ) / (*delta);
    if (node_norm_scaling <= 1e0) { node_norm_scaling = 1e0; 
    } else {
 //    node_norm_scaling /= sqrt((double) dim);
      node_norm_scaling = pow(node_norm_scaling, 3e0);
    }

    if ( node_norm_scaling > 1e2 && nb_nodes < max_nb_nodes ) continue;

    //compute gradient and hessian of the i-th basis function
    basis->get_mat_vec_representation( i, basis_gradient, basis_hessian );

    assert( VectorOperations::norm( basis_gradient ) > 1e-12 );

    //compute candidate for argmax -l_i(x)
    QuadraticMinimization::minimize( q1, basis_gradient, basis_hessian );
    for ( int k = 0; k < dim; ++k )
      q1.at(k) = q1.at(k)*(*delta) + evaluations.nodes[ evaluations.best_index ].at(k);
    poisedness_constant_tmp1 = fabs ( basis->evaluate ( q1, i ) );
    poisedness_constant_tmp1 += 1e0/poisedness_constant_tmp1;
    poisedness_constant_tmp1 *= node_norm_scaling;
			
    //compute candidate for argmax l_i(x)
    VectorOperations::scale ( -1e0, basis_gradient, basis_gradient );
    for ( int k = 0; k < dim; ++k )
      VectorOperations::scale ( -1e0, basis_hessian[k], basis_hessian[k] );
//    basis_gradient = -basis_gradient;
//    basis_hessian  = -basis_hessian;
    QuadraticMinimization::minimize ( q2, basis_gradient, basis_hessian ); 

    for ( int k = 0; k < dim; ++k)
      q2.at(k) = q2.at(k)*(*delta) + evaluations.nodes[ evaluations.best_index ].at(k);
    poisedness_constant_tmp2 = fabs ( basis->evaluate ( q2, i ) );
    poisedness_constant_tmp2 += 1e0/poisedness_constant_tmp2 ;
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
      new_node = q1;
      poisedness_constant = poisedness_constant_tmp1;
      change_index = i;
    } else if (poisedness_constant_tmp2 > poisedness_constant_tmp1 && 
               poisedness_constant_tmp2 > poisedness_constant) {
      new_node = q2;
      poisedness_constant = poisedness_constant_tmp2;
      change_index = i;
    }

  }

  assert ( evaluations.best_index != evaluations.surrogate_nodes_index[change_index] );

    	
  return;
}
//--------------------------------------------------------------------------------
