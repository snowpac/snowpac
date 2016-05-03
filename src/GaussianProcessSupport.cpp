#include "GaussianProcessSupport.hpp"
#include <iostream>
#include <cassert>
#include <fstream>

//--------------------------------------------------------------------------------
void GaussianProcessSupport::initialize ( const int dim, const int number_processes_input,
  double &delta_input, std::vector<double> const &update_at_evaluations_input,
  int update_interval_length_input ) 
{
  nb_values = 0;
  delta = &delta_input;
  number_processes = number_processes_input;
  update_interval_length = update_interval_length_input;
  for (size_t i = 0; i < update_at_evaluations_input.size(); i++ )
    update_at_evaluations.push_back( update_at_evaluations_input.at( i ) );
  GaussianProcess gp ( dim );
  gaussian_process_nodes.resize( 0 );
  values.resize( number_processes );
  noise.resize( number_processes );
  for ( int i = 0; i < number_processes; i++) {
    gaussian_processes.push_back ( gp );
  }
  rescaled_node.resize( dim );
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void GaussianProcessSupport::update_data ( BlackboxData &evaluations ) 
{
  for (int j = 0; j < number_processes; ++j) {
    for (size_t i = nb_values; i < evaluations.values[j].size(); ++i) {
      values[j].push_back( evaluations.values[j].at(i) );    
      noise[j].push_back( evaluations.noise[j].at(i) );    
    }
  }
  nb_values = evaluations.values[0].size( );

  //assert ( nb_values == evaluations.nodes.size() );
  //assert ( nb_values == values[0].size() );
  //assert ( nb_values == noise[0].size() );

  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void GaussianProcessSupport::update_gaussian_processes ( BlackboxData &evaluations ) 
{
  if ( nb_values >= next_update && update_interval_length > 0 ) {
    do_parameter_estimation = true;
    next_update += update_interval_length;
  } 
  if ( update_at_evaluations.size( ) > 0 ) {
    if ( nb_values >= update_at_evaluations[0] ) {
      do_parameter_estimation = true;
      update_at_evaluations.erase( update_at_evaluations.begin() );
    }
  }

  if ( do_parameter_estimation ) {
    gaussian_process_active_index.clear( );
    gaussian_process_nodes.clear( );
    delta_tmp = (*delta);// * 10e0;
//    if (delta_tmp < 0.1) delta_tmp = 0.1;
    best_index = evaluations.best_index;
    for ( int i = 0; i < nb_values; ++i ) {
      if ( diff_norm ( evaluations.nodes[ i ],
                       evaluations.nodes[ best_index ] ) <= 3e0 * (delta_tmp) ) {
        gaussian_process_active_index.push_back ( i );
        rescale ( 1e0/(delta_tmp), evaluations.nodes[i], evaluations.nodes[best_index],
                  rescaled_node);
//        gaussian_process_nodes.push_back( evaluations.nodes[ i ] );
        gaussian_process_nodes.push_back( rescaled_node );
      }
    }
/*
    std::cout << "number nodes = " << gaussian_process_nodes.size() << std::endl;
    std::cout << "number nodes = " << evaluations.nodes.size() << std::endl;
*/

    gaussian_process_values.resize( gaussian_process_active_index.size( ) );
    gaussian_process_noise.resize( gaussian_process_active_index.size( ) );

    for ( int j = 0; j < number_processes; j++ ) {
      for ( size_t i = 0; i < gaussian_process_active_index.size(); ++i ) {
        gaussian_process_values( i ) = values[ j ].at( gaussian_process_active_index[i] );
        gaussian_process_noise( i )  = noise[ j ].at( gaussian_process_active_index[i] );
      }
      gaussian_processes[j].estimate_hyper_parameters( gaussian_process_nodes,
                                                       gaussian_process_values,
                                                       gaussian_process_noise );
      gaussian_processes[j].build( gaussian_process_nodes,
                                   gaussian_process_values,
                                   gaussian_process_noise );
    }
  } else {
    for ( size_t i = last_included; i < values[0].size(); ++i ) {
      gaussian_process_active_index.push_back ( i );
      rescale ( 1e0/(delta_tmp), evaluations.nodes[i], evaluations.nodes[best_index],
                rescaled_node);
      for ( int j = 0; j < number_processes; ++j ) {
//        gaussian_processes[j].update( evaluations.nodes[ i ],
        gaussian_processes[j].update( rescaled_node,
                                      values[ j ].at( i ),
                                      noise[ j ].at( i ) );
      }
    }

  }

  last_included = evaluations.nodes.size();
  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void GaussianProcessSupport::smooth_data ( BlackboxData &evaluations ) 
{  


  update_data( evaluations );

  update_gaussian_processes( evaluations );

  for ( size_t i = 0; i < evaluations.surrogate_nodes_index.size( ); ++i ) {
    rescale ( 1e0/(delta_tmp), evaluations.nodes[evaluations.surrogate_nodes_index[i]], 
              evaluations.nodes[best_index], rescaled_node );
    for ( int j = 0; j < number_processes; ++j ) {
      gaussian_processes[j].evaluate( rescaled_node, mean, variance );
//    gaussian_processes[j].evaluate( 
//        evaluations.nodes[ evaluations.surrogate_nodes_index[ i ] ],
//        mean, variance );
//      std::cout << i << ", "<< j <<" -- variance " << variance << std::endl;
      assert ( variance >= 0e0 );

      weight = exp( - 1e0*sqrt(variance) );
/*      
      tmpdbl = 2e0*sqrt(variance);
      if (tmpdbl  >noise[j].at(evaluations.surrogate_nodes_index[i])) weight = 1e0;
      else
      weight = (noise[j].at(evaluations.surrogate_nodes_index[i]) - tmpdbl )/
               noise[j].at(evaluations.surrogate_nodes_index[i]);
      
*/
//std::cout << " weight = " << weight << std::endl;

      evaluations.values[ j ].at( evaluations.surrogate_nodes_index [ i ] ) = 
        weight * mean  + 
        (1e0-weight) * ( values[ j ].at( evaluations.surrogate_nodes_index [ i ] ) );
      evaluations.noise[ j ].at( evaluations.surrogate_nodes_index [ i ] ) = 
        weight * 2e0 * sqrt (variance)  + 
        (1e0-weight) * ( noise[ j ].at( evaluations.surrogate_nodes_index [ i ] ) );


    }
  }

  do_parameter_estimation = false;


  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
double GaussianProcessSupport::evaluate_objective ( BlackboxData const &evaluations) 
{
  rescale ( 1e0/(delta_tmp), evaluations.nodes[evaluations.best_index], 
            evaluations.nodes[best_index], rescaled_node );
  gaussian_processes[0].evaluate( rescaled_node, mean, variance );
  return mean;
}
//--------------------------------------------------------------------------------
