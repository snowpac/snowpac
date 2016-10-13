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
  std::sort(update_at_evaluations.begin(), update_at_evaluations.end());
  GaussianProcess gp ( dim, *delta );
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
void GaussianProcessSupport::update_data ( BlackBoxData &evaluations )
{
  for (int j = 0; j < number_processes; ++j) {
    for ( unsigned int i = nb_values; i < evaluations.values[j].size(); ++i) {
      values[j].push_back( evaluations.values[j].at(i) );    
      noise[j].push_back( evaluations.noise[j].at(i) );    
    }
  }
  nb_values = evaluations.values[0].size( );

  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void GaussianProcessSupport::update_gaussian_processes ( BlackBoxData &evaluations )
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
        gaussian_process_nodes.push_back( evaluations.nodes[ i ] );
      }
    }

    gaussian_process_values.resize(gaussian_process_active_index.size());
    gaussian_process_noise.resize(gaussian_process_active_index.size());
      
      
    for ( int j = 0; j < number_processes; ++j ) {
      for ( unsigned int i = 0; i < gaussian_process_active_index.size(); ++i ) {
        gaussian_process_values.at(i) = values[ j ].at( gaussian_process_active_index[i] );
        gaussian_process_noise.at(i) = noise[ j ].at( gaussian_process_active_index[i] );
      }
      gaussian_processes[j].estimate_hyper_parameters( gaussian_process_nodes,
                                                       gaussian_process_values,
                                                       gaussian_process_noise );
      gaussian_processes[j].build( gaussian_process_nodes,
                                   gaussian_process_values,
                                   gaussian_process_noise );
    }

      
  } else {
    for ( unsigned int i = last_included; i < values[0].size(); ++i ) {
      gaussian_process_active_index.push_back ( i );
      for ( int j = 0; j < number_processes; ++j ) {
        gaussian_processes[j].update( evaluations.nodes[ i ],
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
void GaussianProcessSupport::smooth_data ( BlackBoxData &evaluations )
{  


  update_data( evaluations );

  update_gaussian_processes( evaluations );


  evaluations.active_index.push_back( evaluations.nodes.size()-1 );
  for ( unsigned int i = 0; i < evaluations.active_index.size( ); ++i ) {
    for ( int j = 0; j < number_processes; ++j ) {
      gaussian_processes[j].evaluate( evaluations.nodes[evaluations.active_index[i]], mean, variance );

      assert ( variance >= 0e0 );

      weight = exp( - 2e0*sqrt(variance) );

      evaluations.values[ j ].at( evaluations.active_index [ i ] ) = 
        weight * mean  + 
        (1e0-weight) * ( values[ j ].at( evaluations.active_index [ i ] ) );
      evaluations.noise[ j ].at( evaluations.active_index [ i ] ) = 
        weight * 2e0 * sqrt (variance)  + 
        (1e0-weight) * ( noise[ j ].at( evaluations.active_index [ i ] ) );

    }
  }
  evaluations.active_index.erase( evaluations.active_index.end()-1 );


  do_parameter_estimation = false;


  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
double GaussianProcessSupport::evaluate_objective ( BlackBoxData const &evaluations)
{
  gaussian_processes[0].evaluate( evaluations.nodes[evaluations.best_index], mean, variance );
  return mean;
}
//--------------------------------------------------------------------------------
