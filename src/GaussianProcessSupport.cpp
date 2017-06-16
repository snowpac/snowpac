#include "GaussianProcessSupport.hpp"
#include <iostream>
#include <cassert>
#include <fstream>
#include <SubsetOfRegressors.hpp>
#include <DeterministicTrainingConditional.hpp>
#include <FullyIndependentTrainingConditional.hpp>
#include <AugmentedSubsetOfRegressors.hpp>
#include <algorithm>

//--------------------------------------------------------------------------------
void GaussianProcessSupport::initialize ( const int dim, const int number_processes_input,
  double &delta_input, std::vector<double> const &update_at_evaluations_input,
  int update_interval_length_input, const std::string gaussian_process_type, const int exitconst) 
{
  nb_values = 0;
  delta = &delta_input;
  number_processes = number_processes_input;
  update_interval_length = update_interval_length_input;
  for (size_t i = 0; i < update_at_evaluations_input.size(); i++ )
    update_at_evaluations.push_back( update_at_evaluations_input.at( i ) );
  std::sort(update_at_evaluations.begin(), update_at_evaluations.end());
  //GaussianProcess gp ( dim, *delta );
  gaussian_process_nodes.resize( 0 );
  values.resize( number_processes );
  noise.resize( number_processes );
  for ( int i = 0; i < number_processes; i++) {
    //std::cout << gaussian_process_type << std::endl;
    if( gaussian_process_type.compare( "GP" ) == 0){
      gaussian_processes.push_back ( std::shared_ptr<GaussianProcess> (new GaussianProcess(dim, *delta)) );
      use_approx_gaussian_process = false;
    }else if( gaussian_process_type.compare( "SOR" ) == 0){
      use_approx_gaussian_process = true;
      gaussian_processes.push_back ( std::shared_ptr<SubsetOfRegressors> (new SubsetOfRegressors(dim, *delta)) );
    }else if( gaussian_process_type.compare( "DTC" ) == 0){
      use_approx_gaussian_process = true;
      gaussian_processes.push_back ( std::shared_ptr<DeterministicTrainingConditional> (new DeterministicTrainingConditional(dim, *delta)) );
    }else if( gaussian_process_type.compare( "FITC" ) == 0){
      use_approx_gaussian_process = true;
      gaussian_processes.push_back ( std::shared_ptr<FullyIndependentTrainingConditional> (new FullyIndependentTrainingConditional(dim, *delta)) );
    }else{
      std::cout << "GPSupport: No value set for GP type. Set to default Full Gaussian Process." << std::endl;
      use_approx_gaussian_process = false;
      gaussian_processes.push_back ( std::shared_ptr<GaussianProcess> (new GaussianProcess(dim, *delta)) );
    }
  }
  rescaled_node.resize( dim );
  NOEXIT = exitconst;
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

  //assert ( nb_values == evaluations.nodes.size() );
  //assert ( nb_values == values[0].size() );
  //assert ( nb_values == noise[0].size() );

  return;
}
//--------------------------------------------------------------------------------

void GaussianProcessSupport::update_gaussian_processes_for_gp( BlackBoxData &evaluations){
  do_parameter_estimation = false;
  for(int i = 0; i < number_processes && !do_parameter_estimation; ++i){
    do_parameter_estimation = gaussian_processes[i]->test_for_parameter_estimation(nb_values, update_interval_length, next_update, update_at_evaluations);
  }

  if ( nb_values >= next_update && update_interval_length > 0 ) {
    next_update += update_interval_length;
  } 
  if ( update_at_evaluations.size( ) > 0 ) {
    if ( nb_values >= update_at_evaluations[0] ) {
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
      bool idx_is_active = false;
      for(int j = 0; j < evaluations.active_index.size(); ++j){
        if(i == evaluations.active_index[j]){
          idx_is_active = true;
          break;
        }
      }
      if ( idx_is_active || diff_norm ( evaluations.nodes[ i ],
                       evaluations.nodes[ best_index ] ) <= 3e0 * (delta_tmp) ) {
        gaussian_process_active_index.push_back ( i );
//        rescale ( 1e0/(delta_tmp), evaluations.nodes[i], evaluations.nodes[best_index],
//                  rescaled_node);
        gaussian_process_nodes.push_back( evaluations.nodes[ i ] );
//        gaussian_process_nodes.push_back( rescaled_node );
      }
    }

    gaussian_process_values.resize(gaussian_process_active_index.size());
    gaussian_process_noise.resize(gaussian_process_active_index.size());
      
      
    for ( int j = 0; j < number_processes; ++j ) {
      for ( unsigned int i = 0; i < gaussian_process_active_index.size(); ++i ) {
        gaussian_process_values.at(i) = values[ j ].at( gaussian_process_active_index[i] );
        gaussian_process_noise.at(i) = noise[ j ].at( gaussian_process_active_index[i] );
      }
      gaussian_processes[j]->estimate_hyper_parameters( gaussian_process_nodes,
                                                       gaussian_process_values,
                                                       gaussian_process_noise );
        //std::cout << "Update Gaussian Process: " << j << std::endl;
      gaussian_processes[j]->build( gaussian_process_nodes,
                                   gaussian_process_values,
                                   gaussian_process_noise );
    }

      
  } else {
      for (unsigned int i = last_included; i < values[0].size(); ++i) {
          gaussian_process_active_index.push_back(i);
//      rescale ( 1e0/(delta_tmp), evaluations.nodes[i], evaluations.nodes[best_index],
//                rescaled_node);
          for (int j = 0; j < number_processes; ++j) {
              gaussian_processes[j]->update(evaluations.nodes[i],
                                            values[j].at(i),
                                            noise[j].at(i));
//        gaussian_processes[j].update( rescaled_node,
//                                      values[ j ].at( i ),
//                                      noise[ j ].at( i ) );
          }
      }

  }
  return;
}

void GaussianProcessSupport::update_gaussian_processes_for_agp( BlackBoxData &evaluations ){

    do_parameter_estimation = false;

    gaussian_process_active_index.clear( );
    gaussian_process_nodes.clear( );
    delta_tmp = (*delta);// * 10e0;
//    if (delta_tmp < 0.1) delta_tmp = 0.1;
    best_index = evaluations.best_index;
    for ( int i = 0; i < nb_values; ++i ) {
      if ( diff_norm ( evaluations.nodes[ i ],
                       evaluations.nodes[ best_index ] ) <= 3e0 * (delta_tmp) ) {
        gaussian_process_active_index.push_back ( i );
//        rescale ( 1e0/(delta_tmp), evaluations.nodes[i], evaluations.nodes[best_index],
//                  rescaled_node);
        gaussian_process_nodes.push_back( evaluations.nodes[ i ] );
//        gaussian_process_nodes.push_back( rescaled_node );
      }
    }

    gaussian_process_values.resize(gaussian_process_active_index.size());
    gaussian_process_noise.resize(gaussian_process_active_index.size());

    int nb_u_points = (int) (gaussian_process_nodes.size()*u_ratio);
    //std::cout << "###Nb u nodes " << nb_u_points << " points###" << std::endl;

    if(nb_u_points >= min_nb_u && !approx_gaussian_process_active){
      cur_nb_u_points = nb_u_points;
      approx_gaussian_process_active = true;
      //std::cout << "###Activating Approximate Gaussian Processes with " << nb_u_points << " points###" << std::endl;
      for(int i = 0; i < number_processes; ++i){
        gaussian_processes[i]->sample_u(cur_nb_u_points);
        
        gaussian_process_values.resize(gaussian_process_active_index.size());
        gaussian_process_noise.resize(gaussian_process_active_index.size());
        for ( unsigned int j = 0; j < gaussian_process_active_index.size(); ++j ) {
          gaussian_process_values.at(j) = values[ i ].at( gaussian_process_active_index[j] );
          gaussian_process_noise.at(j) = noise[ i ].at( gaussian_process_active_index[j] );
        }
        gaussian_processes[i]->build(gaussian_process_nodes,
                                   gaussian_process_values,
                                   gaussian_process_noise);
      }
      do_parameter_estimation = true;
    }else if(nb_u_points < min_nb_u && approx_gaussian_process_active){
      cur_nb_u_points = 0;
      approx_gaussian_process_active = false;
      //std::cout << "###Deactivatin Approximate Gaussian Processes with " << nb_u_points << " points###" << std::endl;
      for(int i = 0; i < number_processes; ++i){
        gaussian_processes[i]->clear_u();

        gaussian_process_values.resize(gaussian_process_active_index.size());
        gaussian_process_noise.resize(gaussian_process_active_index.size());
        for ( unsigned int j = 0; j < gaussian_process_active_index.size(); ++j ) {
          gaussian_process_values.at(j) = values[ i ].at( gaussian_process_active_index[j] );
          gaussian_process_noise.at(j) = noise[ i ].at( gaussian_process_active_index[j] );
        }
        gaussian_processes[i]->build(gaussian_process_nodes,
                                   gaussian_process_values,
                                   gaussian_process_noise);
      }
    }else if(nb_u_points > min_nb_u && cur_nb_u_points != nb_u_points && approx_gaussian_process_active){
      //std::cout << "###Nb u points has changed from " << cur_nb_u_points << " to " << nb_u_points << " points###" << std::endl;
      cur_nb_u_points = nb_u_points;
      for(int i = 0; i < number_processes; ++i){
        gaussian_processes[i]->sample_u(cur_nb_u_points);
        
        gaussian_process_values.resize(gaussian_process_active_index.size());
        gaussian_process_noise.resize(gaussian_process_active_index.size());
        for ( unsigned int j = 0; j < gaussian_process_active_index.size(); ++j ) {
          gaussian_process_values.at(j) = values[ i ].at( gaussian_process_active_index[j] );
          gaussian_process_noise.at(j) = noise[ i ].at( gaussian_process_active_index[j] );
        }
        gaussian_processes[i]->build(gaussian_process_nodes,
                                   gaussian_process_values,
                                   gaussian_process_noise);
      }
      do_parameter_estimation = true;
    }else{
      //std::cout << "###Nothing changed last update" << std::endl;
    }

    for(int i = 0; i < number_processes && !do_parameter_estimation; ++i){
      do_parameter_estimation = gaussian_processes[i]->test_for_parameter_estimation(nb_values, update_interval_length, next_update, update_at_evaluations);
    }

    if ( nb_values >= next_update && update_interval_length > 0 ) {
      next_update += update_interval_length;
    } 
    if ( update_at_evaluations.size( ) > 0 ) {
      if ( nb_values >= update_at_evaluations[0] ) {
        update_at_evaluations.erase( update_at_evaluations.begin() );
      }
    }
    //std::cout << "###Do parameter estimation: " << do_parameter_estimation << std::endl;


    if ( do_parameter_estimation ) {


      if(approx_gaussian_process_active){  
        gaussian_process_active_index.clear( );
        gaussian_process_nodes.clear( );
        delta_tmp = (*delta);// * 10e0;
    //    if (delta_tmp < 0.1) delta_tmp = 0.1;
        best_index = evaluations.best_index;
        for ( int i = 0; i < nb_values; ++i ) {
        //if ( diff_norm ( evaluations.nodes[ i ],
        //                 evaluations.nodes[ best_index ] ) <= 3e0 * (delta_tmp) ) {
          gaussian_process_active_index.push_back ( i );
          gaussian_process_nodes.push_back( evaluations.nodes[ i ] );
        }
        
        gaussian_process_values.resize(gaussian_process_active_index.size());
        gaussian_process_noise.resize(gaussian_process_active_index.size());
          
          
        for ( int j = 0; j < number_processes; ++j ) {
          for ( unsigned int i = 0; i < gaussian_process_active_index.size(); ++i ) {
            gaussian_process_values.at(i) = values[ j ].at( gaussian_process_active_index[i] );
            gaussian_process_noise.at(i) = noise[ j ].at( gaussian_process_active_index[i] );
          }
          gaussian_processes[j]->estimate_hyper_parameters( gaussian_process_nodes,
                                                           gaussian_process_values,
                                                           gaussian_process_noise );
            //std::cout << "Update Gaussian Process: " << j << std::endl;
          gaussian_processes[j]->build( gaussian_process_nodes,
                                       gaussian_process_values,
                                       gaussian_process_noise );
        }

      }else{
        gaussian_process_active_index.clear( );
        gaussian_process_nodes.clear( );
        delta_tmp = (*delta);// * 10e0;
    //    if (delta_tmp < 0.1) delta_tmp = 0.1;
        best_index = evaluations.best_index;
        for ( int i = 0; i < nb_values; ++i ) {
          if ( diff_norm ( evaluations.nodes[ i ],
                           evaluations.nodes[ best_index ] ) <= 3e0 * (delta_tmp) ) {
            gaussian_process_active_index.push_back ( i );
    //        rescale ( 1e0/(delta_tmp), evaluations.nodes[i], evaluations.nodes[best_index],
    //                  rescaled_node);
            gaussian_process_nodes.push_back( evaluations.nodes[ i ] );
    //        gaussian_process_nodes.push_back( rescaled_node );
          }
        }

        gaussian_process_values.resize(gaussian_process_active_index.size());
        gaussian_process_noise.resize(gaussian_process_active_index.size());
          
          
        for ( int j = 0; j < number_processes; ++j ) {
          for ( unsigned int i = 0; i < gaussian_process_active_index.size(); ++i ) {
            gaussian_process_values.at(i) = values[ j ].at( gaussian_process_active_index[i] );
            gaussian_process_noise.at(i) = noise[ j ].at( gaussian_process_active_index[i] );
          }
          gaussian_processes[j]->estimate_hyper_parameters( gaussian_process_nodes,
                                                           gaussian_process_values,
                                                           gaussian_process_noise );
            //std::cout << "Update Gaussian Process: " << j << std::endl;
          gaussian_processes[j]->build( gaussian_process_nodes,
                                       gaussian_process_values,
                                       gaussian_process_noise );
        }
      }

      
  } else {
      for (unsigned int i = last_included; i < values[0].size(); ++i) {
          gaussian_process_active_index.push_back(i);
//      rescale ( 1e0/(delta_tmp), evaluations.nodes[i], evaluations.nodes[best_index],
//                rescaled_node);
          for (int j = 0; j < number_processes; ++j) {
              gaussian_processes[j]->update(evaluations.nodes[i],
                                            values[j].at(i),
                                            noise[j].at(i));
//        gaussian_processes[j].update( rescaled_node,
//                                      values[ j ].at( i ),
//                                      noise[ j ].at( i ) );
          }
      }

  }

}

//--------------------------------------------------------------------------------
void GaussianProcessSupport::update_gaussian_processes ( BlackBoxData &evaluations )
{

  if(use_approx_gaussian_process){
    update_gaussian_processes_for_agp(evaluations);
  }else{
    update_gaussian_processes_for_gp(evaluations);
  }

  last_included = evaluations.nodes.size();
  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
int GaussianProcessSupport::smooth_data ( BlackBoxData &evaluations )
{

  update_data( evaluations );

  bool negative_variance_found;
  do{
    negative_variance_found = false;
    update_gaussian_processes( evaluations );

    evaluations.active_index.push_back( evaluations.nodes.size()-1 );
    for ( unsigned int i = 0; i < evaluations.active_index.size( ); ++i ) {
  //    rescale ( 1e0/(delta_tmp), evaluations.nodes[evaluations.active_index[i]], 
  //              evaluations.nodes[best_index], rescaled_node );
      for ( int j = 0; j < number_processes; ++j ) {
  //      gaussian_processes[j].evaluate( rescaled_node, mean, variance );

         // std::cout << "Smooth Gaussian Process: " << j << std::endl;
        gaussian_processes[j]->evaluate( evaluations.nodes[evaluations.active_index[i]], mean, variance );

        if (variance < 0e0){
          negative_variance_found = true;
          break;
        }

        weight = exp( - 2e0*sqrt(variance) );

        evaluations.values[ j ].at( evaluations.active_index [ i ] ) = 
          weight * mean  + 
          (1e0-weight) * ( values[ j ].at( evaluations.active_index [ i ] ) );
        evaluations.noise[ j ].at( evaluations.active_index [ i ] ) = 
          weight * 2e0 * sqrt (variance)  + 
          (1e0-weight) * ( noise[ j ].at( evaluations.active_index [ i ] ) );
        //std::cout << "Smooth evaluate var: [" << noise[ j ].at( evaluations.active_index [ i ]) << ", " << 2e0 *sqrt(variance) << "]\n" << std::endl;
       //std::cout << "Smooth evaluate [" << evaluations.active_index[i] << ", " << j <<"]: mean,variance " << evaluations.values[ j ].at( evaluations.active_index [ i ] ) << ", " << evaluations.noise[ j ].at( evaluations.active_index [ i ] )  << "\n" << std::endl;
      }
      if (variance < 0e0){
          negative_variance_found = true;
          break;
      }
    }

    evaluations.active_index.erase( evaluations.active_index.end()-1 );
    if(!negative_variance_found){
      for ( int j = 0; j < number_processes; ++j ) {
        gaussian_processes[j]->decrease_nugget();
      }
    }
    if(negative_variance_found){
      for ( int j = 0; j < number_processes; ++j ) {
        bool max_nugget_reached = gaussian_processes[j]->increase_nugget();
        if(max_nugget_reached){
          return -7;
        }
      }
    }
  }while(negative_variance_found);
/*
  for ( unsigned int i = 0; i < evaluations.nodes.size( ); ++i ) {
//    rescale ( 1e0/(delta_tmp), evaluations.nodes[ i ], 
//              evaluations.nodes[best_index], rescaled_node );
    for ( unsigned int j = 0; j < number_processes; ++j ) {
      gaussian_processes[j].evaluate( evaluations.nodes[ i ], mean, variance );
//      gaussian_processes[j].evaluate( rescaled_node, mean, variance );
      assert ( variance >= 0e0 );

      weight = exp( - 2e0*sqrt(variance) );

      evaluations.values[ j ].at( i ) = 
        weight * mean  + 
        (1e0-weight) * ( values[ j ].at( i ) );
      evaluations.noise[ j ].at( i ) = 
        weight * 2e0 * sqrt (variance)  + 
        (1e0-weight) * ( noise[ j ].at( i ) );

    }
  }
*/

  do_parameter_estimation = false;


  return NOEXIT;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
double GaussianProcessSupport::evaluate_objective ( BlackBoxData const &evaluations)
{
 // rescale ( 1e0/(delta_tmp), evaluations.nodes[evaluations.best_index], 
 //           evaluations.nodes[best_index], rescaled_node );
//  gaussian_processes[0].evaluate( rescaled_node, mean, variance );
  gaussian_processes[0]->evaluate( evaluations.nodes[evaluations.best_index], mean, variance );
  return mean;
}

void GaussianProcessSupport::evaluate_gaussian_process_at(const int &idx, std::vector<double> const &loc, double &mean, double &var) {
  gaussian_processes.at(idx)->evaluate(loc, mean, var);
  return;
}

const std::vector<std::vector<double>> &GaussianProcessSupport::get_nodes_at(const int &idx) const {
    return gaussian_processes.at(idx)->getGp_nodes();
}

void GaussianProcessSupport::get_induced_nodes_at(const int idx, std::vector<std::vector<double>> &induced_nodes) {
    gaussian_processes.at(idx)->get_induced_nodes(induced_nodes);
    return;
}

void GaussianProcessSupport::set_constraint_ball_center(const std::vector<double>& center){
  for ( int j = 0; j < number_processes; ++j ) {
      gaussian_processes[j]->set_constraint_ball_center(center);

    }
}

void GaussianProcessSupport::set_constraint_ball_radius(const double& radius){
  for ( int j = 0; j < number_processes; ++j ) {
      gaussian_processes[j]->set_constraint_ball_radius(radius);

    }
}

/*
void GaussianProcessSupport::do_resample_u(){
  for ( int j = 0; j < number_processes; ++j ) {
      gaussian_processes[j]->do_resample_u();
    }
}
*/
//--------------------------------------------------------------------------------
