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
void GaussianProcessSupport::initialize ( const int dim, const int number_processes_input, double &delta_input,
        BlackBoxBaseClass *blackbox, std::vector<int> const &update_at_evaluations_input,
  int update_interval_length_input, const std::string gaussian_process_type, const int exitconst, const bool use_analytic_smoothing)
{
  nb_values = 0;
  delta = &delta_input;
  number_processes = number_processes_input;
  update_interval_length = update_interval_length_input;
  for (size_t i = 0; i < update_at_evaluations_input.size(); i++ ) {
      update_at_evaluations.push_back(update_at_evaluations_input.at(i));
  }
  std::sort(update_at_evaluations.begin(), update_at_evaluations.end());
  //GaussianProcess gp ( dim, *delta );
  gaussian_process_nodes.resize( 0 );
  for ( int i = 0; i < number_processes; i++) {
    //std::cout << gaussian_process_type << std::endl;
    if( gaussian_process_type.compare( "GP" ) == 0){
      gaussian_processes.push_back ( std::shared_ptr<GaussianProcess> (new GaussianProcess(dim, *delta, blackbox)) );
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
      gaussian_processes.push_back ( std::shared_ptr<GaussianProcess> (new GaussianProcess(dim, *delta, blackbox)) );
    }
  }
  rescaled_node.resize( dim );
  this->use_analytic_smoothing = use_analytic_smoothing;
  NOEXIT = exitconst;
  best_index_analytic_information.resize(number_processes);
  for ( int i = 0; i < number_processes; i++) {
    best_index_analytic_information[i].resize(25);
  }
  bootstrap_estimate.resize( 0 );
  smoothing_ctr = 0;
  return;
}
//--------------------------------------------------------------------------------

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
                       evaluations.nodes[ best_index ] ) <= gaussian_process_delta_factor * (delta_tmp) ) {
        gaussian_process_active_index.push_back ( i );
//        rescale ( 1e0/(delta_tmp), evaluations.nodes[i], evaluations.nodes[best_index],
//                  rescaled_node);
        gaussian_process_nodes.push_back( evaluations.nodes[ i ] );
//        gaussian_process_nodes.push_back( rescaled_node );
      }
    }

    gaussian_process_values.resize(gaussian_process_active_index.size());
    gaussian_process_noise.resize(gaussian_process_active_index.size());

    //for (int j = 0; j < number_processes; ++j) {
    //  values[j].clear();
    //  noise[j].clear();
    //  for ( unsigned int i = 0; i < evaluations.values[j].size(); ++i) {
     //   values[j].push_back( evaluations.values[j][i] );
    //    noise[j].push_back( evaluations.noise[j][i] );
    //  }
    //  nb_values = values[0].size( );
    //  assert(noise[j].size() == nb_values);
    //  assert(values[j].size() == nb_values);
    //}

    //Since we rebuild the GP we update its values
    for ( int j = 0; j < number_processes; ++j ) {
      for ( unsigned int i = 0; i < gaussian_process_active_index.size(); ++i ) {
        gaussian_process_values[i] = evaluations.values_MC[ j ][gaussian_process_active_index[i]];
        gaussian_process_noise[i] = evaluations.noise_MC[ j ][gaussian_process_active_index[i]];
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
      for (unsigned int i = last_included; i < nb_values; ++i) {
          gaussian_process_active_index.push_back(i);
//      rescale ( 1e0/(delta_tmp), evaluations.nodes[i], evaluations.nodes[best_index],
//                rescaled_node);
          for (int j = 0; j < number_processes; ++j) {
              gaussian_processes[j]->update(evaluations.nodes[i],
                                            evaluations.values_MC[j][i],
                                            evaluations.noise_MC[j][i]);
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
          gaussian_process_values.at(j) = evaluations.values_MC[ i ].at( gaussian_process_active_index[j] );
          gaussian_process_noise.at(j) = evaluations.noise_MC[ i ].at( gaussian_process_active_index[j] );
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
          gaussian_process_values.at(j) = evaluations.values_MC[ i ].at( gaussian_process_active_index[j] );
          gaussian_process_noise.at(j) = evaluations.noise_MC[ i ].at( gaussian_process_active_index[j] );
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
          gaussian_process_values.at(j) = evaluations.values_MC[ i ].at( gaussian_process_active_index[j] );
          gaussian_process_noise.at(j) = evaluations.noise_MC[ i ].at( gaussian_process_active_index[j] );
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
            gaussian_process_values.at(i) = evaluations.values_MC[ j ].at( gaussian_process_active_index[i] );
            gaussian_process_noise.at(i) = evaluations.noise_MC[ j ].at( gaussian_process_active_index[i] );
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
                           evaluations.nodes[ best_index ] ) <= gaussian_process_delta_factor * (delta_tmp) ) {
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
            gaussian_process_values.at(i) = evaluations.values_MC[ j ].at( gaussian_process_active_index[i] );
            gaussian_process_noise.at(i) = evaluations.noise_MC[ j ].at( gaussian_process_active_index[i] );
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
        for (unsigned int i = last_included; i < nb_values; ++i) {
            gaussian_process_active_index.push_back(i);
  //      rescale ( 1e0/(delta_tmp), evaluations.nodes[i], evaluations.nodes[best_index],
  //                rescaled_node);
            for (int j = 0; j < number_processes; ++j) {
                gaussian_processes[j]->update(evaluations.nodes[i],
                                              evaluations.values_MC[j].at(i),
                                              evaluations.noise_MC[j].at(i));
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

  nb_values = evaluations.values[0].size();
  assert(nb_values == evaluations.nodes.size());
  assert(nb_values == evaluations.noise[0].size());

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
  //update_data( evaluations );

  update_gaussian_processes( evaluations );

  /*
  bool active_index_already_in_set = false;
  for(int cur_active_index : evaluations.active_index ){
    if(cur_active_index == evaluations.nodes.size()-1){
      active_index_already_in_set = true;
      break;
    }
  }
  if(!active_index_already_in_set){
    evaluations.active_index.push_back( evaluations.nodes.size()-1 );
  }
  */

  bool negative_variance_found;

  if (use_analytic_smoothing){
    std::cout << "WE ARE IN PROTOTYPICAL NEW ERROR ESTIMATION MODE. HARDCODED!!!" << std::endl;
    double var_R = 0;
    double cov_RGP = 0;
    double var_GP = 0;
    double bootstrap_diffGPRf = 0;
    double bootstrap_squared = 0;
    double denominator = 0;
    double numerator = 0;
    double optimal_gamma = 0;
    double optimal_gamma_squared = 0;
    double MSE = 0;
    double RMSE = 0;
    double Rtilde = 0;
    double cur_noise_xstar = 0;
    double cur_noise_xstar_MC = 0;
    double heuristic_RMSE = 0;
    double heuristic_Rtilde = 0;
    double heuristic_gamma = 0;
    std::vector<std::vector<double>> active_index_samples;
    std::vector<double> cur_xstar;
    std::vector<double> cur_noise;
    std::vector<double> cur_noise_MC;
    int cur_xstar_idx = -1;
    int cur_xstar_idx_in_gp_active_set = -1;
    bool print_debug_information = false;

    bool is_latest_index_active_index = false;
    for(int i = 0; i < evaluations.active_index.size(); ++i){
      if(evaluations.active_index[i] == evaluations.nodes.size()-1){
        is_latest_index_active_index = true;
        break;
      }
    }
    if(!is_latest_index_active_index){
      evaluations.active_index.push_back( evaluations.nodes.size()-1 );
    }
    bool is_best_index_active_index = false;
    for(int i = 0; i < evaluations.active_index.size(); ++i){
      if(evaluations.active_index[i] == evaluations.best_index){
        is_best_index_active_index = true;
        break;
      }
    }
    if(!is_best_index_active_index){
      evaluations.active_index.push_back( evaluations.best_index );
    }

    best_index_analytic_information.clear();
    best_index_analytic_information.resize(number_processes);
    for ( int i = 0; i < number_processes; i++) {
      best_index_analytic_information[i].resize(25);
    }

    /////////////////
    //double fill_width = compute_fill_width(evaluations);
    bool do_bootstrap_estimation = false;
    if ( (smoothing_ctr++) % 10 == 0){
      bootstrap_estimate.resize(number_processes);
      do_bootstrap_estimation = true;
    }
    assert(bootstrap_estimate.size() == number_processes);
    for ( int j = 0; j < number_processes && do_bootstrap_estimation; ++j ) {
      gaussian_processes[j]->build_inverse();
      cur_xstar_idx = evaluations.best_index;
      cur_xstar = evaluations.nodes[cur_xstar_idx];
      //cur_noise_MC = evaluations.noise_MC[j];
      //Var[R] is squared standard error: (SE[R])^2
      //evaluations.noise however is 2 * SE
      //Thus, divide by two
      //for (double &noise_ctr : cur_noise_MC) {
      //  noise_ctr /= 2.;
      //}
      //cur_noise_xstar_MC = cur_noise_MC[cur_xstar_idx];
      //var_GP = gaussian_processes[j]->compute_var_meanGP(cur_xstar, cur_noise_MC);

      //Pick the samples for the active indices
      active_index_samples.resize(gaussian_process_active_index.size( ));
      for(int i = 0; i < gaussian_process_active_index.size( ); ++i){
        active_index_samples[i].resize(evaluations.samples[j][gaussian_process_active_index[i]].size());
        for(int k = 0; k < active_index_samples[i].size(); ++k){
          active_index_samples[i][k] = evaluations.samples[j][gaussian_process_active_index[i]][k];
        }
      }
      bootstrap_estimate[j] = gaussian_processes[j]->bootstrap_diffGPMC(cur_xstar, active_index_samples, j, 100);
      //bootstrap_squared = bootstrap_diffGPRf*bootstrap_diffGPRf;
      //upper_bound_constant_estimate[j] = (bootstrap_squared + var_GP)/fill_width;
    }
    //std::cout << "###FILLWIDTH " << fill_width << " #UpperBound: ";
    //for ( int j = 0; j < number_processes; ++j ) {
    //  std::cout << upper_bound_constant_estimate[j] << " ";
    //}
    //std::cout << std::endl;
    //////////////////

    for ( int j = 0; j < number_processes; ++j ) {
      if(!do_bootstrap_estimation) { //Otherwise we have built it already for the bootstrap estimation, see above
        gaussian_processes[j]->build_inverse();
      }

      //Pick the samples for the active indices
      //active_index_samples.resize(gaussian_process_active_index.size( ));
      //for(int i = 0; i < gaussian_process_active_index.size( ); ++i){
      //  active_index_samples[i].resize(evaluations.samples[j][gaussian_process_active_index[i]].size());
      //  for(int k = 0; k < active_index_samples[i].size(); ++k){
      //    active_index_samples[i][k] = evaluations.samples[j][gaussian_process_active_index[i]][k];
      //  }
      //}

      for ( unsigned int i = 0; i < evaluations.active_index.size( ); ++i ) {
        cur_xstar_idx = evaluations.active_index[i];

        cur_xstar_idx_in_gp_active_set = -1;
        for ( unsigned int k = 0; k < gaussian_process_active_index.size( ); ++k ) {
          if (cur_xstar_idx == gaussian_process_active_index[k]){
            cur_xstar_idx_in_gp_active_set = k;
            break;
          }
        }
        assert(cur_xstar_idx_in_gp_active_set != -1);

        cur_xstar = evaluations.nodes[cur_xstar_idx];


        cur_noise_MC = evaluations.noise_MC[j];
        //Var[R] is squared standard error: (SE[R])^2
        //evaluations.noise however stores the value (2 * SE[R])
        //Thus, divide by two before squaring below
        for (double &noise_ctr : cur_noise_MC) {
          noise_ctr /= 2.;
        }
        cur_noise_xstar_MC = cur_noise_MC[cur_xstar_idx];

        gaussian_processes[j]->evaluate( cur_xstar, mean, variance );

        if(print_debug_information){
        std::cout << "############################"
                  << "\ncur_xstar_idx: " << cur_xstar_idx
                  << "\ncur_value: (" << evaluations.values[ j ][ cur_xstar_idx ]
                                   << ", " << evaluations.values[ j ][ evaluations.active_index [ i ] ] << ")";
        std::cout << "]\ncur_noise_xstar: " << cur_noise_xstar
                  << "\ncur_xstar: [";
                  for(double n : cur_xstar) {
                    std::cout << n << ' ';
                  }
                  std::cout << "]"<< std::endl;
        }
        assert(!std::isnan(cur_noise_xstar));
        var_R = cur_noise_xstar_MC * cur_noise_xstar_MC;
        cov_RGP = gaussian_processes[j]->compute_cov_meanGPMC(cur_xstar, cur_xstar_idx_in_gp_active_set, cur_noise_xstar_MC);
        var_GP = gaussian_processes[j]->compute_var_meanGP(cur_xstar, cur_noise_MC);
        //bootstrap_diffGPRf = gaussian_processes[j]->bootstrap_diffGPMC(cur_xstar, active_index_samples, j, 100);
        //bootstrap_squared = bootstrap_diffGPRf*bootstrap_diffGPRf;

        bootstrap_squared = bootstrap_estimate[j]*bootstrap_estimate[j];
        //bootstrap_squared = (mean - evaluations.values[ j ][cur_xstar_idx])*(mean - evaluations.values[ j ][cur_xstar_idx]);

        double upper_bound_cov = sqrt(var_R * var_GP);
        int set_upper_bound_cov_flag = 0;
        if(abs(cov_RGP) > upper_bound_cov){
          cov_RGP = cov_RGP >= 0 ? upper_bound_cov : -upper_bound_cov;
          set_upper_bound_cov_flag = 1;
        }

        numerator = var_R - cov_RGP;
        denominator = bootstrap_squared + var_GP + var_R - 2.0 * cov_RGP;
        //denominator = upper_bound_constant_estimate[j] + var_R - 2.0 * cov_RGP;

        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][0] = var_R;
          best_index_analytic_information[j][1] = cov_RGP;
          best_index_analytic_information[j][2] = var_GP;
          best_index_analytic_information[j][3] = bootstrap_squared;
          best_index_analytic_information[j][4] = numerator;
          best_index_analytic_information[j][5] = denominator;
          best_index_analytic_information[j][24] = set_upper_bound_cov_flag;
        }

        optimal_gamma = (std::fabs(denominator) <= DBL_MIN) ? 0.0 : numerator/denominator;
        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][6] = optimal_gamma;
        }
        optimal_gamma = (optimal_gamma > 1.0) ? 1.0 : optimal_gamma;
        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][7] = optimal_gamma;
        }
        optimal_gamma = (optimal_gamma < 0.0 || std::isnan(optimal_gamma)) ? 0.0 : optimal_gamma;
        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][8] = optimal_gamma;
        }

        if(print_debug_information){
        std::cout << "\nvar_R: " << var_R
                  << "\ncov_RGP: " << cov_RGP
                  << "\nvar_GP: " << var_GP
                  << "\nbootstrap_diffGPRf " << sqrt(bootstrap_squared)
                  << "\nR " << evaluations.values[ j ][ cur_xstar_idx ]
                  << "\nmean_GP " << mean
                  << "\noptimal_gamma " << optimal_gamma;
        }

        optimal_gamma_squared = optimal_gamma*optimal_gamma;
        MSE = optimal_gamma_squared * bootstrap_squared +
              optimal_gamma_squared * var_GP +
              (1.0 - optimal_gamma) * (1.0 - optimal_gamma) * var_R +
              2.0 * optimal_gamma * (1.0 - optimal_gamma) * cov_RGP;
        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][9] = MSE;
        }

        if(std::isnan(MSE)){
          if(evaluations.active_index[i] == evaluations.best_index){
            best_index_analytic_information[j][10] = 1;
          }
          optimal_gamma = 0.0;
          MSE = var_R;
        }
        if(!(MSE > 0 && MSE < var_R)){
          if(evaluations.active_index[i] == evaluations.best_index){
            best_index_analytic_information[j][11] = 1;
          }
          if(print_debug_information){
            std::cout << "\nMSE: " << MSE;
            std::cout << "\nMSE Reset" << std::endl;
          }
          optimal_gamma = 0.0;
          MSE = var_R;
        }
        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][12] = MSE;
        }

        RMSE = sqrt(MSE);
        Rtilde = optimal_gamma * mean + (1.0 - optimal_gamma) *
                 ( evaluations.values_MC[ j ][ cur_xstar_idx ] );

        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][13] = mean;
        }
        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][14] = evaluations.values_MC[ j ][ cur_xstar_idx ];
        }
        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][15] = Rtilde;
        }
        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][16] = RMSE;
        }

        heuristic_gamma = exp( - 2e0*sqrt(variance) );
        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][17] = heuristic_gamma;
        }
        heuristic_Rtilde =
            heuristic_gamma * mean  +
            (1e0-heuristic_gamma) * ( evaluations.values_MC[ j ][evaluations.active_index [ i ] ] );
        heuristic_RMSE =
            heuristic_gamma * 2e0 * sqrt (variance)  +
            (1e0-heuristic_gamma) * ( evaluations.noise_MC[ j ][ evaluations.active_index [ i ] ] );

        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][18] = heuristic_Rtilde;
        }

        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][19] = heuristic_RMSE;
        }

        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][20] = variance;
        }

        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][21] = evaluations.noise_MC[ j ][ evaluations.active_index [ i ] ];
        }
        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][22] = gaussian_processes[j]->get_gp_parameters()[1];
        }

        if(evaluations.active_index[i] == evaluations.best_index){
          best_index_analytic_information[j][23] = gaussian_processes[j]->get_gp_parameters()[2];
        }

        if(print_debug_information){
          std::cout << "\nRMSE: " << RMSE
                    << "\nHeuristic RMSE: " << heuristic_RMSE << std::endl;
        }
        evaluations.values[ j ][ cur_xstar_idx ] = Rtilde; // RMSE < heuristic_RMSE ? Rtilde : heuristic_Rtilde;

        evaluations.noise[ j ][ cur_xstar_idx ] = 2 * RMSE;// RMSE < heuristic_RMSE ? RMSE : heuristic_RMSE;

        if(print_debug_information){
          std::cout << "\nMSE: " << MSE
                    << "\nRtilde: " << evaluations.values[ j ][ cur_xstar_idx ]
                    << "\nRtildeNoise: " << evaluations.noise[ j ][ cur_xstar_idx ]
                    << "\n############################" << std::endl;
        }
        assert(!std::isnan(evaluations.values[ j ][ cur_xstar_idx ]));
        assert(!std::isnan(evaluations.noise[ j ][ cur_xstar_idx ]));
      }
    }
    if(!is_latest_index_active_index){
      evaluations.active_index.erase( evaluations.active_index.end()-1 );
    }
    if(!is_best_index_active_index){
      evaluations.active_index.erase( evaluations.active_index.end()-1 );
    }
  }else{
    do{
      negative_variance_found = false;

      bool is_latest_index_active_index = false;
      for(int i = 0; i < evaluations.active_index.size(); ++i){
        if(evaluations.active_index[i] == evaluations.nodes.size()-1){
          is_latest_index_active_index = true;
          break;
        }
      }
      if(!is_latest_index_active_index){
        evaluations.active_index.push_back( evaluations.nodes.size()-1 );
      }
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

          evaluations.values[ j ][ evaluations.active_index [ i ] ] =
            weight * mean  +
            (1e0-weight) * ( evaluations.values_MC[ j ][ evaluations.active_index [ i ] ]);
          evaluations.noise[ j ][ evaluations.active_index [ i ] ] =
            weight * 2e0 * sqrt (variance)  +
            (1e0-weight) * ( evaluations.noise_MC[ j ][ evaluations.active_index [ i ] ] );
          //std::cout << "Smooth evaluate var: [" << noise[ j ].at( evaluations.active_index [ i ]) << ", " << 2e0 *sqrt(variance) << "]\n" << std::endl;
         //std::cout << "Smooth evaluate [" << evaluations.active_index[i] << ", " << j <<"]: mean,variance " << evaluations.values[ j ].at( evaluations.active_index [ i ] ) << ", " << evaluations.noise[ j ].at( evaluations.active_index [ i ] )  << "\n" << std::endl;
        }
        if (variance < 0e0){
            negative_variance_found = true;
            break;
        }
      }

      if(!is_latest_index_active_index){
        evaluations.active_index.erase( evaluations.active_index.end()-1 );
      }
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
  }
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

const std::vector<std::vector<double>> &GaussianProcessSupport::getBest_index_analytic_information() const {
  return best_index_analytic_information;
}

typedef struct {
    const std::vector<double> cons_best_node;
    const double cons_gaussian_process_delta;
} constraint_data;

typedef struct {
    const std::vector<std::vector<double>> obj_gaussian_process_nodes;
} objective_data;

double GaussianProcessSupport::compute_fill_width(BlackBoxData& evaluations){

  assert(gaussian_process_nodes.size() > 0);
  int dim = gaussian_process_nodes[0].size();
  //initialize optimizer from NLopt library
  std::vector<double> best_node = evaluations.nodes[evaluations.best_index];

  constraint_data cons_data = { best_node, gaussian_process_delta_factor };
  objective_data obj_data = { gaussian_process_nodes };

  std::vector<double> lb(dim);
  std::vector<double> ub(dim);
  for(int i = 0; i < dim; ++i){
    lb[i] = best_node[i] - 1.0001*gaussian_process_delta_factor*delta_tmp;
    ub[i] = best_node[i] + 1.0001*gaussian_process_delta_factor*delta_tmp;
  }

//  nlopt::opt opt(nlopt::LD_CCSAQ, dimp1);
//  nlopt::opt opt(nlopt::LN_BOBYQA, dimp1);
//
  nlopt::opt opt(nlopt::GN_DIRECT, dim);

  //opt = nlopt_create(NLOPT_LN_COBYLA, dim+1);
  opt.set_lower_bounds( lb );
  opt.set_upper_bounds( ub );

  //std::vector<double> tol = {1};
  //opt.add_inequality_mconstraint(GaussianProcessSupport::ball_constraint, &cons_data, tol);
  opt.set_max_objective( GaussianProcessSupport::fill_width_objective, &obj_data);

  // opt.set_xtol_abs(1e-2);
//  opt.set_xtol_rel(1e-2);
//set timeout to NLOPT_TIMEOUT seconds
  opt.set_maxtime(1.0);
  opt.set_maxeval(1000);
  //perform optimization to get correction factors

  int exitflag=-20;
  double optval;
  try {
    exitflag = opt.optimize(best_node, optval);
  } catch (...) {
    std::cout << "Something went wrong in fill width optimization: " << exitflag << std::endl;
  }
  return optval;
}

double GaussianProcessSupport::fill_width_objective(std::vector<double> const &x,
                                   std::vector<double> &grad,
                                   void *data){
  auto obj_data = reinterpret_cast< objective_data* >(data);

  std::vector<std::vector<double>> gp_nodes = obj_data->obj_gaussian_process_nodes;
  double min_fill_width = DBL_MAX;
  double cur_fill_width = 0.0;
  for(int i = 0; i < gp_nodes.size(); ++i){
    cur_fill_width = 0.0;
    for(int j = 0; j < gp_nodes[i].size(); ++j){
      cur_fill_width += (x[j] - gp_nodes[i][j]) * (x[j] - gp_nodes[i][j]);
    }
    if(cur_fill_width < min_fill_width){
      min_fill_width = cur_fill_width;
    }
  }
  return sqrt(min_fill_width);
}

void GaussianProcessSupport::ball_constraint(unsigned int m, double* c, unsigned n, const double *x, double *grad, void *data){
  auto cons_data = reinterpret_cast<constraint_data*>(data);

  double distance = 0.0;
  for(int i = 0; i < n ; ++i){
    distance += (x[i] - cons_data->cons_best_node[i])*(x[i] - cons_data->cons_best_node[i]);
  }
  distance = std::sqrt(distance);
  c[0] = distance - cons_data->cons_gaussian_process_delta;
}

void GaussianProcessSupport::set_gaussian_process_delta(double gaussian_process_delta) {
  GaussianProcessSupport::gaussian_process_delta_factor = gaussian_process_delta;
}
/*
void GaussianProcessSupport::do_resample_u(){
  for ( int j = 0; j < number_processes; ++j ) {
      gaussian_processes[j]->do_resample_u();
    }
}
*/
//--------------------------------------------------------------------------------
