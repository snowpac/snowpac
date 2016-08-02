#ifndef HNoiseDetection
#define HNoiseDetection

#include "SurrogateModelBaseClass.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include "math.h"

template<class TSurrogateModel>
class NoiseDetection : protected VectorOperations {
  private:
    std::vector<TSurrogateModel> *surrogate_models;
    double *delta;
    int observation_span;
    int nb_allowed_noisy_iterations;
    int nb_surrogate_models;
    std::vector<int> noisy_iterations_counter;
    std::vector<double> tr_radii;
    std::vector< std::vector<double> > norms_of_hessians;
    std::vector<double> sum_norms;
    std::vector<double> sum_norms_tr_radii;
    std::vector< double > exponents;
    double threshold;
    double sum_delta, sum_sq_delta;
    double nb_observations;
    double logdelta;
    double lognorm_dbl;
    std::vector<double> lognorm;
    bool delete_first;
    int tmp_int;
  protected:
    bool noise_has_been_detected;
    NoiseDetection ( std::vector<TSurrogateModel> &surrogate_models_input,
                     double &delta_input ) 
                   { delta = &delta_input;
                     surrogate_models = &surrogate_models_input;
                     threshold = 1e0; 
                     nb_surrogate_models = 0;
                     noise_has_been_detected = false;
                   }

//--------------------------------------------------------------------------------
    void initialize_noise_detection( 
      int observation_span_input, int nb_allowed_noisy_iterations_input ) 
    {
      observation_span = observation_span_input;
      nb_allowed_noisy_iterations = nb_allowed_noisy_iterations_input;
      nb_surrogate_models = (*surrogate_models).size();
      norms_of_hessians.resize( nb_surrogate_models );;
      sum_norms_tr_radii.resize( nb_surrogate_models );
      sum_norms.resize( nb_surrogate_models );
      noisy_iterations_counter.resize( nb_surrogate_models );
      exponents.resize( nb_surrogate_models );
      lognorm.resize( nb_surrogate_models);
      for ( int i = 0; i < nb_surrogate_models; ++i ) {
        sum_norms.at(i)                = 0e0;
        sum_norms_tr_radii.at(i)        = 0e0;
        noisy_iterations_counter.at(i) = 0;
      }
      sum_delta = 0e0;
      sum_sq_delta = 0e0;
      noise_has_been_detected = false;
      return;
    }
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
    void reset_noise_detection( ) 
    {
      tr_radii.clear();
      norms_of_hessians.clear();
      norms_of_hessians.resize( nb_surrogate_models );
      for ( int i = 0; i < nb_surrogate_models; ++i ) {
        sum_norms.at(i)                = 0e0;
        sum_norms_tr_radii.at(i)       = 0e0;
        noisy_iterations_counter.at(i) = 0;
      }
      sum_delta = 0e0;
      sum_sq_delta = 0e0;
      delete_first = false;
      noise_has_been_detected = false;
      noise_has_been_detected = detect_noise( );
      return;
    }
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
    bool detect_noise( ) 
    {

      for ( int i = 0; i < nb_surrogate_models; ++i ) { 
        lognorm[ i ] = norm( (*surrogate_models)[i].hessian() ) / ((*delta)*(*delta));
        if ( fabs(lognorm [ i ]) < 1e-16 ) { 
          return false;
        }
      }

      tr_radii.push_back( *delta );
      if ( tr_radii.size() > observation_span ) delete_first = true;
      if ( delete_first ) { 
        logdelta      = log(tr_radii[0]); 
        sum_delta    -= logdelta;
        sum_sq_delta -= logdelta*logdelta;
        for ( int i = 0; i < nb_surrogate_models; ++i ) { 
          lognorm_dbl = log(norms_of_hessians[i][0]);
          sum_norms_tr_radii[i] -= lognorm_dbl*logdelta;
          sum_norms[i]          -= lognorm_dbl;
          norms_of_hessians[i].erase( norms_of_hessians[i].begin() );
        }
        tr_radii.erase( tr_radii.begin() );
      }
      logdelta = log(*delta);
      sum_delta += logdelta;
      sum_sq_delta += logdelta*logdelta;
 
      nb_observations = (double) tr_radii.size();

      for ( int i = 0; i < nb_surrogate_models; ++i ) { 
        norms_of_hessians[i].push_back( lognorm[i] );
        lognorm[i] = log( lognorm[i] );
        sum_norms_tr_radii[i] += lognorm[i] * logdelta;
        sum_norms[i]          += lognorm[i];
        if ( nb_observations == observation_span ) {
          exponents[i] = ( sum_norms_tr_radii[i] - sum_norms[i] * sum_delta / nb_observations );
          exponents[i] *= nb_observations;
          exponents[i] /= (nb_observations * sum_sq_delta - sum_delta * sum_delta);
          if ( exponents[i] < -threshold ) noisy_iterations_counter[i]++;
          if ( noisy_iterations_counter[i] > nb_allowed_noisy_iterations ) 
            noise_has_been_detected = true;
        }
      }
      return noise_has_been_detected;
    }
//--------------------------------------------------------------------------------


};




#endif
