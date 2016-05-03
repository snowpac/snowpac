#ifndef HGaussianProcessSupport
#define HGaussianProcessSupport

#include "GaussianProcess.hpp"
#include "BlackBoxData.hpp"
#include "VectorOperations.hpp"
#include <vector>

class GaussianProcessSupport : protected VectorOperations {
  private:
    double *delta;
    bool do_parameter_estimation = true;
    int number_processes;
    int nb_values = 0;
    std::vector<int> update_at_evaluations;
    int update_interval_length;
    int next_update = 0;
    int last_included;
    int best_index;
    double delta_tmp;
    double variance, mean; 
    double weight;
    Eigen::VectorXd gaussian_process_values;
    Eigen::VectorXd gaussian_process_noise;
    std::vector< std::vector<double> > gaussian_process_nodes;
    std::vector<int> gaussian_process_active_index; 
    std::vector< std::vector<double> > values;
    std::vector< std::vector<double> > noise;
    std::vector<GaussianProcess> gaussian_processes;
    std::vector<double> rescaled_node;
    void update_data ( BlackboxData& );
    void update_gaussian_processes ( BlackboxData& );
  public:
    void initialize ( const int, const int, double&,
                      std::vector<double> const&, int );
    void smooth_data ( BlackboxData& );    
    double evaluate_objective ( BlackboxData const& );    
};

#endif
