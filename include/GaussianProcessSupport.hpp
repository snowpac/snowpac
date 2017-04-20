#ifndef HGaussianProcessSupport
#define HGaussianProcessSupport

#include "GaussianProcess.hpp"
#include "BlackBoxData.hpp"
#include "VectorOperations.hpp"
#include <vector>

class GaussianProcessSupport : protected VectorOperations {
  private:
    double *delta;
    bool do_parameter_estimation = false;
    int number_processes;
    int nb_values = 0;
    std::vector<int> update_at_evaluations;
    int update_interval_length;
    int next_update = 0;
    int last_included = 0;
    int best_index;
    double delta_tmp;
    double variance, mean; 
    double weight;
    std::vector<double> gaussian_process_values;
    std::vector<double> gaussian_process_noise;
    std::vector< std::vector<double> > gaussian_process_nodes;
    std::vector<int> gaussian_process_active_index; 
    std::vector< std::vector<double> > values;
    std::vector< std::vector<double> > noise;
    std::vector<std::shared_ptr<GaussianProcess>> gaussian_processes;
    std::vector<double> rescaled_node;

    bool use_approx_gaussian_process = false;
    bool approx_gaussian_process_active = false;
    const double u_ratio = 0.15;
    const int min_nb_u = 2;
    int cur_nb_u_points = 0;

    int NOEXIT;

    void update_data ( BlackBoxData& );
    void update_gaussian_processes ( BlackBoxData& );

    void update_gaussian_processes_for_agp( BlackBoxData&);
    void update_gaussian_processes_for_gp (BlackBoxData&);

    //void do_resample_u(); //TODO check if able to remove this one
  public:
    void initialize ( const int, const int, double&,
                      std::vector<double> const&, int , const std::string, const int exitconst);
    int smooth_data ( BlackBoxData& );
    double evaluate_objective ( BlackBoxData const& );

    void evaluate_gaussian_process_at(const int&, std::vector<double> const&, double&, double&);

    const std::vector<std::vector<double>> &get_nodes_at(const int&) const;

    void get_induced_nodes_at(const int idx, std::vector<std::vector<double>> &induced_nodes);

    void set_constraint_ball_center(const std::vector<double>& center);

    void set_constraint_ball_radius(const double& radius);
};

#endif
