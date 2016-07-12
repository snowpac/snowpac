#ifndef HRegularizedMinimumFrobeniusNormModel
#define HRegularizedMinimumFrobeniusNormModel

#include "BasisForMinimumFrobeniusNormModel.hpp"
#include "SurrogateModelBaseClass.hpp"
#include "VectorOperations.hpp"
//#include "BlackBoxData.hpp"
#include <Eigen/Dense>
#include <vector>
#include "nlopt.hpp"

struct RegularizationData {
    unsigned int dim;
    int best_index;
    BasisForSurrogateModelBaseClass *basis;
    VectorOperations *vo;
    std::vector<double> g;
    std::vector< std::vector<double> > H;
    std::vector< std::vector<double> > H_total;
    std::vector<double> g_total;
};

class RegularizedMinimumFrobeniusNormModel : public SurrogateModelBaseClass,
                                             public VectorOperations {
  private:
    Eigen::MatrixXd M;
    std::vector<double> noise_values;
    void regularize_coefficients ( );
//    std::vector<double> parameters;
    std::vector<double> lb, ub;
    double res;
    int size;
    RegularizationData rd;
    VectorOperations vo;
  public:
    RegularizedMinimumFrobeniusNormModel ( BasisForMinimumFrobeniusNormModel& );
    double evaluate ( std::vector<double> const& );
    std::vector<double> &gradient ( std::vector<double> const& );
    void set_function_values ( std::vector<double> const&, std::vector<double> const&,
                               std::vector<int> const&, int );
    static double regularization_objective(std::vector<double> const&,
                                           std::vector<double>&, void*);
};

#endif
