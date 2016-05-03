#ifndef HRegularizedMinimumFrobeniusNormModel
#define HRegularizedMinimumFrobeniusNormModel

#include "BasisForMinimumFrobeniusNormModel.hpp"
#include "SurrogateModelBaseClass.hpp"
#include "VectorOperations.hpp"
#include "BlackboxData.hpp"
#include <Eigen/Dense>
#include <vector>
#include "nlopt.hpp"

class RegularizedMinimumFrobeniusNormModel : public SurrogateModelBaseClass,
                                             public VectorOperations {
  private:
    Eigen::MatrixXd M;
    std::vector<double> noise_values;
    void regularize_coefficients ( );
    std::vector<double> parameters;
    std::vector<double> lb, ub;
    double res;
    int size;
  public:
    RegularizedMinimumFrobeniusNormModel ( BasisForMinimumFrobeniusNormModel& );
    double evaluate ( std::vector<double> const& );
    std::vector<double> &gradient ( std::vector<double> const& );
    void set_function_values ( std::vector<double> const&, std::vector<double> const&,
                               std::vector<int> const& );
    static double regularization_objective(std::vector<double> const&,
                                            std::vector<double>&, void*);
};

#endif
