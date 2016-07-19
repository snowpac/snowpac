#ifndef HMinimumFrobeniusNormModel
#define HMinimumFrobeniusNormModel

#include "BasisForMinimumFrobeniusNormModel.hpp"
#include "SurrogateModelBaseClass.hpp"
//#include "BlackBoxData.hpp"
#include "VectorOperations.hpp"
#include <Eigen/Dense>
#include <vector>

class MinimumFrobeniusNormModel : public SurrogateModelBaseClass,
                                  protected VectorOperations {     
  private:
    int size;
    int dim;
  public:
    MinimumFrobeniusNormModel ( BasisForMinimumFrobeniusNormModel& );
    double evaluate ( std::vector<double> const& );
    std::vector<double> &gradient ( std::vector<double> const& );
    std::vector<double> &gradient ( );
    std::vector< std::vector<double> > &hessian ( std::vector<double> const& );
    std::vector< std::vector<double> > &hessian ( );
    void set_function_values ( std::vector<double> const& );
};

#endif
