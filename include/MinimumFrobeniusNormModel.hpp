#ifndef HMinimumFrobeniusNormModel
#define HMinimumFrobeniusNormModel

#include "BasisForSurrogateModelBaseClass.hpp"
#include "SurrogateModelBaseClass.hpp"
#include "VectorOperations.hpp"
#include <Eigen/Dense>
#include <vector>

class MinimumFrobeniusNormModel : public SurrogateModelBaseClass,
                                  protected VectorOperations {     
  private:
    int size;
    int dim;
    std::vector<double> matvecproduct;
  public:
    MinimumFrobeniusNormModel ( BasisForSurrogateModelBaseClass& );
    double evaluate ( std::vector<double> const& );
    std::vector<double> &gradient ( );
    std::vector< std::vector<double> > &hessian ( );
    void set_function_values ( std::vector<double> const& );
};

#endif
