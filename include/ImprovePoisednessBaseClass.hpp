#ifndef HImprovePoisednessBaseClass
#define HImprovePoisednessBaseClass

#include "BlackboxData.hpp"
#include "BasisForSurrogateModelBaseClass.hpp"
#include <Eigen/Dense>
#include <vector>

class ImprovePoisednessBaseClass {
  protected:
    BasisForSurrogateModelBaseClass *basis;
    double threshold_for_poisedness_constant;
    double poisedness_constant;
    std::vector<bool> index_of_changed_nodes;
  public:
    ImprovePoisednessBaseClass ( double threshold_for_poisedness_constant_input, 
                                 BasisForSurrogateModelBaseClass &basis_input ) :
                                 threshold_for_poisedness_constant( 
                                   threshold_for_poisedness_constant_input ),
                                 poisedness_constant ( 0e0 ) { basis = &basis_input; }
    virtual int replace_node ( int, BlackboxData const&, std::vector<double> const& ) = 0;
    virtual void improve_poisedness ( int, BlackboxData& ) = 0;
};

#endif