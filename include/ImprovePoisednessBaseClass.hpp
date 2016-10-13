#ifndef HImprovePoisednessBaseClass
#define HImprovePoisednessBaseClass

#include "BlackBoxData.hpp"
#include "BasisForSurrogateModelBaseClass.hpp"
#include <Eigen/Dense>
#include <vector>

class ImprovePoisednessBaseClass {
  protected:
    BasisForSurrogateModelBaseClass *basis;
    double threshold_for_poisedness_constant;
    double poisedness_constant;

  public:
    ImprovePoisednessBaseClass ( double threshold_for_poisedness_constant_input, 
                                 BasisForSurrogateModelBaseClass &basis_input ) :
                                 threshold_for_poisedness_constant( 
                                   threshold_for_poisedness_constant_input ),
                                 poisedness_constant ( 0e0 ) { basis = &basis_input; }
    virtual int replace_node ( int, BlackBoxData&, std::vector<double> const& ) = 0;
    virtual void improve_poisedness ( int, BlackBoxData& ) = 0;
};

#endif
