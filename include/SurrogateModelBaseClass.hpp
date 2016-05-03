#ifndef HSurrogateModelBaseClass
#define HSurrogateModelBaseClass

#include "BasisForSurrogateModelBaseClass.hpp"

class SurrogateModelBaseClass {
  protected: 
    std::vector<double> model_gradient;
    std::vector<double> function_values;
    BasisForSurrogateModelBaseClass *basis;
  public:
    SurrogateModelBaseClass ( BasisForSurrogateModelBaseClass &basis_input ) 
                            { basis = &basis_input; }
    virtual double evaluate ( std::vector<double> const& ) = 0;    
    virtual std::vector<double> &gradient ( std::vector<double> const& ) = 0;
    virtual void set_function_values ( std::vector<double> const&,
                                       std::vector<double> const&,
                                       std::vector<int> const& ) = 0;
    int dimension ( ) { return basis->dimension( ); }
};

#endif
