#ifndef HBlackBoxBaseClass
#define HBlackBoxBaseClass

#include <vector>

class BlackBoxBaseClass {
  public:
    // black box arguments:
    // 1) <node to be evaluated>
    // 2) <return_values> 0=objective value, 1..=constraint values
    // 3) <noise estimate> 0=objective value, 1..=constraint values
    virtual void evaluate ( std::vector<double> const &x, std::vector<double> &vals,
                            void *param) { return; }
    virtual void evaluate ( std::vector<double> const &x, std::vector<double> &vals,
                            std::vector<double> &noise, void *param ) { return; }

};

#endif
