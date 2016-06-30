#ifndef HTransformActiveNodes
#define HTransformActiveNodes

#include <vector>

//! Class to trasform node
/*!
  Class to transform node x into y = s(x-c) with a center node c and a scaling factor s
*/
class TransformNode {
  private:
    double *s;
    std::vector<double> *c;
  public:
    std::vector<double> &transform ( std::vector<double> &x, std::vector<double>&, double);
};


#endif
