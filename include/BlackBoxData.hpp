#ifndef HBlackboxData
#define HBlackboxData

#include <Eigen/Core>
#include <vector>

struct BlackboxData {
  int max_nb_nodes;
  int best_index;
  std::vector< std::vector<double> > nodes;
  std::vector< std::vector<double> > values;
  std::vector< std::vector<double> > noise;
  std::vector<int> surrogate_nodes_index;
};

#endif
