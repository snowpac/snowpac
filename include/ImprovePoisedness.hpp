#ifndef HImprovePoisedness
#define HImprovePoisedness

#include "ImprovePoisednessBaseClass.hpp"
#include "BlackboxData.hpp"
#include "CholeskyFactorization.hpp"
#include "VectorOperations.hpp"
#include "QuadraticMinimization.hpp"
#include <Eigen/Dense>
#include <vector>
#include <math.h>

//! Improve poisedness of interpolation nodes
class ImprovePoisedness : public ImprovePoisednessBaseClass,
                          protected QuadraticMinimization {
  private:
    int dim;
    int nb_nodes;
    size_t max_nb_nodes;
    double *delta;

    //declare auxiliary variables for replace_node
    double maxvalue, LK, norm;
    //declare auxiliary variables for compute_poisedness_constant
    double poisedness_constant_tmp1, poisedness_constant_tmp2;
    double node_norm_scaling;
//    Eigen::VectorXd q;
    std::vector<double> q1, q2;
    std::vector<double> basis_values;
    std::vector<double> basis_gradient;
    std::vector< std::vector<double> > basis_hessian;
    //define auxiliary variables
    Eigen::VectorXd tmp_node;
    bool print_output;
    int change_index;    
    bool model_has_been_improved;
    void compute_poisedness_constant ( int, std::vector<double>&, BlackboxData& );
  public:
    //! Constructor
    /*!
     Set parameters required for the improvement of the poisedness of interploation nodes
     \param B basis for surrogate model
     \param poisedness_threshold threshold for poisedness constant
     \param m maximal number of interpolation nodes
     \param rad radius arround current best point of ball that contains well poised points
     \param verbose switch output on (verbose = 3) or off (verbose = 0)
     \see BlackboxData
    */
    ImprovePoisedness ( BasisForSurrogateModelBaseClass&, double, int, double&, int );
    //! Destructor
    ~ImprovePoisedness () { }
    //! Find node to be replaced by better poised node
    /*!
     Finds a node to replace another interpolation node to improve poisedness
     \param reference_node index of node that is not replaced
     \param evaluations interpolation nodes, \see BlackboxData
     \param new_node new node to replace an existing interpolation node
    */
    int replace_node ( int, BlackboxData const&, std::vector<double> const& );
    //! Improves poisedness of interpolation nodes
    /*!
     Improves poisedness of interpolation nodes by maximizing the absolute value of basis functions.\n
     Nodes to replace existing interpolation nodes are computed and appended to the list of nodes, 
     \see BlackboxData \n
     The index of nodes to reduce the poisedness value are indicated in evaluations \see BlackboxData \n
     \param reference_node index of node that is not replaced
     \param evaluations structure containing interpolation nodes, \see BlackboxData
    */
    void improve_poisedness ( int, BlackboxData& );
};

#endif
