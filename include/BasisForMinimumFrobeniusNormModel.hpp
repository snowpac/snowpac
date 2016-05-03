#ifndef HBasisForMinimumFrobeniusNormModel
#define HBasisForMinimumFrobeniusNormModel

#include "BlackboxData.hpp"
#include "BasisForSurrogateModelBaseClass.hpp"
#include "QuadraticMonomial.hpp"
#include "VectorOperations.hpp"
#include <Eigen/Dense>
#include <vector>
#include "math.h"


class BasisForMinimumFrobeniusNormModel : public BasisForSurrogateModelBaseClass, 
                                          public QuadraticMonomial,
                                          protected VectorOperations {
  private:
    int nb_nodes;
    Eigen::MatrixXd A_sysmat;
    Eigen::MatrixXd S_coeffsolve;
    Eigen::MatrixXd F_rhsmat;
    std::vector<double> cache_point_basis_values;
    std::vector<double> cache_point_basis_gradients;
    bool clear_cache_values;
    bool clear_cache_gradients;
    int counter;
    double tmp_dbl;
    std::vector<double> scaled_node;
    std::vector<double> best_node;
    double *delta;
    std::vector< std::vector<double> > scaled_gradients;
    std::vector< std::vector< std::vector<double> > > scaled_Hessians;
    std::vector<double> tmp_vec;
    //! Evaluations of surrogate basis functions at nodes used to construct the basis
    std::vector<double> basis_values;
    //! Cached gradients of surrogate basis function
    /*! 
     \see compute_gradients
    */
    std::vector< std::vector<double> > basis_gradients;    
    //! Computes gradiens of all basis functions at \param x
    void compute_gradients ( std::vector<double> const& );
  public:
    BasisForMinimumFrobeniusNormModel ( int, double& );
    void set_nb_nodes ( int );
    std::vector<double> &evaluate ( std::vector<double> const& );    
    double evaluate ( std::vector<double> const&, int);
    std::vector<double> &gradient ( std::vector<double> const&, int );

    void compute_basis_coefficients ( BlackboxData const& );
    void get_mat_vec_representation ( int, std::vector<double>&,
                                      std::vector< std::vector<double> >&);
    void compute_mat_vec_representation ( int );

};

#endif