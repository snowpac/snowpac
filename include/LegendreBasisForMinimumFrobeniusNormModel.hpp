#ifndef HLegendreBasisForMinimumFrobeniusNormModel
#define HLegendreBasisForMinimumFrobeniusNormModel

#include "BasisForSurrogateModelBaseClass.hpp"
#include "QuadraticLegendre.hpp"
#include "VectorOperations.hpp"
#include <Eigen/Dense>
#include <vector>
#include "math.h"


class LegendreBasisForMinimumFrobeniusNormModel : public BasisForSurrogateModelBaseClass, 
                                                  public QuadraticLegendre,
                                                  protected VectorOperations {
  private:
    int nb_nodes;
    Eigen::MatrixXd A_sysmat;
    Eigen::MatrixXd S_coeffsolve;
    Eigen::MatrixXd F_rhsmat;
    int counter;
    //! Evaluations of surrogate basis functions at nodes used to construct the basis
    std::vector<double> basis_values;
    std::vector<double> basis_constants;
    std::vector< std::vector<double> > basis_gradients; 
    std::vector< std::vector< std::vector<double> > > basis_Hessians;   
  public:
    LegendreBasisForMinimumFrobeniusNormModel ( int );
    void set_nb_nodes ( int );
    std::vector<double> &evaluate ( std::vector<double> const& );    
    double evaluate ( std::vector<double> const&, int);
    double &value( int );
    std::vector<double> &gradient ( int );
    std::vector< std::vector<double> > &hessian ( int );
    void compute_basis_coefficients ( std::vector< std::vector<double> > const& );
    void compute_mat_vec_representation ( int );

};

#endif
