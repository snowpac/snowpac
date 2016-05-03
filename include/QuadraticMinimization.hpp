#ifndef HQuadraticMinimization
#define HQuadraticMinimization

#include "VectorOperations.hpp"
#include "CholeskyFactorization.hpp"
#include "TriangularMatrixOperations.hpp"

//! Quadratic minimization in unit ball
class QuadraticMinimization : protected VectorOperations, 
                              private TriangularMatrixOperations,
                              protected CholeskyFactorization {
  private:
    int dim;
    //declare auxiliary variables for the More/Sorensen algorithm
    std::vector< std::vector<double> > A, M;
    std::vector<double> z_hat, u, uu;
    double tau, yzhat, syzhat, tmp_dbl, tmp_norm, tmp_save, udu, offset;
    int p;     //auxiliary variable for indication of positive-definiteness of the matrix in the cholesky factorization
    int counter;
    double sigma1, sigma2, norm_y, norm_g;
    double norm_g_d_delta, l1_norm_B, lam_S, lam_L, lam_U, lam;   
  public:
    //! Constructor
    /*!
     Contructor to set dimension of quadratic optimizatin problem
     \param n dimension of quadratic optimization problem
    */
    QuadraticMinimization ( int );
    //! Destructor
    ~QuadraticMinimization () {}
    //! Solve quadratic minimization in unit ball
    /*!
     Solves the quadratic minimization y = argmin g'x + 0.5x'Hx subject to ||x|| <= 1.\n
     Implementation of algorithm from More/Sorensen, Computing a trust region step (1983).
     \param y solution of quadratic optimization problem
     \param g gradient of quadratic opjective function
     \param H hessian of quadratic objective function
    */
    void minimize ( std::vector<double>&, std::vector<double> const&, 
                    std::vector< std::vector<double> > const& );
};

#endif
