#ifndef HTriangularMatrixOperations
#define HTriangularMatrixOperations

#include <vector>
#include "math.h"

//! Forward and backward substiutions with lower triangular matrices
/*!
 Forward and backward substituion with a lower triangular system matrix.
 The vector of the right-hand side is over-written with the solution of the linear system.
*/
class TriangularMatrixOperations {
  private: 
    int dim;
    double tmp;
    // auxiliary variables for compute_large_norm_solutio
    double c, c1, c2;                //hold cosine value
    double s, s1, s2;                //hold sine value
    double tDDt, pDDp, pDDt;         //hold intermediate vector/matrix/vector products
    double yy;                       //holds partial inner product of y with itself
    double beta, alpha, r, mu;       //auxiliary variables to compute c and s
    double phi1, phi2;               //auxiliary variables to compute c and s
    double y1, y2;                   //temporary place-holders for next component of y
    std::vector<double> w, d, p;
//    Eigen::VectorXd d, p;//, tmp_ev;
  public:
    //! Constructor 
    /*! 
     Set dimension of linear systems
     \param n dimension of linear systems
    */
    TriangularMatrixOperations ( int );
    //! Destructor
    ~TriangularMatrixOperations ( ) { }
    //! Forward substituion
    /*! Solves the linear system Ly = x for a lower triangular matrix L
        The input vextor x contains the right-hand side on input and the solution on output
        \param L Lower triangular matrix
        \param x Vector of right-hand side on input, solution vector on output
    */
    void forward_substitution ( std::vector< std::vector<double> > const&,
                                std::vector<double> & );
    //! Forward substituion
    /*! Solves the linear system Ly = x for a lower triangular matrix L
        The input vextor x contains the right-hand side on input and the solution on output
        \param L Lower triangular matrix
        \param x Vector of right-hand side on input, solution vector on output
    */
    void forward_substitution_not_inplace ( std::vector< std::vector<double> > const&,
                                std::vector<double> const& , std::vector<double> &);
    //! Backward substituion
    /*! Solves the linear system L'y = x for a lower triangular matrix L
        The input vextor x contains the right-hand side on input and the solution on output
        \param L Lower triangular matrix
        \param x Vector of right-hand side on input, solution vector on output
    */
    void backward_substitution ( std::vector< std::vector<double> > const&,
                                 std::vector<double>& );
    //! Backward substituion
    /*! Solves the linear system L'y = x for a lower triangular matrix L
        The input vextor x contains the right-hand side on input and the solution on output
        \param L Lower triangular matrix
        \param x Vector of right-hand side on input, solution vector on output
    */
    void backward_substitution_not_inplace ( std::vector< std::vector<double> > const&,
                                         std::vector<double> const&, std::vector<double>& );
    //! Look behind algorithm (Cline et al. 1982)
    /*!
     Computes an approximation of the largest norm solution y of the linear system Ly = p
     for a vector ||p|| = 1.\n
     Algorithm implemented from Cline et al., Generalizing the LINPACK condition estimator, 1982.
     \param L lower triangular matrix
     \param y on output the solution an approximation to the largest norm solution
    */
    void compute_large_norm_solution ( std::vector< std::vector<double> > const&, 
                                       std::vector<double>& );
};

#endif
