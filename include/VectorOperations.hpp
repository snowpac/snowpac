#ifndef HVectorOperations
#define HVectorOperations

#include <vector>

//--------------------------------------------------------------------------------
//! Vector operations
class VectorOperations {
  private:
    double dbl;
    int size, size1, size2;
  public:
    //! Setting vector to zero
    /*!
     Sets vector v to zero
     \param v on output a vector with elements zero
    */
    void set_zero ( std::vector<double>& );
    //! Scaling of vector
    /*!
     Computes w = s v 
     \param s scaling factor
     \param v input vector to be scaled
     \param w scaled vector s v
    */
    void scale ( double, std::vector<double> const&, 
                 std::vector<double>& );
    //! Scaling of matrix
    /*!
     Computes W = s V
     \param s scaling factor
     \param V input matrix to be scaled
     \param W scaled matrix s v
    */
    void scale ( double, std::vector< std::vector<double> > const&, 
                 std::vector< std::vector<double> >& );
    //! Adding scaled vector
    /*!
     Computes w = w + s v
     \param s scaling factor
     \param v input vector 
     \param w input and output vector
    */
    void add( double, std::vector<double> const&,
              std::vector<double> &);
    //! Adding vector
    /*!
     Computes w = w + v
     \param v input vector 
     \param w input and output vector
    */
    void add( std::vector<double> const&,
              std::vector<double> &);
    //! Adding scaled matrix
    /*!
     Computes W = W + s V
     \param s scaling factor
     \param V input matrix 
     \param W input and output matrix
    */
    void add( double, std::vector< std::vector<double> > const&,
              std::vector< std::vector<double> > &);
    //! Substracting two vectors
    /*!
     Computes w = v1 - v2
     \param v1 input vector
     \param v2 input vector
     \param output vector
    */
    void minus( std::vector<double> const&, std::vector<double> const&,
                std::vector<double>& );
    //! Rescaling and shifting vector 
    /*! 
     Computes w = (v1 - v2)s
     \param s scaling factor
     \param v1 vector to be rescaled and shifted
     \param v2 reference vector
     \param w output vector
    */
    void rescale( double, std::vector<double> const&, std::vector<double> const&,
                  std::vector<double>& );
    //! Norm of difference of vectors
    /*!
     Computes the 2 norm of the difference of two vectors
     \param v1 input vector
     \param v2 input vector
     \return norm of v1-v2
    */
    double diff_norm ( std::vector<double> const&, std::vector<double> const& );
    //! Dot product of two vectors
    /*!
     Computes the dot product of two vectors
     \param v1 input vector
     \param v2 input vector
     \return dot product of v1 and v2
    */
    double dot_product ( std::vector<double> const&, std::vector<double> const& );
    //! Transpose of matrix
    /*!
     Computes the product of two matrices
     \param V input matrix
     \param V_t transposed matrix
     \return V_t output matrix
    */
    void mat_transpose ( std::vector< std::vector<double> > const&, std::vector< std::vector<double> > &);
    //! Product of two matrices
    /*!
     Computes the product of two matrices
     \param V1 input matrix
     \param V2 input matrix
     \param W output matrix
     \return product of V1 and V2
    */
    void mat_product ( std::vector< std::vector<double> > const&, 
                       std::vector< std::vector<double> > const&,
                       std::vector< std::vector<double> > & );
    //! Product of matrix with vector
    /*!
     Computes the product of matrix with vector
     \param V1 input matrix
     \param v2 input vector
     \param w output vector
     \return product of V1 and v2
    */
    void mat_vec_product ( std::vector< std::vector<double> > const&, 
                           std::vector<double> const&,
                           std::vector<double> & );
    //! Product of vector^T with matrix
    /*!
     Computes the product of vector^T with matrix
     \param V1 input matrix
     \param v2 input vector
     \param w output vector
     \return product of V1 and v2
    */
    void vec_mat_product ( std::vector< std::vector<double> > const&,
                           std::vector<double> const&,
                           std::vector<double> & );
    //! Product of matrix with its transpose V'*V
    /*!
     Computes the product of two matrices
     \param V input matrix
     \param W output matrix
     \return product of V' and V
    */
    void mat_square ( std::vector< std::vector<double> > const&, 
                      std::vector< std::vector<double> > & );
    //! Norm of a vector
    /*!
     Computes the norm of a vector
     \param v input vector
     \return norm of v
    */
    double norm ( std::vector<double> const& );
    //! Frobenius norm of a matrix
    /*!
     Computes the Frobenius norm of a matrix
     \param V input matrix
     \return norm of V
    */
    double norm ( std::vector< std::vector<double> > const& );

    void print_matrix( std::vector< std::vector<double> > const&);

    void print_vector(std::vector<double> const &);
};
//--------------------------------------------------------------------------------

#endif
