#ifndef HVectorOperations
#define HVectorOperations

#include <vector>

//--------------------------------------------------------------------------------
//! Vector operations
class VectorOperations {
  private:
    double dbl;
    int size;
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
    //! Adding scaled vector
    /*!
     Computes w = w + s v
     \param s scaling factor
     \param v input vector 
     \param w input and output vector
    */
    void add( double, std::vector<double> const&,
              std::vector<double> &);
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
    //! Norm of a vector
    /*!
     Computes the norm of a vector
     \param v input vector
     \return norm of v
    */
    double norm ( std::vector<double> const& );
};
//--------------------------------------------------------------------------------

#endif
