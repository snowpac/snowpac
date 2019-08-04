#ifndef HBlackBoxBaseClass
#define HBlackBoxBaseClass

#include <vector>

//! Black box base class
class BlackBoxBaseClass {
  public:
    //! Prototype for black box function to be provided by the user
    /*!
     Prototype for black box function to be provided by the user. 
     Required interface for deterministic optimization (stochastic_optimization = false).
     \param x design point at which the black box has to be evaluated
     \param vals array of values of black box evaluations
      - vals[0] = objective evaluation
      - vals[1 ... ] = constraint evaluations
     \param param user data
    */ 
    virtual void evaluate ( std::vector<double> const &x, std::vector<double> &vals,
                            void *param) { return; }

    //! Prototype for black box function to be provided by the user, which expects an asynonchronous evaluation of the blackbox
    /*!
     Prototype for black box function to be provided by the user. 
     Required interface for deterministic optimization (stochastic_optimization = false).
     \param x design point at which the black box has to be evaluated
     \param vals array of values of black box evaluations
      - vals[0] = objective evaluation
      - vals[1 ... ] = constraint evaluations
     \param param user data
    */ 
    virtual void evaluate_nowait ( std::vector<double> const &x, std::vector<double> &vals,
                            void *param) { return; }

    //! Prototype for black box function to be provided by the user
    /*!
     Prototype for black box function to be provided by the user.
     Required interface for stochastic optimization (stochastic_optimization = true).
     \param x design point at which the black box has to be evaluated
     \param vals array of values of black box evaluations
      - vals[0] = objective evaluation
      - vals[1 ... ] = constraint evaluations
     \param noise array of noise estimate in black box evaluations
      - noise[0] = estimate of magnitude of noise in objective evaluation
      - noise[1 ... ] = estimate of magnitude of noise in constraint evaluations  
     \param param user data
    */ 
    virtual void evaluate ( std::vector<double> const &x, std::vector<double> &vals,
                            std::vector<double> &noise, void *param ) { return; }

    //! Prototype for black box function to be provided by the user, which expects an asynonchronous evaluation of the blackbox
    /*!
     Prototype for black box function to be provided by the user.
     Required interface for stochastic optimization (stochastic_optimization = true).
     \param x design point at which the black box has to be evaluated
     \param vals array of values of black box evaluations
      - vals[0] = objective evaluation
      - vals[1 ... ] = constraint evaluations
     \param noise array of noise estimate in black box evaluations
      - noise[0] = estimate of magnitude of noise in objective evaluation
      - noise[1 ... ] = estimate of magnitude of noise in constraint evaluations  
     \param param user data
    */ 
    virtual void evaluate_nowait ( std::vector<double> const &x, std::vector<double> &vals,
                            std::vector<double> &noise, void *param ) { return; }

    //! Prototype for black box function to be provided by the user which uses analytic smoothing approach.
    //! That means we need all the samples as well.
    /*!
     Prototype for black box function to be provided by the user.
     Required interface for stochastic optimization (stochastic_optimization = true).
     \param x design point at which the black box has to be evaluated
     \param vals array of values of black box evaluations
      - vals[0] = objective evaluation
      - vals[1 ... ] = constraint evaluations
     \param noise array of noise estimate in black box evaluations
      - noise[0] = estimate of magnitude of noise in objective evaluation
      - noise[1 ... ] = estimate of magnitude of noise in constraint evaluations
     \param param user data
    */
    virtual void evaluate ( std::vector<double> const &x, std::vector<double> &vals,
                            std::vector<double> &noise, std::vector<std::vector<double>> &vals_samples, void *param ) { return; }

    //! TODO
    //!
    virtual double evaluate_samples ( std::vector<double> const &samples, const unsigned int index, std::vector<double>const& x ) { return -1; }

    //! Prototype for synchronize method provided by the user, which synchronizes after a number of asynchronous evaluate_nowait() calls
    /*!
     Prototype for black box function to be provided by the user.
     Required interface for stochastic optimization (stochastic_optimization = true).

     TODO: Think about parameters
    */ 
    virtual void synchronize () { return; }

};

#endif
