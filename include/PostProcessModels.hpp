#ifndef HPostProcessModels
#define HPostProcessModels

#include "BlackBoxData.hpp"
#include "MinimumFrobeniusNormModel.hpp"
#include "LegendreBasisForMinimumFrobeniusNormModel.hpp"
#include "VectorOperations.hpp"

template<class TSurrogateModel = MinimumFrobeniusNormModel,
         class TBasisForSurrogateModel = LegendreBasisForMinimumFrobeniusNormModel>
class PostProcessModels : VectorOperations {
  private:
    int nb_models;
    int dim;
    double delta;
    std::unique_ptr<TSurrogateModel> surrogate_model_prototype;
    TBasisForSurrogateModel surrogate_basis;
    std::vector<TSurrogateModel> surrogate_models;
    std::shared_ptr<BlackBoxData> evaluations;
    std::vector<double> vec_output;
    std::vector<double> c_vec;
    std::vector< std::vector<double> > g_vec;
    std::vector< std::vector< std::vector<double> > > H_vec;
  public:
    PostProcessModels ( BlackBoxData &evaluations_input ) :
      surrogate_basis ( evaluations_input.nodes[0].size() )
    {
      surrogate_model_prototype.reset( new TSurrogateModel(surrogate_basis) );
      evaluations = std::make_shared<BlackBoxData>(evaluations_input);
      nb_models = evaluations->values.size();
      dim = evaluations->nodes[0].size();
      delta = evaluations->get_scaling();
      vec_output.resize(dim);
      c_vec.resize( nb_models );
      g_vec.resize( nb_models );
      H_vec.resize( nb_models );
 
      surrogate_basis.compute_basis_coefficients ( evaluations->get_scaled_active_nodes( delta ) ) ;

      for (int i = 0; i < nb_models; ++i ) {
          surrogate_models.push_back ( *surrogate_model_prototype );
          surrogate_models[ i ].set_function_values ( evaluations->get_active_values(i) );
      }

      for ( int i = 0; i < nb_models; ++i ) {
        g_vec[i].resize( dim );
        H_vec[i].resize( dim );
        for ( int j = 0; j < dim; ++j ) {
          H_vec[i][j].resize( dim );
        }
        get_c_g_H( i, c_vec[i], g_vec[i], H_vec[i]);
      }
        
      return;
    }

//--------------------------------------------------------------------------------
    void get_c_g_H ( int model_number, double &c, std::vector<double> &g,
                     std::vector< std::vector<double> > &H )
    {

      double evaluations_scaling = delta;
      std::vector<double> evaluations_center_node = evaluations->nodes[ evaluations->best_index ];
      for (int i = 0; i < dim; ++i) g[i] = 0.0;
      c = surrogate_models[model_number].evaluate( g );
      c -= VectorOperations::dot_product( surrogate_models[model_number].gradient(), 
                              evaluations_center_node) / evaluations_scaling;
      this->mat_vec_product( surrogate_models[model_number].hessian(), 
                             evaluations_center_node, g ); 
      c += 0.5*VectorOperations::dot_product( evaluations_center_node, g ) / 
           ( evaluations_scaling * evaluations_scaling );

      this->scale ( -1.0/(evaluations_scaling * evaluations_scaling), g, g);
      this->add( 1.0/evaluations_scaling, surrogate_models[model_number].gradient(), g);  

      H = surrogate_models[model_number].hessian();
      for ( int i = 0; i < dim; ++i ) {
        for ( int j = 0; j < dim; ++j ) {
          H[i][j] /= (evaluations_scaling * evaluations_scaling);
        }
      }

      return; 
    }
//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------
    double get_trustregion ( )
    {
      return evaluations->get_scaling(); 
    }
//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------
    double get_c ( int model_number, std::vector<double> &x )
    {
      assert( x.size() == (unsigned) dim );
      double c = c_vec[ model_number ];
      c += VectorOperations::dot_product( g_vec[ model_number], x );
      this->mat_vec_product( H_vec[model_number], x, vec_output );
      c += 0.5*VectorOperations::dot_product( vec_output, x );
      return c; 
    }
//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------
    std::vector<double> get_g ( int model_number, std::vector<double> &x )
    {
      this->mat_vec_product( H_vec[model_number], x, vec_output);
      this->add( g_vec[model_number], vec_output );
      return vec_output;
    }
//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------
    std::vector< std::vector<double> > get_H ( int model_number )
    {
      return H_vec[model_number]; 
    }
//--------------------------------------------------------------------------------

};

#endif
