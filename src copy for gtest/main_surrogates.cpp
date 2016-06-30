#include "BasisForMinimumFrobeniusNormModel.hpp"
#include "MinimumFrobeniusNormModel.hpp"
#include "BlackboxData.hpp"
#include <fstream>
#include <iostream>


double myfunc ( std::vector<double> &x ) {
  double result = 1e0;
  for ( int i = 0; i < x.size(); ++i ) {
    result += x.at(i) + x.at(i)*x.at(i);
    for ( int j = i+1; j < x.size(); ++j ) {
       result += x.at(i)*x.at(j);
    }
  }
  return result;
}

void set_zero( std::vector<double> &v ){
  for (int i = 0; i < v.size(); ++i)
    v[i] = 0e0;
  return;
} 

int main ( int argv, char** argc ) 
{
  double delta = 1;
  int dim = 2;
  //int nb_nodes = (dim*dim + 3*dim+2)/2;
  int nb_nodes = 2*dim+1;
  BlackboxData data;
  data.values.resize( nb_nodes );
  data.noise.resize(1);
  BasisForMinimumFrobeniusNormModel basis ( dim, delta );
  MinimumFrobeniusNormModel mfn_model ( basis );

  std::vector<double> node (dim);

  node.resize( dim );
  set_zero( node );
  data.nodes.push_back( node );
  data.values[0].push_back( myfunc( node ) );
  data.surrogate_nodes_index.push_back( 0 );
  data.best_index = 0;

  for ( int i = 0; i < dim; ++i ) {
    set_zero( node );
    node.at(i) += delta;
    data.nodes.push_back( node );
    data.values[0].push_back( myfunc( node ) );
    data.surrogate_nodes_index.push_back( data.nodes.size()-1 );
  }
  if ( nb_nodes > dim+1 ) {
    for ( int i = 0; i < dim; ++i ) {
      set_zero( node );
      node.at(i) -= delta;
      data.nodes.push_back( node );
      data.values[0].push_back( myfunc( node ) );
      data.surrogate_nodes_index.push_back( data.nodes.size()-1 );
    }
  }
  if ( nb_nodes > 2*dim+1 ) {
    for ( int i = 0; i < dim; ++i ) {
      for ( int j = i+1; j < dim; ++j ) {
        set_zero( node );
        node.at(i) += delta;
        node.at(j) += delta;
        data.nodes.push_back( node );
        data.values[0].push_back( myfunc( node ) );
        data.surrogate_nodes_index.push_back( data.nodes.size()-1 );
      }
    }
  }
  basis.compute_basis_coefficients ( data );

  mfn_model.set_function_values ( data.values[0], data.noise[0], 
                                  data.surrogate_nodes_index );

  std::vector<double> x_loc(2);
  std::vector<double> fvals(1);
  std::ofstream outputfile ( "surrogate_data_o.dat" );
  if ( outputfile.is_open( ) ) {
    for (double i = -1.0; i <= 2.0; i+=0.1) {
      x_loc.at(0) = i;
      for (double j = -1.0; j < 2.0; j+=0.1) {
        x_loc.at(1) = j;
        fvals.at(0) = mfn_model.evaluate( x_loc );
        outputfile << x_loc.at(0) << "; " << x_loc.at(1) << "; " << fvals.at(0) << std::endl;
      }
    }
    outputfile.close( );
  } else std::cout << "Unable to open file." << std::endl;



  return 1;
}
