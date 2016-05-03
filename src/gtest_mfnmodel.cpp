#include "BasisForMinimumFrobeniusNormModel.hpp"
#include "MinimumFrobeniusNormModel.hpp"
#include "BlackboxData.hpp"

#include <Eigen/Core>
#include <vector>
#include <math.h>
#include "gtest/gtest.h"

//--------------------------------------------------------------------------------
double myfunc_test1 ( std::vector<double> &x ) {
  double result = 1e0;
  for ( int i = 0; i < x.size(); ++i ) {
    result += x.at(i) + x.at(i)*x.at(i);
    for ( int j = i+1; j < x.size(); ++j ) {
       result += x.at(i)*x.at(j);
    }
  }
  return result;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void set_zero( std::vector<double> &v ){
  for (int i = 0; i < v.size(); ++i)
    v[i] = 0e0;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double evaluate_mfnmodel_test1 ( std::vector<double> x, double delta, int nb_nodes ) {

  int dim = x.size();

  BlackboxData data;
  double delta_i = delta;
  BasisForMinimumFrobeniusNormModel basis ( dim, delta_i );
  MinimumFrobeniusNormModel mfn_model ( basis );

  std::vector<double> node ( dim ) ;
  
  data.values.resize( nb_nodes );
  data.noise.resize(1);

  node.resize( dim );
  set_zero( node );
  data.nodes.push_back( node );
  data.values[0].push_back( myfunc_test1( node ) );
  data.surrogate_nodes_index.push_back( 0 );
  data.best_index = 0;

  for ( int i = 0; i < dim; ++i ) {
    set_zero( node );
    node.at(i) += delta;
    data.nodes.push_back( node );
    data.values[0].push_back( myfunc_test1( node ) );
    data.surrogate_nodes_index.push_back( data.nodes.size()-1 );
  }
  if ( nb_nodes > dim+1 ) {
    for ( int i = 0; i < dim; ++i ) {
      set_zero( node );
      node.at(i) -= delta;
      data.nodes.push_back( node );
      data.values[0].push_back( myfunc_test1( node ) );
      data.surrogate_nodes_index.push_back( data.nodes.size()-1 );
    }
  }
  if ( nb_nodes > 2*dim+1 ) {
    for ( int i = 0; i < dim; ++i ) {
      for ( int j = i+1; j < dim; j++ ) {
        set_zero( node );
        node.at(i) += 0.5*delta;
        node.at(j) += 0.5*delta;
        data.nodes.push_back( node );
        data.values[0].push_back( myfunc_test1( node ) );
        data.surrogate_nodes_index.push_back( data.nodes.size()-1 );
      }
    }
  }


  basis.compute_basis_coefficients ( data );

  mfn_model.set_function_values ( data.values[0], data.noise[0], 
                                  data.surrogate_nodes_index );


  return mfn_model.evaluate ( x );

}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( MinimumFrobeniusNormModelTest, evaluation_interpolation_nodes ) {
  std::vector< std::vector<double> > x;
  std::vector<double> point;
  double delta;
  double result;
  int dim;
  double tol = 1e-6;

  Eigen::VectorXd deltas(2);
  Eigen::VectorXi nb_nodes(3);

  deltas << 1.0, 0.5;

  dim = 5;
  nb_nodes(0) = dim+1;
  nb_nodes(1) = 2*dim+1;
  nb_nodes(2) = (dim*dim + 3*dim +2)/2;
  point.resize(5); 

  for ( int d = 0; d < deltas.size(); ++d ) {
    x.clear();
    set_zero(point);
    x.push_back(  point );
    for ( int i = 0; i < dim; ++i ) {
      set_zero( point );
      point.at(i) += deltas(d);
      x.push_back( point );
    }
    for ( int i = 0; i < dim; ++i ) {
      set_zero( point );
      point.at(i) -= deltas(d);
      x.push_back( point );
    }
    for ( int i = 0; i < dim; ++i ) {
      for ( int j = i+1; j < dim; ++j ) {
        set_zero(point);
        point.at(i) += 0.5*deltas(d);
        point.at(j) += 0.5*deltas(d);
        x.push_back( point );
      }
    }
    for ( int n = 0; n < nb_nodes.size(); ++n ) {
      for ( int i = 0; i < nb_nodes(n); ++i ) {
        EXPECT_NEAR ( myfunc_test1( x[i] ), 
                      evaluate_mfnmodel_test1( x[ i ], deltas(d), nb_nodes(n) ),  
                      tol );
      }
    }
  }

}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( MinimumFrobeniusNormModelTest, evaluation_nodes ) {
  std::vector< std::vector<double> > x;
  std::vector<double> point;
  double delta;
  double result;
  int dim;
  double tol = 1e-6;

  Eigen::VectorXd deltas(2);
  Eigen::VectorXi nb_nodes(3);

  deltas << 1.0, 0.5;

  dim = 3;
  nb_nodes(0) = dim+1;
  nb_nodes(1) = 2*dim+1;
  nb_nodes(2) = (dim*dim + 3*dim +2)/2; 
  point.resize(dim);

  for ( int d = 0; d < deltas.size(); ++d ) {

    x.clear();
    set_zero( point );
    x.push_back( point );

    for ( int i = 0; i < dim; ++i ) {
      set_zero( point );
      point.at(i) += deltas(d) * 0.25;
      x.push_back( point );
    }
    for ( int i = 0; i < dim; ++i ) {
      set_zero( point );
      point.at(i) -= deltas(d) * 0.5;
      point.at(dim-1) -= deltas(d) * 0.25;
      x.push_back( point );
    }
    for ( int i = 0; i < dim; ++i ) {
      for ( int j = i+1; j < dim; ++j ) {
        set_zero( point );
        point.at(i) += 0.25*deltas(d);
        point.at(j) += 0.25*deltas(d);
        x.push_back( point );
      }
    }

    for ( int n = 0; n < nb_nodes.size(); ++n ) {
      for ( int i = 0; i < nb_nodes(n); ++i ) {
        result = myfunc_test1( x[i] );
        if ( n == 0 && i == 1 && d == 0) result = 1.5;
        if ( n == 0 && i == 2 && d == 0) result = 1.5;
        if ( n == 0 && i == 3 && d == 0) result = 1.5;      
        if ( n == 1 && i == 4 && d == 0) result = 0.5625;
        if ( n == 1 && i == 5 && d == 0) result = 0.5625;
        if ( n == 1 && i == 4 && d == 1) result = 0.703125;
        if ( n == 1 && i == 5 && d == 1) result = 0.703125;
        if ( n == 0 && i >= 1 && d == 1) result = 1.1875;
        EXPECT_NEAR ( result,
                      evaluate_mfnmodel_test1( x[ i ], deltas(d), nb_nodes(n) ),  
                      tol );
      }
    }
  }

}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( MinimumFrobeniusNormModelTest, evaluation_interpolation_coinciding_nodes ) {
  std::vector< std::vector<double> > x;
  std::vector<double> point;
  double delta;
  double result;
  int dim;
  double tol = 1e-6;

  Eigen::VectorXd deltas(2);
  Eigen::VectorXi nb_nodes(3);

  deltas << 1.0, 0.5;

  dim = 5;
  nb_nodes(0) = dim+1;
  nb_nodes(1) = 2*dim+1;
  nb_nodes(2) = (dim*dim + 3*dim +2)/2;
  point.resize(5); 

  for ( int d = 0; d < deltas.size(); ++d ) {
    x.clear();
    set_zero(point);
    x.push_back(  point );
    for ( int i = 0; i < dim; ++i ) {
      set_zero( point );
      point.at(i) += deltas(d);
      x.push_back( point );
    }
    for ( int i = 0; i < dim; ++i ) {
      set_zero( point );
      point.at(i) += deltas(d);
      x.push_back( point );
    }
    for ( int i = 0; i < dim; ++i ) {
      for ( int j = i+1; j < dim; ++j ) {
        set_zero(point);
        point.at(i) += 0.5*deltas(d);
        point.at(j) += 0.5*deltas(d);
        x.push_back( point );
      }
    }
    for ( int n = 0; n < nb_nodes.size(); ++n ) {
      for ( int i = 0; i < nb_nodes(n); ++i ) {
        EXPECT_NEAR ( myfunc_test1( x[i] ), 
                      evaluate_mfnmodel_test1( x[ i ], deltas(d), nb_nodes(n) ),  
                      tol );
      }
    }
  }

}
//--------------------------------------------------------------------------------
