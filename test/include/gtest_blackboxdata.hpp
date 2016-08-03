#include "BlackBoxData.hpp"
#include "VectorOperations.hpp"
#include "gtest/gtest.h"
#include <vector>
#include <iostream>

//--------------------------------------------------------------------------------
class Wrapper_BlackBoxData : public VectorOperations 
{
  public:
    int blackboxdata_test1 ( )
    {

      BlackBoxData data;

      int nb_functions = 2;
      int dim          = 2;

      data.initialize( nb_functions, dim );

      std::vector<double> x( dim );
      std::vector<double> v( nb_functions );
      std::vector<double> r( nb_functions );

      data.best_index = 0;

      x.at(0) = 1e0; x.at(1) = 1e0;
      v.at(0) = 0e0; v.at(1) = 0e0;
      r.at(0) = 100e0; r.at(1) = -100e0;
      data.nodes.push_back( x ); 
      data.values[0].push_back( v[0] );
      data.values[1].push_back( v[1] );
      data.noise[0].push_back( r[0] );
      data.noise[1].push_back( r[1] );
      data.active_index.push_back( 0 );

      x.at(0) = 0e0; x.at(1) = 5e-1;
      v.at(0) = 1e0; v.at(1) = -1e0;
      r.at(0) = 90e0; r.at(1) = -90e0;
      data.nodes.push_back( x ); 
      data.values[0].push_back( v[0] );
      data.values[1].push_back( v[1] );
      data.noise[0].push_back( r[0] );
      data.noise[1].push_back( r[1] );
      data.active_index.push_back( 1 );

      x.at(0) = 2e0; x.at(1) = 1e0;
      v.at(0) = 2e0; v.at(1) = -2e0;
      r.at(0) = 80e0; r.at(1) = -80e0;
      data.nodes.push_back( x ); 
      data.values[0].push_back( v[0] );
      data.values[1].push_back( v[1] );
      data.noise[0].push_back( r[0] );
      data.noise[1].push_back( r[1] );
      data.active_index.push_back( 2 );

      x.at(0) = 1.5e0; x.at(1) = 1.5e0;
      v.at(0) = 3e0; v.at(1) = -3e0;
      r.at(0) = 70e0; r.at(1) = -70e0;
      data.nodes.push_back( x ); 
      data.values[0].push_back( v[0] );
      data.values[1].push_back( v[1] );
      data.noise[0].push_back( r[0] );
      data.noise[1].push_back( r[1] );
      data.active_index.push_back( 3 );

      x.at(0) = 0e0; x.at(1) = 0e0;
      v.at(0) = 4e0; v.at(1) = -4e0;
      r.at(0) = 60e0; r.at(1) = -60e0;
      data.nodes.push_back( x ); 
      data.values[0].push_back( v[0] );
      data.values[1].push_back( v[1] );
      data.noise[0].push_back( r[0] );
      data.noise[1].push_back( r[1] );

      x.at(0) = -.3e0; x.at(1) = -.3e0;
      v.at(0) = 5e0; v.at(1) = -5e0;
      r.at(0) = 50e0; r.at(1) = -50e0;
      data.nodes.push_back( x ); 
      data.values[0].push_back( v[0] );
      data.values[1].push_back( v[1] );
      data.noise[0].push_back( r[0] );
      data.noise[1].push_back( r[1] );
      data.active_index.push_back( 5 );

      x.at(0) = 1e0; x.at(1) = 1e0;

      std::vector< std::vector<double> > active_nodes;
      std::vector<double> active_values;
      std::vector<double> active_noise;

      double scaling = 1e0;
      active_nodes = data.get_scaled_active_nodes ( scaling );

      x.at(0) = 0e0; x.at(1) = 0e0;
      if ( diff_norm( active_nodes.at(0), x) > 1e-6 ) return -1;
      x.at(0) = -1e0; x.at(1) = -5e-1;
      if ( diff_norm( active_nodes.at(1), x) > 1e-6 ) return -2;
      x.at(0) = 1e0; x.at(1) = 0e0;
      if ( diff_norm( active_nodes.at(2), x) > 1e-6 ) return -3;
      x.at(0) = 5e-1; x.at(1) = 5e-1;
      if ( diff_norm( active_nodes.at(3), x) > 1e-6 ) return -4;
      x.at(0) = -1.3e0; x.at(1) = -1.3e0;
      if ( diff_norm( active_nodes.at(4), x) > 1e-6 ) return -5;


      active_values = data.get_active_values( 0 );
      for ( unsigned i = 0; i < data.active_index.size(); ++i )
        if ( fabs( active_values.at( i ) - ((double)data.active_index[i]) ) > 1e-6 ) return -100;
      active_values = data.get_active_values( 1 );
      for ( unsigned i = 0; i < data.active_index.size(); ++i )
        if ( fabs( active_values.at( i ) + ((double)data.active_index[i]) ) > 1e-6 ) return -200;
      active_noise = data.get_active_noise( 0 );
      for ( unsigned i = 0; i < data.active_index.size(); ++i )
        if ( fabs( active_noise.at( i ) - 100e0 + 1e1*((double)data.active_index[i]) ) > 1e-6 ) return -300;
      active_noise = data.get_active_noise( 1 );
      for ( unsigned i = 0; i < data.active_index.size(); ++i )
        if ( fabs( active_noise.at( i ) + 100e0 - 1e1*((double)data.active_index[i]) ) > 1e-6 ) return -400;


      data.active_index.erase(data.active_index.begin()+1);
      data.active_index.push_back( 4 );      

      x.at(0) = 1e0; x.at(1) = 1e0;
      scaling = 1e0;
      active_nodes = data.get_scaled_active_nodes ( scaling );
      x.at(0) = 0e0; x.at(1) = 0e0;
      if ( diff_norm( active_nodes.at(0), x) > 1e-6 ) return -6;
      x.at(0) = 1e0; x.at(1) = 0e0;
      if ( diff_norm( active_nodes.at(1), x) > 1e-6 ) return -7;
      x.at(0) = 5e-1; x.at(1) = 5e-1;
      if ( diff_norm( active_nodes.at(2), x) > 1e-6 ) return -8;
      x.at(0) = -1.3e0; x.at(1) = -1.3e0;
      if ( diff_norm( active_nodes.at(3), x) > 1e-6 ) return -9;
      x.at(0) = -1e0; x.at(1) = -1e0;
      if ( diff_norm( active_nodes.at(4), x) > 1e-6 ) return -10;

      x.at(0) = 1e0; x.at(1) = 1e0;
      scaling = 2e0;
      active_nodes = data.get_scaled_active_nodes ( scaling );
      x.at(0) = 0e0; x.at(1) = 0e0;
      if ( diff_norm( active_nodes.at(0), x) > 1e-6 ) return -11;
      x.at(0) = 5e-1; x.at(1) = 0e0;
      if ( diff_norm( active_nodes.at(1), x) > 1e-6 ) return -12;
      x.at(0) = .25e0; x.at(1) = .25e0;
      if ( diff_norm( active_nodes.at(2), x) > 1e-6 ) return -13;
      x.at(0) = -.65e0; x.at(1) = -.65e0;
      if ( diff_norm( active_nodes.at(3), x) > 1e-6 ) return -14;
      x.at(0) = -5e-1; x.at(1) = -5e-1;
      if ( diff_norm( active_nodes.at(4), x) > 1e-6 ) return -15;

      active_values = data.get_active_values( 0 );
      for ( unsigned i = 0; i < data.active_index.size(); ++i )
        if ( fabs( active_values.at( i ) - ((double)data.active_index.at(i)) ) > 1e-6 ) return -1000;
      active_values = data.get_active_values( 1 );
      for ( unsigned i = 0; i < data.active_index.size(); ++i )
        if ( fabs( active_values.at( i ) + ((double)data.active_index.at(i)) ) > 1e-6 ) return -2000;
      active_noise = data.get_active_noise( 0 );
      for ( unsigned i = 0; i < data.active_index.size(); ++i )
        if ( fabs( active_noise.at( i ) - 100e0 + 1e1*((double)data.active_index.at(i)) ) > 1e-6 ) return -3000;
      active_noise = data.get_active_noise( 1 );
      for ( unsigned i = 0; i < data.active_index.size(); ++i )
        if ( fabs( active_noise.at( i ) + 100e0 - 1e1*((double)data.active_index.at(i)) ) > 1e-6 ) return -4000;


      return 1;
    }
};
//--------------------------------------------------------------------------------




//--------------------------------------------------------------------------------
TEST ( BlackBoxDataTest, blackboxdata_test1 ) 
{
  Wrapper_BlackBoxData W;
  EXPECT_EQ( 1, W.blackboxdata_test1() );
}
//--------------------------------------------------------------------------------
