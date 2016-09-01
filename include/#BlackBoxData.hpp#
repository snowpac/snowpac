#ifndef HBlackBoxData
#define HBlackBoxData

#include "VectorOperations.hpp"
#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>

class BlackBoxData : public VectorOperations{
  private:
    std::vector< std::vector<double> > scaled_active_nodes;
    std::vector<double> active_values;
    std::vector<double> active_noise;  
    std::vector<double> scaled_node;
    double scaling;

  public:
    BlackBoxData ( int n, int dim ) { initialize (n, dim); } 
    BlackBoxData ( const char *filename ) { read_from_file(filename); }
    BlackBoxData ( ) { }
    int max_nb_nodes;
    int best_index;
    std::vector< std::vector<double> > nodes;
    std::vector< std::vector<double> > values;
    std::vector< std::vector<double> > noise;
    std::vector<int> active_index;

    void initialize( int, int );
    void delete_history();
    int write_to_file( const char* );
    int read_from_file( const char* );
    double get_scaling () { return scaling; } 
    std::vector<double> &transform( std::vector<double> const& );
    std::vector< std::vector<double> > &get_scaled_active_nodes( double );
    std::vector<double> &get_active_values( int );
    std::vector<double> &get_active_noise( int );

};


#endif

