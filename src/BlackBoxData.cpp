#include "BlackBoxData.hpp"
#include <sys/stat.h>

//--------------------------------------------------------------------------------
void BlackBoxData::initialize ( int n, int dim ) { 
  scaling = 1e0;
  scaled_node.resize( dim );
  for ( int i = 0; i < dim; ++i )
    scaled_node[i] = 0e0;
  values.resize( n );
  values_MC.resize( n );
  noise.resize( n ); 
  noise_MC.resize( n );
  samples.resize( n );
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void BlackBoxData::delete_history () {
  int nb_active_nodes = active_index.size();
  int found_point;
  int nb_nodes = nodes.size();
  int n = values.size();
  
  for ( int j = 0; j < nb_active_nodes; ++j ) {  
    if ( best_index == active_index.at(j) ) {
      best_index = j;
      break;
    }
  }

  for ( int i = nb_nodes-1; i >=0; --i ) {
    found_point = -1;
    for ( int j = 0; j < nb_active_nodes; ++j ) {  
      if ( i == active_index.at(j) ) {
        found_point = j;
        break;
      }
    }
    if ( found_point >= 0 ) {
      active_index.erase( active_index.begin() + found_point );
      nb_active_nodes--;
    } else {
      nodes.erase ( nodes.begin() + i );
      for ( int j = 0; j < n; ++j ) {
        values[j].erase( values[j].begin() + i );
        if ( noise[0].size() > 0 )
          noise[j].erase ( noise[j].begin() + i );
        if ( noise_MC[0].size() > 0 )   
          noise_MC[j].erase ( noise_MC[j].begin() + i );        
      }
    }
  } 
  nb_nodes = nodes.size();

  assert( active_index.size() == 0 );
  for ( int i = 0; i < nb_nodes; ++i ) 
    active_index.push_back( i );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
int BlackBoxData::write_to_file ( const char *filename) {
  int EXIT_FLAG = 1; //success
  struct stat buffer;   
  // check if file exists
  if ( stat (filename, &buffer) == 0 ) {
    // do not overwrite file if it exists 
    std::cout << "Error   : Output file already exists." << std::endl;    
    EXIT_FLAG = -1; // fail
    return EXIT_FLAG;
  }
  std::fstream output_file;
  output_file.open( filename, std::fstream::out );
  output_file.precision(16);
  output_file << scaling << " ; ";
  output_file << best_index << " ; ";
  output_file << max_nb_nodes << " ; ";
  output_file << nodes.size() << " ; ";
  output_file << nodes.at(0).size() << " ; ";
  output_file << values.size() << " ; ";
  output_file << values[0].size() << " ; ";
  output_file << active_index.size() << std::endl;

  for ( unsigned int i = 0; i < active_index.size()-1; ++i )
    output_file << active_index[i] << " ; ";
  output_file <<  active_index.back() << std::endl;  
  for ( unsigned int i = 0; i < nodes.size(); ++i ) {
    for ( unsigned int j = 0; j < nodes[i].size()-1; ++j )
      output_file << nodes[i][j] << " ; ";
    output_file <<  nodes[i].back() << std::endl;  
  }
  for ( unsigned int i = 0; i < values.size(); ++i ) {
    for ( unsigned int j = 0; j < values[i].size()-1; ++j )
       output_file << values[i][j] << " ; ";
    output_file << values[i].back() << std::endl;  
  }
  for ( unsigned int i = 0; i < noise.size(); ++i ) {
    for ( unsigned int j = 0; j < noise[i].size()-1; ++j )
       output_file << noise[i][j] << " ; ";
    output_file << noise[i].back() << std::endl;  
  }

  return EXIT_FLAG;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
int BlackBoxData::read_from_file ( const char *filename) {
  int EXIT_FLAG = 1; //success
  struct stat buffer;   
  // check if file exists
  if ( stat (filename, &buffer) != 0 ) {
    // do not overwrite file if it exists 
    std::cout << "Error   : Input file does not exist." << std::endl;    
    EXIT_FLAG = -1; // fail
    return EXIT_FLAG;
  }

  std::ifstream file( filename );
  std::string line_in_file; 
  std::string delimiter = " ; ";
  std::string str_value;
  int end_position;
  std::getline(file, line_in_file);

  // read scaling
  end_position = line_in_file.find(delimiter);
  str_value = line_in_file.substr( 0, end_position );
  line_in_file.erase(0, end_position + delimiter.length());
  int scaling_loc = std::stod( str_value );

  // best node
  end_position = line_in_file.find(delimiter);
  str_value = line_in_file.substr( 0, end_position );
  line_in_file.erase(0, end_position + delimiter.length());
  best_index = std::stoi( str_value );

  // max number of active nodes  
  end_position = line_in_file.find(delimiter);
  str_value = line_in_file.substr( 0, end_position );
  line_in_file.erase(0, end_position + delimiter.length());
  max_nb_nodes = std::stoi( str_value );

  // read number of nodes  
  end_position = line_in_file.find(delimiter);
  str_value = line_in_file.substr( 0, end_position );
  line_in_file.erase(0, end_position + delimiter.length());
  int nb_nodes = std::stoi( str_value );

  // read dimension 
  end_position = line_in_file.find(delimiter);
  str_value = line_in_file.substr( 0, end_position );
  line_in_file.erase(0, end_position + delimiter.length());
  int dim = std::stoi( str_value );

  // read nb_values
  end_position = line_in_file.find(delimiter);
  str_value = line_in_file.substr( 0, end_position );
  line_in_file.erase(0, end_position + delimiter.length());
  int nb_models = std::stoi( str_value );

  // read nb_noise
  end_position = line_in_file.find(delimiter);
  str_value = line_in_file.substr( 0, end_position );
  line_in_file.erase(0, end_position + delimiter.length());
  int nb_values = std::stoi( str_value );

  // read nb_active_nodes
  end_position = line_in_file.find(delimiter);
  str_value = line_in_file.substr( 0, end_position );
  line_in_file.erase(0, end_position + delimiter.length());
  int nb_active_nodes = std::stoi( str_value );

  initialize( nb_models, dim );
  scaling = scaling_loc;

  ///
  int line_counter = 0;
  double tmp_dbl;
  std::vector<double> node( dim );

  // read active_index
  std::getline(file, line_in_file);
  while ( ( end_position = line_in_file.find(delimiter) ) != std::string::npos ) { 
    str_value = line_in_file.substr( 0, end_position );
    line_in_file.erase(0, end_position + delimiter.length());
    active_index.push_back( std::stoi( str_value ) ); 
  }
  active_index.push_back( std::stoi( line_in_file ) ); 

  while ( std::getline(file, line_in_file) )
  {
    if ( line_counter < nb_nodes ) {
      for ( int i = 0; i < dim; ++i ) {
        end_position = line_in_file.find(delimiter);
        str_value = line_in_file.substr( 0, end_position );
        line_in_file.erase(0, end_position + delimiter.length());
        node[i] = std::stod( str_value );        
      }
      nodes.push_back( node );
    }
    if ( line_counter >= nb_nodes && line_counter < nb_nodes + nb_models ) {
      while ( ( end_position = line_in_file.find(delimiter) ) != std::string::npos ) { 
        str_value = line_in_file.substr( 0, end_position );
        line_in_file.erase(0, end_position + delimiter.length());
        values[line_counter-nb_nodes].push_back( std::stod( str_value ) ); 
      }
      values[line_counter-nb_nodes].push_back( std::stod( line_in_file ) ); 
    }
    if ( line_counter >= nb_nodes+nb_models && line_counter < nb_nodes+2*nb_models) {
      while ( ( end_position = line_in_file.find(delimiter) ) != std::string::npos ) { 
        str_value = line_in_file.substr( 0, end_position );
        line_in_file.erase(0, end_position + delimiter.length());
        noise[line_counter-nb_nodes-nb_models].push_back( std::stod( str_value ) ); 
      }
      noise[line_counter-nb_nodes-nb_models].push_back( std::stod( line_in_file ) ); 
    }
    line_counter++;
  }


  return EXIT_FLAG;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &BlackBoxData::transform( std::vector<double> const& x ) {
  rescale( 1e0 / scaling, x, nodes[ best_index ], scaled_node);
  return scaled_node;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector< std::vector<double> > &BlackBoxData::get_scaled_active_nodes ( 
  double scaling_input) {
  scaling = scaling_input;
  int scaled_active_nodes_size = scaled_active_nodes.size();
  int active_index_size = active_index.size();
  if ( scaled_active_nodes_size > active_index_size ) {
    scaled_active_nodes.resize( active_index_size );
    scaled_active_nodes_size = active_index_size;
  }
  for ( int i = 0; i < scaled_active_nodes_size; ++i ) {
    rescale( 1e0 / scaling,  nodes[ active_index[i] ], 
             nodes[ best_index], scaled_active_nodes[i]);  
  }
  for ( int i = scaled_active_nodes_size; i < active_index_size; ++i ) {
    rescale( 1e0 / scaling,  nodes[ active_index[i] ], 
             nodes[ best_index ], scaled_node);
    scaled_active_nodes.push_back( scaled_node );
  }
  return scaled_active_nodes;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &BlackBoxData::get_active_values( int i ) {
  int active_values_size = active_values.size();
  int active_index_size = active_index.size();
  if ( active_values_size > active_index_size ) {
    active_values.resize( active_index_size );
    active_values_size = active_index_size;
  }
  for ( int j = 0; j < active_values_size; ++j ) {
    active_values.at(j) = values.at( i ).at( active_index.at(j) );
  }
  for ( int j = active_values_size; j < active_index_size; ++j ) {
    active_values.push_back( values.at( i ).at(  active_index.at(j) ) );
  }
  return active_values;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &BlackBoxData::get_active_noise( int i ) {
  int active_noise_size = active_noise.size();
  int active_index_size = active_index.size();
  if ( active_noise_size > active_index_size ) { 
    active_noise.resize( active_index_size );
    active_noise_size = active_index_size;
  }
  for (int j = 0; j < active_noise_size; ++j )
    active_noise[j] = noise[i][ active_index[j] ];
  for (int j = active_noise_size; j < active_index_size; ++j )
    active_noise.push_back( noise[ i ][ active_index[j] ] );
  return active_noise;
}
//--------------------------------------------------------------------------------



