#include "RegularizedMinimumFrobeniusNormModel.hpp"
#include "BasisForMinimumFrobeniusNormModel.hpp"
#include "BlackboxData.hpp"

#include <Eigen/Core>
#include <vector>
#include <random>
#include <fstream>

std::random_device rd;
int randSeed = rd();
std::mt19937 gen (randSeed);  
std::uniform_real_distribution<double> dis(-1,1);

double myfunc ( Eigen::VectorXd const &x, double &noise) {

  int nbs = 1000;
  double sample, noise_loc;
  noise = 0e0;
    
  for ( int i = 0; i < nbs; i++ ) {
    sample = dis(gen);
    noise += sample*sample;
    noise_loc += sample;
  }
  noise = sqrt( noise / ((double) nbs-1));

  return pow(x(0)-2e0,2e0)+pow(x(1)-1e0,2e0) +  noise_loc / ((double)nbs);
}


int main () {

  double delta = sqrt(0.5);
  int dim = 2;
  int nb_nodes = 5;

  BasisForMinimumFrobeniusNormModel basis( dim, delta );

  RegularizedMinimumFrobeniusNormModel rmfn ( basis );

  BlackboxData bb;
  bb.values.push_back ( Eigen::VectorXd::Zero( nb_nodes ) );
  bb.noise.push_back ( Eigen::VectorXd::Zero( nb_nodes ) );

  Eigen::VectorXd point ( dim );
  double fval;
  double noise;

  
  Eigen::VectorXd x( dim );
  x << 0.5, 0.5;

  
  point = x;
  fval = myfunc ( point, noise );  
  bb.nodes.push_back( point );
  bb.values[0](0) = fval;
  bb.noise[0](0) = noise;
  bb.surrogate_nodes_index.push_back( 0 );

  for (int i = 0; i < dim; i++) {
    point = x;
    point(i) += delta;
    fval = myfunc ( point, noise );  
    bb.nodes.push_back( point );
    bb.values[0](i+1) = fval;
    bb.noise[0](i+1) = noise;
    bb.surrogate_nodes_index.push_back( i+1 );
  }
  for (int i = 0; i < dim; i++) {
    point = x;
    point(i) -= delta;
    fval = myfunc ( point, noise );  
    bb.nodes.push_back( point );
    bb.values[0](i+1+dim) = fval;
    bb.noise[0](i+1+dim) = noise;
    bb.surrogate_nodes_index.push_back( dim+i+1 );
  }




  for ( int i = 0; i < nb_nodes; i++ )


  std::cout << "x0" << std::endl;

  basis.compute_basis_coefficients ( bb );

  std::cout << "x0" << std::endl;

  rmfn.set_function_values ( bb.values[0], bb.noise[0], bb.surrogate_nodes_index );  

  std::cout << "x0" << std::endl;

  Eigen::VectorXd x_loc(2);
  Eigen::VectorXd fvals(3);
  std::ofstream outputfile ( "surrogate_data_o.dat" );
  if ( outputfile.is_open( ) ) {
    for (double i = -1.0; i <= 2.0; i+=0.01) {
      x_loc(0) = i;
      for (double j = -1.0; j < 2.0; j+=0.01) {
        x_loc(1) = j;
        fvals(0) = rmfn.evaluate( x_loc );
        outputfile << x_loc(0) << "; " << x_loc(1) << "; " << fvals(0) << std::endl;
      }
    }
    outputfile.close( );
  } else std::cout << "Unable to open file." << std::endl;


  return 1;
}
