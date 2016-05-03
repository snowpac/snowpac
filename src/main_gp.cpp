#include "GaussianProcess.hpp"
#include <random>
#include <vector>
#include <Eigen/Core>
#include "math.h"
#include <iostream>
#include <fstream>

std::random_device rd;
int randSeed = rd();
std::mt19937 gen (randSeed);  
std::normal_distribution<double> dis(0,1);

std::random_device rdu;
int randSeedu = rdu();
std::mt19937 genu (randSeedu);  
std::uniform_real_distribution<double> disu(-1,1);



double myfunc ( std::vector<double> const &x ) 
{
  return x[0] * cos(x[1]*2e0);
} 

int main ( int argv, char** argc ) 
{
  GaussianProcess GP(2);

  int nb_nodes = 100;

  std::vector< std::vector<double> > nodes;
  Eigen::VectorXd fvals(nb_nodes);
  Eigen::VectorXd noise(nb_nodes);

  std::vector<double> x(2);

  double scale = 1e0;
  for ( int i = 0; i < nb_nodes; ++i ) {
    if ( i > 20 ) scale = 0.01;
    x[0] = dis(gen)*scale;
    x[1] = dis(gen)*scale;
    nodes.push_back( x );
    fvals(i) = myfunc( x ) + disu(genu)*0.3;
    noise(i) = 0.3 + disu(genu)*0.05;
  }

  GP.estimate_hyper_parameters (  nodes, fvals, noise );
  GP.build( nodes, fvals, noise );

  double mean, var;

  std::ofstream outputfile ( "gp_data_o.dat" );
  if ( outputfile.is_open( ) ) {
    for (double i = -2.0; i <= 2.0; i+=0.1) {
      x[0] = i;
      for (double j = -2.0; j < 2.0; j+=0.1) {
        x[1] = j;
        GP.evaluate ( x, mean, var );
        outputfile << x[0] << "; " << x[1] << "; " << mean<< "; " << 
                     var << std::endl;
      }
    }
    outputfile.close( );
  } else std::cout << "Unable to open file." << std::endl;




  return 1;
}
