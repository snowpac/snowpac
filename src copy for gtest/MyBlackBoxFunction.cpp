#include "BlackBoxBaseClass.hpp"
#include <math.h>
#include <iostream>
#include <random>

      std::random_device rd;
//      int randSeed = 25041981;//rd();
      int randSeed = rd();
      std::mt19937 gen (randSeed);  
      std::uniform_real_distribution<double> dis(-1,1);


class MyBlackBoxFunction : public BlackBoxBaseClass {
    int nbs = 1000;
    double sample;
    double noise_loc = 0e0;
  public:
    void evaluate( std::vector<double> const &x, std::vector<double> &vals, 
                   std::vector<double> &noise, void *data) override {

      if ( x.size() == 4 ) {
        vals.at(0) = pow(x.at(0),2e0)+pow(x.at(1),2e0)+2e0* pow(x.at(2),2e0)+
                     pow(x.at(3),2e0)-5e0*x.at(0)-5e0*x.at(1)-21e0
                     *x.at(2)+7e0*x.at(3);
        vals.at(1) = -(-pow(x.at(0),2e0)-pow(x.at(1),2e0)-pow(x.at(2),2e0)-
                      pow(x.at(3),2e0)-x.at(0)+x.at(1)-x.at(2)+x.at(3)+8e0);
        vals.at(2) = -(-pow(x.at(0),2e0)-2e0*pow(x.at(1),2e0)-pow(x.at(2),2e0)-
                      2e0*pow(x.at(3),2e0) +x.at(0)+x.at(3)+10e0);
        vals.at(3) = -(-2e0*pow(x.at(0),2e0) -pow(x.at(1),2e0)-pow(x.at(2),2e0)-
                      2e0*x.at(0)+x.at(1)+x.at(3)+5e0);
      }

      if ( vals.size() == 3) {
        vals.at(0) = pow(x.at(0)-2e0,2e0)+pow(x.at(1)-1e0,2e0);
        vals.at(1) = -(-pow(x.at(0),2e0)+x.at(1));
        vals.at(2) = -(x.at(0)-pow(x.at(1),2e0));
      }

      if ( vals.size() == 1 )
        vals.at(0) = pow( 1e0 - x.at(0), 2e0 ) + 1e0*pow( x.at(1) - x.at(0)*x.at(0), 2e0 );

      if ( x.size() == 7 ) {
        double V1 = 2e0*pow(x.at(0),2e0);   
        double V2 = pow(x.at(1),2e0);
        vals.at(0) = pow((x.at(0)-10e0), 2e0) + 5e0*pow((x.at(1)-12e0),2e0)+pow(x.at(2),4e0)
             +3e0*pow((x.at(3)-11e0),2e0)+10e0*pow(x.at(4),6e0)+7e0*pow(x.at(5),2e0)
             +pow(x.at(6),4e0)-4e0*x.at(5)*x.at(6)-10e0*x.at(5)-8e0*x.at(6) ;
        vals.at(1) = -(-V1-3e0*pow(V2,2e0)-x.at(2)-4e0*pow(x.at(3),2e0)-5e0*x.at(4)+127e0);
        vals.at(2) = -(-7e0*x.at(0)-3e0*x.at(1)-10e0*pow(x.at(2),2e0)- x.at(3)+x.at(4)+282e0);
        vals.at(3) = -(-23e0*x.at(0)-V2-6e0*pow(x.at(5),2e0) +8e0*x.at(6)+196e0);
        vals.at(4) = -(-2e0*V1-V2+3e0*x.at(0)*x.at(1) -2e0*pow(x.at(2),2e0)-5e0*x.at(5)+11e0*x.at(6));
      } 


      for (int j = 0; j < vals.size(); j++ ) {
        noise_loc = 0e0;
        noise.at(j) = 0e0;
        for ( int i = 0; i < nbs; i++ ) {
          sample = dis(gen);
          noise.at( j ) += sample*sample;
          noise_loc += sample;
        }
        vals.at( j ) += noise_loc / ((double)nbs);
        noise.at( j ) = 2e0*sqrt( noise.at(j) / ((double) nbs-1)) / sqrt((double) nbs);
      }

//    std::cout << vals << std::endl;
//    std::cout << noise << std::endl << std::endl;

    return;
  }


  void evaluate( std::vector<double> const &x, std::vector<double> &vals, 
                   void *data) override {

      if ( vals.size() == 3) {
        vals.at(0) = pow(x.at(0)-2e0,2e0)+pow(x.at(1)-1e0,2e0);
        vals.at(1) = -(-pow(x.at(0),2e0)+x.at(1));
        vals.at(2) = -(x.at(0)-pow(x.at(1),2e0));
      }

      if ( vals.size() == 1 )
       vals.at(0) = pow( 1e0 - x.at(0), 2e0 ) + 1e0*pow( x.at(1) - x.at(0)*x.at(0), 2e0 );

      if ( vals.size() == 5 ) {
        double V1 = 2e0*pow(x.at(0),2e0);   
        double V2 = pow(x.at(1),2e0);
        vals.at(0) = pow((x.at(0)-10e0), 2e0) + 5e0*pow((x.at(1)-12e0),2e0)+pow(x.at(2),4e0)
             +3e0*pow((x.at(3)-11e0),2e0)+10e0*pow(x.at(4),6e0)+7e0*pow(x.at(5),2e0)
             +pow(x.at(6),4e0)-4e0*x.at(5)*x.at(6)-10e0*x.at(5)-8e0*x.at(6) ;
        vals.at(1) = -(-V1-3e0*pow(V2,2e0)-x.at(2)-4e0*pow(x.at(3),2e0)-5e0*x.at(4)+127e0);
        vals.at(2) = -(-7e0*x.at(0)-3e0*x.at(1)-10e0*pow(x.at(2),2e0)- x.at(3)+x.at(4)+282e0);
        vals.at(3) = -(-23e0*x.at(0)-V2-6e0*pow(x.at(5),2e0) +8e0*x.at(6)+196e0);
        vals.at(4) = -(-2e0*V1-V2+3e0*x.at(0)*x.at(1) -2e0*pow(x.at(2),2e0)-5e0*x.at(5)+11e0*x.at(6));
      } 

    return;
  }

};
