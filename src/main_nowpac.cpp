#include "NOWPAC.hpp"
#include "MyBlackBoxFunction.cpp"
#include "RegularizedMinimumFrobeniusNormModel.hpp"


int main () {

//randSeed = -1796460251;

//std::cout << "++++++++++++++++++++++++++++++" << std::endl; 
//std::cout << " random seed = " << randSeed << std::endl; 
//std::cout << "++++++++++++++++++++++++++++++" << std::endl; 



  MyBlackBoxFunction mybb;
  

  std::vector<int> uae ( 4 );
  uae[0] = 50;
  uae[1] = 100;
  uae[2] = 150;
  uae[3] = 250;

  int dim = 7;
  NOWPAC<> opt ( dim );
//  NOWPAC<RegularizedMinimumFrobeniusNormModel> opt ( dim );
  opt.set_blackbox( mybb, 4 );  
  opt.set_option( "eta_0" , 1e-3 );
  opt.set_option( "eta_1" , 1e-1 );
  opt.set_option( "eps_c"                         , 1e-3 );
  opt.set_option( "mu"                            , 1e0  );
  opt.set_option( "poisedness_threshold"          , 5e2  );
  opt.set_option( "inner_boundary_path_constants" , 1e-1 );
  opt.set_option( "verbose"                       , 3    );
  opt.set_option( "stochastic_optimization"       , false );
  opt.set_option( "update_at_evaluations"         , uae  );
  //  opt.set_option( "update_interval_length"        , 20   );
  opt.set_trustregion( 1e0, 1e-3 ); 
//  opt.set_max_number_evaluations( 6 );
//  opt.set_max_number_evaluations( 40 );
  std::vector<double> x0;
  
  if (dim == 4) {
    x0.resize( dim );
    x0[0] = 0e0;
    x0[1] = 0e0;
    x0[2] = 0e0;
    x0[3] = 0e0;
  }

  if (dim == 2) {
    x0.resize( dim );
    x0[0] = 0.5;
    x0[1] = 0.5;
  }
  //x0 << 0.5, 0.5;
  if ( dim ==  7) {
  x0.resize(7);
  x0.at(0) = 1e0;
  x0.at(1) = 2e0;
  x0.at(2) = 0e0;
  x0.at(3) = 4e0;
  x0.at(4) = 0e0;
  x0.at(5) = 1e0;
  x0.at(6) = 1e0;
  }
  double val;

//  std::cout << x0.size() << std::endl;

  opt.optimize(x0, val);

  if (x0.size() == 7 ) 
  std::cout << fabs(0.680630057275e03 - val)/0.680630057275e03 << std::endl;
  if (x0.size() == 2 ) 
  std::cout << fabs(1e0 - val) << std::endl;

  std::cout << "..." << std::endl;

  return 1;
}
