#(S)NOWPAC#
(Stochastic) Nonlinear Optimization With Path-Augmented Constraints

Author  : Florian Augustin, MIT (2014)
          Friedrich Menhorn, TUM (2018)
Contact : nowpac (at) mit (dot) edu 

## Derivative free optimizer for nonlinear constrained problems  ##
```
min f(x)  subject to c(x) <= 0 
```
where f denotes the objective function and c the constraints. 


External dependencies:
----------------------
- Eigen (linear algebra library, eigen.tuxfamily.org)
- NLopt (optimization library, ab-initio.mit.edu/wiki/index.php/NLopt)
- Gtest (google test library, optional)
- Boost (https://www.boost.org, use e.g. apt-get install libboost-all-dev on Unix)
- Compiler with C++11 standard support

Installation:
-------------
* **Automatic download and installation of dependencies**
```
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=<c++ compiler> 
      -DCMAKE_C_COMPILER=<c compiler>
      -DNOWPAC_INSTALL_PREFIX=<install_dir> 
      -DNOWPAC_ENABLE_SHARED=<ON|OFF>
      -DNOWPAC_ENABLE_TESTS=<ON|OFF>
../ 
make install -j4
```
* Default: NOWPAC_INSTALL_PREFIX = CMAKE_BINARY_DIRECTORY 
* Dependencies will be installed in `<install_dir>/nowpac/external/`
* **Manual installation of dependencies**
```
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=<c++ compiler> 
      -DCMAKE_INSTALL_PREFIX=<install_dir>
      -DNOWPAC_EIGEN_INCLUDE_PATH=</path/to/eigen/include>
      -DNOWPAC_NLOPT_INCLUDE_PATH=</path/to/nlopt/include> 
      -DNOWPAC_NLOPT_LIBRARY_PATH=</path/to/nlopt/library>
      -DNOWPAC_GTEST_INCLUDE_PATH=</path/to/gtest/include> 
      -DNOWPAC_GTEST_LIBRARY_PATH=</path/to/gtest/library>
      -DNOWPAC_ENABLE_SHARED=<ON|OFF> 
      -DNOWPAC_ENABLE_TESTS=<ON|OFF>
      ../ 
make install -j4
```
* Expected directory structure for Eigen: `</path/to/eigen/include>/Eigen/Dense`
* Expected directory structure for NLopt: `</path/to/nlopt/include>/nlopt.h`
* Expected directory structure for NLopt: `</path/to/nlopt/library>/libnlopt.dylib/so/a`
* Expected directory structure for Gtest: `</path/to/gtest/include>/gtest/gtest.h`
* Expected directory structure for Gtest: `</path/to/gtest/library>/libgtest.a`
* Libraries `libnowpac` / `libnowpacshared` will be built in `<install_dir>/nowpac/lib`
* Header files are contained in `<install_dir>/nowpac/include`


Required inputs:
----------------
* Number of design variables  
```
int n;
```
* Black box evaluator (see BlackBoxBaseClass.hpp) 
```
 void BlackBoxBaseClass::evaluate (std::vector<double> const &x, 
                                   std::vector<double> &vals, 
                                   void *<user parameter>);
```
vals[0] is the objective value, vals[1] ... vals[m] are the constraint values
* Initial trust-region radius
```
double delta_init;
```
* Minimal trust-region radius (or other stopping criteria)
```
double delta_min;
```
* Initial design
```
std::vector<double> x(n);
```

Optional inputs:
----------------
* Number of constraints
```
int m;
```
* Black box evaluator for noisy evaluations (see BlackBoxBaseClass.hpp) 
```
void BlackBoxBaseClass::evaluate (std::vector<double> const &x, 
                                  std::vector<double> &vals, 
                                  std::vector<double> &noise, 
                                  void *<user parameter>);
```
noise[0] is the magnitude of the noise (e.g. Monte Carlo standard error) in the objective function, noise[1], ..., noise[m] are the magnitudes of the noise in the constraint functions 1, ..., m

* Structure for user data
```
typedef struct { ... ; } <data_for_user_function>;
```


Internal parameters:
--------------------
* Reduction factor for trust-region in rejected steps (in ]0, 1[)
```
double gamma = 8e-1;
```
* Enlargement factor for trust-region in successful steps (> 1)
```
double gamma_inc = 1.4e0;
```
* Reduction factor for trust-region in model improvement (in ]0, 1[)
```
double omega = 6e-1;
```
* Reduction factor for trust-region if feasibility is violated (in ]0, 1[)
```
double theta = 5e-1;
```
* Threshold for step rejection (in [0, eta_1])
```
double eta_0 = 1e-1;
```
* Threshold for step acceptance (in [eta_0, 1[)
```
double eta_1 = 7e-1;
```
* Upper bound factor between trust-region radius and criticality measure (> 0)
```
double mu = 1e0;
```
* Threshold for trusting the criticality measure (> 0)
```
double eps_c = 1e-6;
```
* Inner boundary-path constant (> 0)
```
double eps_b = 1e1;
```
or, to set inner boundary-path constants for each constraint individually,
```
std::vector<double> eps_b(m);
```
* Maximal allowed trust-region radius (> delta_init)
```
double max_trust_region_radius = 1e0;
```
* Geometry threshold to control poisedness of trust-region nodes (> 1)
```
double geometry_threshold = 5e2;
```
* Output: no output (0), output final result (1), reduced output (2), detailed output (3)
```
int statistics = 1;
```
* Switch to turn noise detection on (true, false)
```
bool noise_detection = false;
```
* Termination of code due to detected noise (true, false)
```
bool noise_termination = false;
```
* Number of allowed noisy iterations (>= 0)
```
int allowed_noisy_iterations = 3;
```
* Number of rejected iterations after which the noise detection starts (>= 2)
```
int observation_span = 5;
```
* Set maximal number of acceptable/successful iterations (>= 1)
```
int max_nb_accepted_steps = inf;
```
* Allow stochastic optimization mode (true, false)
```
bool stochastic_optimization = false;
```
* Set after how many black-box evaluations the Gaussian process has to be updated (> 0)
```
std::vector<int> GP_adaption_steps;
```
* Set after how many black-box evaluations the Gaussian process has to be updated (> 0)
```
int GP_update_interval_length;
```


Usage of NOWPAC:
----------------
```
// Initialize NOWAC
NOWPAC<> <objectname>((int) n);
// alternatively if output to file is required: 
// NOWPAC<> <objectname>((int) n, (String) <name_of_output_file>);

// Set blackbox evaluator ( without constraints )
<objectname>.set_blackbox((BlackBoxBaseClass) <name of blackbox function>);

// Set blackbox evaluator ( with constraints, optional )
<objectname>.set_blackbox((&BlackBoxBaseClass) <name of blackbox function>, (int) m);

// Set user parameters (optional)
<objectname>.set_userdata(&<data_for_user_function>);

// Set initial trust-region radius (required, if not specified otherwise)
<objectname>.set_trustregion((double) delta_init);

// Set one of the following one (or more) of the following stopping criteria (required)
// 1) Set initial and minimal trust-region radii 
<objectname>.set_trustregion((double) delta_init, (double) delta_min);
// 2) Set maximal number of black-box function evaluations
<objectname>.set_max_number_evaluations((int) <max_number_of_evaluations>);
// 3) Set maximal number of acceptable/successful iterations

// Set lower bound constraints (optional)
<objectname>.set_lower_bounds(std::vector<double> <lower bounds, size (n)>);

// Set upper bound constraints (optional)
<objectname>.set_upper_bounds(std::vector<double> <upper bounds, size (n)>);

// Set options (optional)
<objectname>.set_option((std::string) <option name>, (type from table above) <option value>);

// Start optimization
(int) EXIT_FLAG = OPT.optimize( (std::vector<double>&) <x>, (double&) <opt_value>);
```


Output file:
--------------------

Column range  | Entry in output file
------------- | -------------
1  | number of evaluations
[2, 1+n] | current best point 
2+n | current objective best value
3+n | current trust region radius
[4+n, 3+n+m] | current constraint values at best point

Return values:
--------------------
* Maximal number of evaluations reached (1)
* Minimal trust region radius reached (0)
* Exiting due to noise (-2)
* Exiting due to parameter inconsistency (-4)
* Unexpected error in user function occurred (-5)
* Other termination reason (-6)
