# SPECIFICATION OF INCLUDE DIRECTORIES
NOWPAC_INST_DIR = /usr/local
PATH_SRC        = ../src
PATH_INC        = ../src
PATH_INC_NOWPAC = ../../include
PATH_SRC_NOWPAC = ../../src
PATH_INC_NLOPT  = $(NOWPAC_INST_DIR)/nowpac/external/nlopt/include
PATH_INC_EIGEN  = $(NOWPAC_INST_DIR)/nowpac/external/eigen/include/eigen3
PATH_INC_GTEST  = /Users/Florian/home/libraries/googletest-master/install/include


# SPECIFICATION OF LIB DIRECTORIES
PATH_LIB_NLOPT  = $(NOWPAC_INST_DIR)/nowpac/external/nlopt/lib
PATH_LIB_GTEST  = /Users/Florian/home/libraries/googletest-master/install/lib 

# SET COMPILER FLAGS
CFLAGS = -Wno-deprecated 

# CHOOSE COMPILER
CC = /usr/local/Cellar/gcc/5.3.0/bin/g++-5


# ASSEMBLE INCLUDE PATH
INCLUDE_PATHS = -I$(PATH_INC)
INCLUDE_PATHS += -I$(PATH_INC_NOWPAC)
INCLUDE_PATHS += -I$(PATH_INC_NLOPT)
INCLUDE_PATHS += -I$(PATH_INC_EIGEN)
INCLUDE_PATHS += -I$(PATH_INC_GTEST)


# ASSEMBLE LIB PATH
LIB_PATHS = -L$(PATH_LIB_NLOPT)
LIB_PATHS += -L$(PATH_LIB_GTEST)


# DEFINE OBJECT FILES NEEDED TO BUILD THE LIBRARY
OBJNOWPAC = VectorOperations.o GaussianProcessSupport.o GaussianProcess.o TriangularMatrixOperations.o CholeskyFactorization.o BasisForMinimumFrobeniusNormModel.o  MinimumFrobeniusNormModel.o ImprovePoisedness.o QuadraticMonomial.o QuadraticMinimization.o
OBJ = main_test.o

NOWPAC: $(OBJ) $(OBJNOWPAC)
	$(CC) -g -std=c++11 $(INCLUDE_PATHS) $(CFLAGS) -o main_test $(OBJ) $(OBJNOWPAC) $(LIB_PATHS) -lnlopt -lgtest 

%.o: $(PATH_SRC_NOWPAC)/%.cpp 
	$(CC) -g -std=c++11 $(INCLUDE_PATHS) $(CFLAGS) -c $<

%.o: $(PATH_SRC)/%.cpp 
	$(CC) -g -std=c++11 $(INCLUDE_PATHS) $(CFLAGS) -c $<

clean:
	rm -f $(OBJ) $(OBJNOWPAC)

