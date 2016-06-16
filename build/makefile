# SPECIFICATION OF INCLUDE DIRECTORIES
NOWPAC_INST_DIR = /usr/local
PATH_SRC        = ../src
PATH_INC_EIGEN  = $(NOWPAC_INST_DIR)/nowpac/external/eigen/include/eigen3
PATH_INC_NLOPT  = $(NOWPAC_INST_DIR)/nowpac/external/nlopt/include
PATH_INC_NOWPAC = $(NOWPAC_INST_DIR)/nowpac/include
PATH_INCLUDE    = ../include
PATH_LIB_NOWPAC = $(NOWPAC_INST_DIR)/nowpac/lib
PATH_LIB_NLOPT  = $(NOWPAC_INST_DIR)/nowpac/external/nlopt/lib
PATH_INC_GTEST  = /Users/Florian/home/libraries/googletest-master/install/include
PATH_LIB_GTEST  = /Users/Florian/home/libraries/googletest-master/install/lib 

# SET COMPILER FLAGS
CFLAGS = -Wno-deprecated

# CHOOSE COMPILER
CC = /usr/local/Cellar/gcc/5.3.0/bin/g++-5

# ASSEMBLE LIB PATH
LIB_PATHS = -L$(PATH_LIB_NLOPT)
LIB_PATHS += -L$(PATH_LIB_GTEST)
#LIB_PATHS += -L$(PATH_LIB_NOWPAC)

# ASSEMBLE INCLUDE PATH
INCLUDE_PATHS = -I$(PATH_INC_EIGEN)
INCLUDE_PATHS += -I$(PATH_INC_NLOPT)
INCLUDE_PATHS += -I$(PATH_INC_GTEST)
INCLUDE_PATHS += -I$(PATH_INCLUDE)
#INCLUDE_PATHS += -I$(PATH_INC_NOWPAC)

# ASSEMBLE LIB PATH

# DEFINE OBJECT FILES NEEDED TO BUILD THE LIBRARY
OBJ = VectorOperations.o GaussianProcessSupport.o GaussianProcess.o TriangularMatrixOperations.o CholeskyFactorization.o BasisForMinimumFrobeniusNormModel.o  MinimumFrobeniusNormModel.o ImprovePoisedness.o QuadraticMonomial.o QuadraticMinimization.o

SURMODEL: $(OBJ) main_nowpac.o TEST
	$(CC) -g -std=c++11 $(INCLUDE_PATHS) $(CFLAGS) -o main_prog $(OBJ) main_nowpac.o $(LIB_PATHS) -lnlopt -lgtest -lm

TEST: $(OBJ) main_test.o
	$(CC) -g -std=c++11 $(INCLUDE_PATHS) $(CFLAGS) -o main_test $(OBJ) main_test.o $(LIB_PATHS) -lnlopt -lgtest -lm

GP: $(OBJ) main_gp.o
	$(CC) -g -std=c++11 $(INCLUDE_PATHS) $(CFLAGS) -o main_gp $(OBJ) main_gp.o $(LIB_PATHS) -lnlopt -lgtest -lm


%.o: $(PATH_SRC)/%.cpp
	$(CC) -g -std=c++11 $(INCLUDE_PATHS) $(CFLAGS) -c $<


clean:
	rm -f *.o

