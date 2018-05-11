INCLUDE(ExternalProject)

SET(NOWPAC_NLOPT_URL "http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz")

ExternalProject_Add( NLOPT
   
   PREFIX ${NOWPAC_INSTALL_PREFIX}/nowpac/external/nlopt/build
   URL ${NOWPAC_NLOPT_URL}
   CONFIGURE_COMMAND ../NLOPT/configure --prefix=${NOWPAC_INSTALL_PREFIX}/nowpac/external/nlopt --without-octave --without-matlab --enable-shared --enable-static
   BUILD_COMMAND make 
   INSTALL_COMMAND make install
   LOG_DOWNLOAD 1
   LOG_CONFIGURE 1
   LOG_BUILD 1
   LOG_INSTALL 1
   BUILD_IN_SOURCE 0

)

#SET_PROPERTY( TARGET NLOPT PROPERTY FOLDER nowpac/external)
SET(NOWPAC_NLOPT_INCLUDE_PATH ${NOWPAC_INSTALL_PREFIX}/nowpac/external/nlopt/include)
SET(NOWPAC_NLOPT_LIBRARY_PATH ${NOWPAC_INSTALL_PREFIX}/nowpac/external/nlopt/lib)


