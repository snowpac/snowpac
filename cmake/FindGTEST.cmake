
INCLUDE (FindPackageHandleStandardArgs)

FIND_PATH ( NOWPAC_GTEST_INCLUDE_PATH gtest/gtest.h          
            PATHS "${CMAKE_INSTALL_PREFIX}/nowpac/external/gtest/src/GTEST/include/"
                  "${NOWPAC_INSTALL_PREFIX}/nowpac/external/gtest/src/GTEST/include/" )

FIND_LIBRARY ( NOWPAC_GTEST_LIBRARY NAMES gtest 
               PATHS "${CMAKE_INSTALL_PREFIX}/nowpac/external/gtest/src/GTEST-build/"
                     "${NOWPAC_INSTALL_PREFIX}/nowpac/external/gtest/src/GTEST-build/" ) 

FIND_PATH ( NOWPAC_GTEST_LIBRARY_PATH "libgtest.a" 
            PATHS "${CMAKE_INSTALL_PREFIX}/nowpac/external/gtest/src/GTEST-build/"
                  "${NOWPAC_INSTALL_PREFIX}/nowpac/external/gtest/src/GTEST-build/" )


FIND_PACKAGE_HANDLE_STANDARD_ARGS( GTEST DEFAULT_MSG NOWPAC_GTEST_INCLUDE_PATH NOWPAC_GTEST_LIBRARY NOWPAC_GTEST_LIBRARY_PATH)

