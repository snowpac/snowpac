file(REMOVE_RECURSE
  "/usr/local/nowpac/lib/libnowpacshared.pdb"
  "/usr/local/nowpac/lib/libnowpacshared.dylib"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/nowpacshared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
