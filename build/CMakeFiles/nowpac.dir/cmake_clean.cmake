file(REMOVE_RECURSE
  "/usr/local/nowpac/lib/libnowpac.pdb"
  "/usr/local/nowpac/lib/libnowpac.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/nowpac.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
