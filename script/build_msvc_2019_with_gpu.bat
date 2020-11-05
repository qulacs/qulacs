mkdir visualstudio
cd visualstudio
cmake -G "Visual Studio 16 2019" -D USE_GPU:STR=Yes -D USE_MPI:STR=No .
cd ..
cmake --build ./visualstudio --target ALL_BUILD --config Release
cmake --build ./visualstudio --target python --config Release

