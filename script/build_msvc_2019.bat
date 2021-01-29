mkdir visualstudio
cd visualstudio
cmake -G "Visual Studio 16 2019" -D BOOST_ROOT:STR=C:\boost_1_75_0 -D USE_GPU:STR=No -D USE_MPI:STR=No ..
cd ..
cmake --build ./visualstudio --target ALL_BUILD --config Release
cmake --build ./visualstudio --target python --config Release

