mkdir visualstudio
cd visualstudio
cmake -G "Visual Studio 15 2017 Win64" -D USE_GPU:STR=No -D USE_MPI:STR=No ..
cd ..
