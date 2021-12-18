mkdir visualstudio
cd visualstudio
cmake -G "Visual Studio 16 2019" -A "x64" -D USE_GPU:STR=Yes ..
cd ..
cmake --build ./visualstudio --target ALL_BUILD --config Release
cmake --build ./visualstudio --target python --config Release

