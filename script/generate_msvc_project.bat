mkdir visualstudio
cd visualstudio
cmake -G "Visual Studio 14 2015 Win64" -D USE_PYTHON:STR=Yes -D USE_GPU:STR=Yes -D USE_TEST:STR=Yes ..
cd ..
