if not defined USE_TEST (
    set USE_TEST=No
)
mkdir visualstudio
cd visualstudio
cmake -G "Visual Studio 15 2017 Win64" -D USE_GPU:STR=Yes -D USE_TEST=%USE_TEST% ..
cd ..
