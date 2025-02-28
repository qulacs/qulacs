if not defined USE_TEST (
    set USE_TEST=No
)
mkdir visualstudio
cd visualstudio
cmake -G "Visual Studio 17 2022" -D USE_GPU:STR=No -D USE_TEST=%USE_TEST% ..
cd ..
