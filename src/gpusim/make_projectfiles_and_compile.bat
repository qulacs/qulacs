mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" ..
cd ../
cmake --build ./build --target ALL_BUILD --config Release

