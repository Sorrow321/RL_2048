Build game .so

```
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` 2048_game.cpp binding.cpp -o game`python3-config --extension-suffix`
```