Build game .so

```
c++ -O3 -std=c++17 -Wall -shared -fPIC `python3 -m pybind11 --includes` 2048_game.cpp bindings.cpp -o game`
python3-config --extension-suffix`
```
