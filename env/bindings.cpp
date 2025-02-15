#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "2048_game.cpp"

namespace py = pybind11;

PYBIND11_MODULE(game, m) {
    m.doc() = "pybind11 2048 game module"; // Module docstring

    py::class_<Game>(m, "Game")
        .def(py::init<>())
        .def("simulate_action", &Game::simulate_action)
        .def("get_state", &Game::get_state)
        .def("get_current_score", &Game::get_current_score)
        .def("reset", &Game::reset)
        .def("get_possible_actions", &Game::get_possible_actions);
}