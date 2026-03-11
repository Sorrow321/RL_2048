"""
Test for Bug 1: merge() dropping tiles when two merges happen in the same row/column.

Before the fix, [2,2,4,4] sliding down produced [0,2,4,8] instead of [0,0,4,8].
The first tile of the second merge pair was left behind as a "ghost" tile.

NOTE: action() spawns a random tile after each valid move, so tests use a
nearly-full board where the spawn can only land in known empty cells,
and we assert on the cells that must be deterministic.
"""
import game


def test_double_merge_down():
    """Column [2,2,4,4] sliding DOWN should produce [_,_,4,8] in col 0."""
    g = game.Game()
    #               col0  col1  col2  col3
    g.set_state([
        [  2,    8,   32,  128],   # row 0
        [  2,   16,   64,  256],   # row 1
        [  4,   32,  128,  512],   # row 2
        [  4,   64,  256, 1024],   # row 3
    ])
    state, reward, done = g.action(0)  # DOWN

    # Merged tiles must be at the bottom of column 0
    assert state[2][0] == 4,  f"Expected 4 at [2][0], got {state[2][0]}"
    assert state[3][0] == 8,  f"Expected 8 at [3][0], got {state[3][0]}"
    assert reward == 4 + 8,   f"Expected reward 12, got {reward}"

    # Rows 0-1 of col 0 are the only empty spots; one gets the spawned tile
    spawned = [state[0][0], state[1][0]]
    assert sorted(spawned) in ([0, 2], [0, 4]), \
        f"Expected one spawn and one empty in col0 top, got {spawned}"
    print("PASS test_double_merge_down")


def test_double_merge_up():
    """Column [4,4,2,2] sliding UP should produce [8,4,_,_] in col 0."""
    g = game.Game()
    g.set_state([
        [  4,    8,   32,  128],
        [  4,   16,   64,  256],
        [  2,   32,  128,  512],
        [  2,   64,  256, 1024],
    ])
    state, reward, done = g.action(1)  # UP

    assert state[0][0] == 8,  f"Expected 8 at [0][0], got {state[0][0]}"
    assert state[1][0] == 4,  f"Expected 4 at [1][0], got {state[1][0]}"
    assert reward == 8 + 4,   f"Expected reward 12, got {reward}"

    spawned = [state[2][0], state[3][0]]
    assert sorted(spawned) in ([0, 2], [0, 4]), \
        f"Expected one spawn and one empty in col0 bottom, got {spawned}"
    print("PASS test_double_merge_up")


def test_double_merge_right():
    """Row [2,2,4,4] sliding RIGHT should produce [_,_,4,8] in row 0."""
    g = game.Game()
    g.set_state([
        [  2,    2,    4,    4],
        [  8,   16,   32,   64],
        [ 128, 256,  512, 1024],
        [  16,  32,   64,  128],
    ])
    state, reward, done = g.action(2)  # RIGHT

    assert state[0][2] == 4,  f"Expected 4 at [0][2], got {state[0][2]}"
    assert state[0][3] == 8,  f"Expected 8 at [0][3], got {state[0][3]}"
    assert reward == 4 + 8,   f"Expected reward 12, got {reward}"

    spawned = [state[0][0], state[0][1]]
    assert sorted(spawned) in ([0, 2], [0, 4]), \
        f"Expected one spawn and one empty in row0 left, got {spawned}"
    print("PASS test_double_merge_right")


def test_double_merge_left():
    """Row [4,4,2,2] sliding LEFT should produce [8,4,_,_] in row 0."""
    g = game.Game()
    g.set_state([
        [  4,    4,    2,    2],
        [  8,   16,   32,   64],
        [ 128, 256,  512, 1024],
        [  16,  32,   64,  128],
    ])
    state, reward, done = g.action(3)  # LEFT

    assert state[0][0] == 8,  f"Expected 8 at [0][0], got {state[0][0]}"
    assert state[0][1] == 4,  f"Expected 4 at [0][1], got {state[0][1]}"
    assert reward == 8 + 4,   f"Expected reward 12, got {reward}"

    spawned = [state[0][2], state[0][3]]
    assert sorted(spawned) in ([0, 2], [0, 4]), \
        f"Expected one spawn and one empty in row0 right, got {spawned}"
    print("PASS test_double_merge_left")


def test_quad_merge_down():
    """Column [4,4,4,4] sliding DOWN should produce [_,_,8,8] in col 0."""
    g = game.Game()
    g.set_state([
        [  4,    8,   32,  128],
        [  4,   16,   64,  256],
        [  4,   32,  128,  512],
        [  4,   64,  256, 1024],
    ])
    state, reward, done = g.action(0)  # DOWN

    assert state[2][0] == 8,  f"Expected 8 at [2][0], got {state[2][0]}"
    assert state[3][0] == 8,  f"Expected 8 at [3][0], got {state[3][0]}"
    assert reward == 8 + 8,   f"Expected reward 16, got {reward}"

    spawned = [state[0][0], state[1][0]]
    assert sorted(spawned) in ([0, 2], [0, 4]), \
        f"Expected one spawn and one empty in col0 top, got {spawned}"
    print("PASS test_quad_merge_down")


def test_single_merge_still_works():
    """Column [0,0,4,4] sliding DOWN should produce [_,_,_,8] — regression check."""
    g = game.Game()
    g.set_state([
        [  0,    8,   32,  128],
        [  0,   16,   64,  256],
        [  4,   32,  128,  512],
        [  4,   64,  256, 1024],
    ])
    state, reward, done = g.action(0)  # DOWN

    assert state[3][0] == 8,  f"Expected 8 at [3][0], got {state[3][0]}"
    assert reward == 8,       f"Expected reward 8, got {reward}"
    print("PASS test_single_merge_still_works")


if __name__ == "__main__":
    test_double_merge_down()
    test_double_merge_up()
    test_double_merge_right()
    test_double_merge_left()
    test_quad_merge_down()
    test_single_merge_still_works()
    print("\nAll merge tests passed!")
