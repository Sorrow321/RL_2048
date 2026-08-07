"""Render a recorded game (az.watch JSON) as an animated GIF for the README.

    python -m az.render_gif runs/best_8192_game.json assets/demo.gif --stride 10
"""
import argparse
import json

from PIL import Image, ImageDraw, ImageFont

TILE_BG = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563",
    32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61",
    512: "#edc850", 1024: "#edc22e", 2048: "#edc22f",
}
DARK_TILE = "#3c3a32"  # 4096+
BOARD_BG = "#bbada0"
PAGE_BG = "#faf8ef"
TEXT = "#776e65"

CELL, GAP, HEADER = 64, 8, 34
SIZE = 4 * CELL + 5 * GAP  # board pixels
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def tile_font(value, cache={}):
    size = 24 if value < 128 else (20 if value < 1024 else 16)
    if size not in cache:
        cache[size] = ImageFont.truetype(FONT_PATH, size)
    return cache[size]


def render_frame(board_hex, header):
    b = int(board_hex, 16)
    img = Image.new("RGB", (SIZE, SIZE + HEADER), PAGE_BG)
    d = ImageDraw.Draw(img)
    d.text((GAP, 8), header, fill=TEXT, font=ImageFont.truetype(FONT_PATH, 15))
    d.rounded_rectangle([0, HEADER, SIZE, HEADER + SIZE], radius=6, fill=BOARD_BG)
    for r in range(4):
        for c in range(4):
            v = (b >> (4 * (r * 4 + c))) & 0xF
            t = (1 << v) if v else 0
            x = GAP + c * (CELL + GAP)
            y = HEADER + GAP + r * (CELL + GAP)
            d.rounded_rectangle([x, y, x + CELL, y + CELL], radius=4,
                                fill=TILE_BG.get(t, DARK_TILE))
            if t:
                f = tile_font(t)
                tw = d.textlength(str(t), font=f)
                th = f.size
                d.text((x + (CELL - tw) / 2, y + (CELL - th) / 2 - 2),
                       str(t), fill=TEXT if t <= 4 else "#f9f6f2", font=f)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game_json")
    ap.add_argument("out_gif")
    ap.add_argument("--stride", type=int, default=10, help="render every Nth move")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    game = json.load(open(args.game_json))
    frames = []
    score = 0
    for i, m in enumerate(game["moves"]):
        score += m["reward"]
        if i % args.stride == 0:
            frames.append(render_frame(
                m["board"], f"move {i + 1:>5}   score {score:>7,}"))
    frames.append(render_frame(
        game["final_board"],
        f"game over — score {game['score']:,}  max {game['max_tile']}"))
    # hold the final frame
    durations = [1000 // args.fps] * (len(frames) - 1) + [3000]

    frames[0].save(args.out_gif, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"wrote {args.out_gif}: {len(frames)} frames")


if __name__ == "__main__":
    main()
