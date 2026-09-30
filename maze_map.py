from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from myenv import MyEnv


OUTPUT = Path("report_assets/maze_map.png")


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))

    for row in range(MyEnv.HEIGHT):
        for col in range(MyEnv.WIDTH):
            is_wall = MyEnv.GRID[row][col] == "1"

            ax.add_patch(
                Rectangle(
                    (col, MyEnv.HEIGHT - 1 - row),
                    1,
                    1,
                    facecolor="black" if is_wall else "white",
                    edgecolor="gray",
                    linewidth=1.0,
                )
            )

    sr, sc = MyEnv.START
    gr, gc = MyEnv.GOAL

    ax.text(
        sc + 0.5,
        MyEnv.HEIGHT - 1 - sr + 0.5,
        "S",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
    )

    ax.text(
        gc + 0.5,
        MyEnv.HEIGHT - 1 - gr + 0.5,
        "G",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
    )

    ax.set_xlim(0, MyEnv.WIDTH)
    ax.set_ylim(0, MyEnv.HEIGHT)
    ax.set_aspect("equal")

    ax.set_xticks(range(MyEnv.WIDTH + 1))
    ax.set_yticks(range(MyEnv.HEIGHT + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    ax.set_title(
        "CS 272 PA2 Maze\n"
        "S = Start (0,0), G = Goal (9,8)"
    )

    ax.legend(
        handles=[
            Patch(facecolor="black", edgecolor="gray", label="Wall (1)"),
            Patch(facecolor="white", edgecolor="gray", label="Open (0)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()