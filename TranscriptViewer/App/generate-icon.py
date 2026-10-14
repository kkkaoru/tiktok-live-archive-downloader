"""Rasterize a 1024px app icon as PPM, then converted by generate-icon.sh."""

from pathlib import Path


def fill(
    px: bytearray, width: int, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]
) -> None:
    red, green, blue = color
    for y in range(max(0, y0), min(width, y1)):
        row = y * width * 3
        for x in range(max(0, x0), min(width, x1)):
            i = row + x * 3
            px[i] = red
            px[i + 1] = green
            px[i + 2] = blue


def main() -> None:
    width = 1024
    px = bytearray(width * width * 3)
    fill(px, width, 0, 0, width, width, (22, 38, 72))
    fill(px, width, 248, 96, 776, 720, (12, 20, 40))
    fill(px, width, 272, 120, 752, 696, (46, 86, 150))
    fill(px, width, 304, 560, 720, 600, (230, 236, 245))
    fill(px, width, 304, 616, 680, 656, (230, 236, 245))
    fill(px, width, 304, 672, 640, 712, (244, 201, 93))
    fill(px, width, 508, 120, 520, 712, (255, 255, 255))
    dest = Path(__file__).with_name("AppIcon.ppm")
    dest.write_bytes(b"P6\n1024 1024\n255\n" + px)
    print(dest)


if __name__ == "__main__":
    main()
