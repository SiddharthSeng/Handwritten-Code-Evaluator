"""Generate sample test images with Python code rendered in a monospace font.

These images serve as canonical demo/test images for the project,
allowing anyone who clones the repo to test immediately without
needing their own handwritten image.
"""

from PIL import Image, ImageDraw, ImageFont


def generate_code_image(
    code_lines: list[str],
    output_path: str,
    width: int = 800,
    font_size: int = 26,
    bg_color: str = "white",
    text_color: str = "#222222",
) -> None:
    """Render lines of code onto a white image and save it.

    Uses a monospace font (Consolas on Windows, falls back to
    DejaVu Sans Mono or PIL default).
    """
    # Try common monospace fonts
    font = None
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "cour.ttf"):
        try:
            font = ImageFont.truetype(name, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    # Calculate image height from line count
    line_height = int(font_size * 1.7)
    padding_x, padding_y = 40, 40
    height = padding_y * 2 + line_height * len(code_lines)
    height = max(height, 400)

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    for i, line in enumerate(code_lines):
        y = padding_y + i * line_height
        draw.text((padding_x, y), line, fill=text_color, font=font)

    img.save(output_path)
    print(f"Saved: {output_path} ({width}x{height})")


if __name__ == "__main__":
    import os

    tests_dir = os.path.join(os.path.dirname(__file__), "..", "tests")
    os.makedirs(tests_dir, exist_ok=True)

    # Sample 1: greet function
    generate_code_image(
        code_lines=[
            'def greet(name):',
            '    print("Hello, " + name)',
            '',
            'greet("World")',
        ],
        output_path=os.path.join(tests_dir, "sample_handwritten.png"),
    )

    # Sample 2: loop with squares
    generate_code_image(
        code_lines=[
            'for i in range(5):',
            '    print(i * i)',
        ],
        output_path=os.path.join(tests_dir, "sample_loop.png"),
    )

    print("Done — generated 2 sample test images.")
