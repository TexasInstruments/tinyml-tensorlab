import random
import string
from io import BytesIO
from pathlib import Path

import numpy as np
import qrcode
from PIL import Image, ImageDraw
from barcode import Code128
from barcode.writer import ImageWriter


OUT_DIR = Path("machine_readable_code")
IMG_SIZE = 28
NUM_PER_CLASS = 3000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def random_text(min_len=4, max_len=16):
    chars = string.ascii_uppercase + string.digits
    length = random.randint(min_len, max_len)
    return "".join(random.choice(chars) for _ in range(length))


def to_28x28_binary(img):
    """
    Convert image to grayscale 28x28 black-white image.

    Important:
    - NEAREST resize keeps hard black/white edges.
    - thresholding removes antialiasing gray pixels.
    """
    img = img.convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)

    arr = np.array(img)
    arr = np.where(arr > 127, 255, 0).astype(np.uint8)

    return Image.fromarray(arr, mode="L")


def make_qr_image():
    """
    Create a simple black-white QR-like image.

    Note:
    For real scan-ability, 28x28 is too aggressive.
    For classification, it is fine because we only need QR structure.
    """
    payload = random_text()

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=4,
        border=1,
    )
    qr.add_data(payload)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    return to_28x28_binary(img)


def make_barcode_image():
    """
    Create a Code128 barcode image and resize to 28x28.

    We hide text because text under barcode will confuse the class signal.
    """
    payload = random_text(8, 16)

    barcode_obj = Code128(payload, writer=ImageWriter())

    buffer = BytesIO()
    barcode_obj.write(
        buffer,
        options={
            "write_text": False,
            "module_height": 12.0,
            "module_width": 0.35,
            "quiet_zone": 1.0,
            "dpi": 200,
        },
    )

    buffer.seek(0)
    img = Image.open(buffer)
    return to_28x28_binary(img)


def make_other_image():
    """
    Create blank/garbage non-code images.

    This intentionally mixes:
    - blank white
    - blank black
    - random noise
    - random simple lines/shapes

    That prevents the 'other' class from being just one trivial pattern.
    """
    mode = random.choice(["white", "black", "noise", "lines", "blocks"])

    if mode == "white":
        arr = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)

    elif mode == "black":
        arr = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    elif mode == "noise":
        arr = np.random.choice([0, 255], size=(IMG_SIZE, IMG_SIZE)).astype(np.uint8)

    elif mode == "blocks":
        arr = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
        for _ in range(random.randint(2, 8)):
            x1 = random.randint(0, IMG_SIZE - 4)
            y1 = random.randint(0, IMG_SIZE - 4)
            x2 = random.randint(x1 + 1, min(IMG_SIZE, x1 + 8))
            y2 = random.randint(y1 + 1, min(IMG_SIZE, y1 + 8))
            arr[y1:y2, x1:x2] = random.choice([0, 255])

    else:  # lines
        img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=255)
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(1, 6)):
            x1 = random.randint(0, IMG_SIZE - 1)
            y1 = random.randint(0, IMG_SIZE - 1)
            x2 = random.randint(0, IMG_SIZE - 1)
            y2 = random.randint(0, IMG_SIZE - 1)
            draw.line((x1, y1, x2, y2), fill=0, width=random.randint(1, 2))
        arr = np.array(img)

    return Image.fromarray(arr, mode="L")


def save_dataset():
    class_generators = {
        "qr": make_qr_image,
        "barcode": make_barcode_image,
        "other": make_other_image,
    }

    for class_name in class_generators:
        class_dir = OUT_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

    for class_name, generator in class_generators.items():
        print(f"Generating {class_name} images...")

        for i in range(NUM_PER_CLASS):
            img = generator()
            img.save(OUT_DIR / class_name / f"{class_name}_{i:05d}.png")

    print(f"Done. Dataset saved at: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    save_dataset()