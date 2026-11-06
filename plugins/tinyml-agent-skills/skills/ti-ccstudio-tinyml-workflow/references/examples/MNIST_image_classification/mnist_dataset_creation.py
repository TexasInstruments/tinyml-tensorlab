import os
from pathlib import Path
from torchvision import datasets
from torchvision.utils import save_image
import torch

# Where your .gz files live (./MNIST/raw/*.gz inside this root)
MNIST_ROOT = Path("./data")
# Where you want classes/0..9 written
OUT_ROOT = Path("./mnist_classes")

def main():
    ds_train = datasets.MNIST(root=str(MNIST_ROOT), train=True,  download=False)
    ds_test  = datasets.MNIST(root=str(MNIST_ROOT), train=False, download=False)

    # Combine train + test since your flow splits later
    data   = torch.cat([ds_train.data,   ds_test.data],   dim=0)   # (70000, 28, 28)
    labels = torch.cat([ds_train.targets, ds_test.targets], dim=0) # (70000,)

    classes_dir = OUT_ROOT / "classes"
    classes_dir.mkdir(parents=True, exist_ok=True)
    for c in range(10):
        (classes_dir / str(c)).mkdir(parents=True, exist_ok=True)

    # Save as PNGs under classes/<digit>/
    for i in range(len(data)):
        img = data[i].unsqueeze(0).float() / 255.0      # (1,28,28) in [0,1]
        label = int(labels[i])
        # unique filename; include index so it’s stable
        dst = classes_dir / str(label) / f"{i:06d}.png"
        if not dst.exists():
            save_image(img, str(dst))

    print(f"Exported {len(data)} images into {classes_dir}/<0..9>")

if __name__ == "__main__":
    main()