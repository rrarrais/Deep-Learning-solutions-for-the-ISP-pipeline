import os

from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
from skimage import io

# This code needs revision before being used.
# Its main objective is to unite output and ground truth images inside a single plot

def plot_sample(sample, name="teste"):
    os.makedirs("/workspace/pytorch_dncnn/apply/output", exist_ok=True)

    fig = plt.figure()

    # Input
    ax = plt.subplot(1, 2, 1)
    ax.set_title("Network output")
    ax.axis("off")
    plt.imshow(sample["output"])
    plt.pause(0.001)

    # Output
    bx = plt.subplot(1, 2, 2)
    bx.set_title("Ground truth")
    bx.axis("off")
    # output = sample['output'].numpy().transpose(1, 2, 0)
    plt.imshow(sample["gt"])
    plt.pause(0.001)

    fig.savefig(f"/workspace/pytorch_dncnn/apply/output/{name}.jpg")


def _main():
    input_folder = "./apply"

    for idx in range(8):
        output_image = io.imread(os.path.join(input_folder, f"output{idx}.png"))
        ground_image = io.imread(os.path.join(input_folder, f"groundtruth{idx}.png"))
        plot_sample({"output":output_image, "gt":ground_image}, f"image{idx}")

if __name__ == "__main__":
    _main()