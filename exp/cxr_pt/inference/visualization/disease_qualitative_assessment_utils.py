import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from PIL import Image


def visualize_disease_segmap(
    image_path,
    probability_map,
    overlay_alpha,
    class_names=None,
    dpi=200,
    bbox=None,
    colors=None,
    save_dir=None,
    duplication=False,
):

    orig_image = np.array(Image.open(image_path).convert("RGB"))
    height, width, _ = orig_image.shape

    segmentation_map = torch.argmax(probability_map, dim=0).detach().cpu().numpy()

    bg_index = None
    if class_names is not None and "background" in class_names:
        bg_index = class_names.index("background")

    num_classes = probability_map.shape[0]

    colors_list = []
    for i, class_name in enumerate(class_names):
        if class_name in colors:
            colors_list.append(colors[class_name])
        else:
            raise NotImplementedError(f"{class_name} is not in colors")

    cmap = ListedColormap(colors_list, name="fixed_cmap")

    colored_segmap = cmap(segmentation_map)

    if bg_index is not None:
        colored_segmap[..., 3] = np.where(
            segmentation_map == bg_index, 0, overlay_alpha
        )
    else:
        colored_segmap[..., 3] = overlay_alpha

    orig_image_norm = orig_image / 255.0

    alpha_channel = colored_segmap[..., 3:4]

    composite = (
        alpha_channel * colored_segmap[..., :3] + (1 - alpha_channel) * orig_image_norm
    )
    composite_display = (composite * 255).astype(np.uint8)

    plt.figure(figsize=(2 * width / dpi, height / dpi), dpi=dpi)

    plt.subplot(1, 2, 1)
    plt.imshow(orig_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(composite_display, extent=[0, width, height, 0])
    plt.title("Segmentation Map Overlay")
    plt.axis("off")

    if bbox is not None:
        draw_disease_bboxes(
            bbox, class_names, bg_index, width, height, colors, duplication
        )

    # background는 legend에서 제외해야 함
    handles = []
    for i in range(num_classes):
        label = class_names[i] if class_names is not None else f"Class {i}"
        if bg_index is not None and i == bg_index:
            continue
        patch = mpatches.Patch(color=cmap(i), label=label)
        handles.append(patch)

    plt.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=15)

    save_path = os.path.join(
        save_dir, f"seg_map_{image_path.split('/')[-1].replace('.png', '')}.png"
    )

    plt.savefig(
        save_path,
        bbox_inches="tight",
    )

    save_only_disease_segmap(
        segmentation_map,
        cmap,
        bbox,
        class_names,
        bg_index,
        width,
        height,
        dpi,
        overlay_alpha,
        orig_image_norm,
        colors,
        image_path,
        save_dir,
        duplication,
    )


def draw_disease_bboxes(
    bbox, class_names, bg_index, image_width, image_height, colors, duplication
):
    if bbox is not None:

        bbox_colors = []
        for i, class_name in enumerate(class_names):
            if class_name in colors:
                bbox_colors.append(colors[class_name])
            else:
                raise NotImplementedError(f"{class_name} is not in colors")

        for j, box in enumerate(bbox):
            x_min, y_min, x_max, y_max = box

            if bg_index is not None and j == bg_index:
                continue  # background 제외

            if (
                0 <= x_min <= 1
                and 0 <= y_min <= 1
                and 0 <= x_max <= 1
                and 0 <= y_max <= 1
            ):
                x_min *= image_width
                y_min *= image_height
                x_max *= image_width
                y_max *= image_height

            x_min = max(0, min(x_min, image_width))
            y_min = max(0, min(y_min, image_height))
            x_max = max(0, min(x_max, image_width))
            y_max = max(0, min(y_max, image_height))

            class_index = j
            color = bbox_colors[class_index]

            rect = plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=3,
                edgecolor=color,
                facecolor="none",
            )
            plt.gca().add_patch(rect)


def save_only_disease_segmap(
    segmentation_map,
    cmap,
    bbox,
    class_names,
    bg_index,
    width,
    height,
    dpi,
    overlay_alpha,
    orig_image_norm,
    colors,
    image_path,
    save_dir,
    duplication,
):

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    pure_segmap = cmap(segmentation_map)

    mask = (segmentation_map != bg_index)[..., None]
    alpha_channel = np.where(mask, overlay_alpha, 0)
    pure_segmap[..., 3] = alpha_channel.squeeze(-1)

    only_segmap_composite = (
        alpha_channel * pure_segmap[..., :3] + (1 - alpha_channel) * orig_image_norm
    )
    only_segmap_composite = (only_segmap_composite * 255).astype(np.uint8)

    ax.imshow(only_segmap_composite, extent=[0, width, height, 0])

    if bbox is not None:
        draw_disease_segmap_bboxes(
            ax, bbox, class_names, bg_index, width, height, colors, duplication
        )

    ax.axis("off")

    save_path = os.path.join(
        save_dir, f"only_segmap_{image_path.split('/')[-1].replace('.png', '')}.png"
    )
    plt.savefig(
        save_path,
        bbox_inches=None,
        pad_inches=0,
    )


def draw_disease_segmap_bboxes(
    ax, bbox, class_names, bg_index, image_width, image_height, colors, duplication
):
    if bbox is not None:

        colors_list = []
        for i, class_name in enumerate(class_names):
            if class_name in colors:
                colors_list.append(colors[class_name])
            else:
                raise NotImplementedError(f"{class_name} is not in colors")

        for j, box in enumerate(bbox):
            x_min, y_min, x_max, y_max = box

            if (
                0 <= x_min <= 1
                and 0 <= y_min <= 1
                and 0 <= x_max <= 1
                and 0 <= y_max <= 1
            ):
                x_min *= image_width
                y_min *= image_height
                x_max *= image_width
                y_max *= image_height

            x_min = max(0, min(x_min, image_width))
            y_min = max(0, min(y_min, image_height))
            x_max = max(0, min(x_max, image_width))
            y_max = max(0, min(y_max, image_height))

            if bg_index is not None and j == bg_index:
                continue  # background 제외

            class_index = j
            color = colors_list[class_index]

            rect = plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=3,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
