import cv2
import numpy as np
import open_clip
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitImageProcessor,
    BlipImageProcessor,
    CLIPProcessor,
)
from transformers.image_transforms import convert_to_rgb


def load_processor(config):
    model_config = config["model_config"]

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["text_config"]["pretrained_tokenizer_name_or_path"]
    )

    # load image processor
    if model_config["vision_config"]["model_type"] in [
        "siglip",
        "dinov2",
        "sam",
    ]:
        if model_config["vision_config"]["model_type"] == "dinov2":

            if (
                model_config["vision_config"]["pretrained_name_or_path"]
                == "StanfordAIMI/dinov2-base-xray-224"
            ):
                if model_config["vision_config"].get("keep_aspect_ratio", False):
                    # import original dino config to put in
                    auto_processor = AutoProcessor.from_pretrained(
                        model_config["vision_config"]["pretrained_name_or_path"]
                    )

                    config_details = vars(auto_processor)

                    image_processor = AspectRatioBlipImageProcessor(**config_details)
                else:
                    image_processor = AutoProcessor.from_pretrained(
                        model_config["vision_config"]["pretrained_name_or_path"]
                    )
            else:
                image_processor = AutoProcessor.from_pretrained(
                    model_config["vision_config"]["pretrained_name_or_path"]
                )

        else:
            image_processor = AutoProcessor.from_pretrained(
                model_config["vision_config"]["pretrained_name_or_path"]
            )

    elif model_config["vision_config"]["model_type"] == "clip":
        image_processor = AutoProcessor.from_pretrained(
            model_config["vision_config"]["pretrained_name_or_path"]
        ).image_processor
    elif model_config["vision_config"]["model_type"] == "biomedclip":
        _, image_processor = open_clip.create_model_from_pretrained(
            model_config["vision_config"]["pretrained_name_or_path"]
        )
    elif model_config["vision_config"]["model_type"] == "xrayclip":
        image_processor = BlipImageProcessor.from_pretrained(
            model_config["vision_config"]["pretrained_name_or_path"]
        )
    elif model_config["vision_config"]["model_type"] == "m3ae":
        image_processor = M3AEImageProcessor()
    else:
        raise NotImplementedError()

    # adapt image size
    img_size = model_config["vision_config"].get("img_size")
    if img_size and image_processor:
        image_processor = adapt_img_size(image_processor, img_size, model_config)

    return {"tokenizer": tokenizer, "image_processor": image_processor}


def adapt_img_size(image_processor, img_size, model_config):
    if isinstance(image_processor, BitImageProcessor):
        image_processor.size["shortest_edge"] = img_size
        image_processor.crop_size = {"height": img_size, "width": img_size}

    elif isinstance(image_processor, BlipImageProcessor):
        image_processor.size = {"height": img_size, "width": img_size}

    elif isinstance(image_processor, CLIPProcessor):
        image_processor.size = {"height": img_size, "width": img_size}

    else:
        if model_config["vision_config"]["model_type"] == "biomedclip":
            image_processor.transforms[0].size = img_size
            image_processor.transforms[1].size = (img_size, img_size)

    return image_processor


# TODO: make integrated processor
# class VisionLanguageProcessor(ProcessorMixin):


class M3AEImageProcessor:
    def __init__(
        self,
        resize_size=256,
        crop_size=224,
        horizontal_flip=0.3,
        image_mean=[0.4978],
        image_std=[0.2449],
        degrees=[-30.0, 30.0],
        translate=[0.1, 0.1],
        scale=[0.9, 1.1],
        brightness=(0.8, 1.2),
        contrast=(0.8, 1.2),
    ):
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.horizontal_flip = horizontal_flip
        self.image_mean = image_mean
        self.image_std = image_std
        self.degrees = degrees
        self.translate = translate
        self.train_transform = []
        self.train_transform.append(transforms.RandomCrop(crop_size))
        self.train_transform.append(transforms.RandomHorizontalFlip(p=horizontal_flip))
        self.train_transform.append(
            transforms.RandomAffine(degrees, translate=translate, scale=scale)
        )
        self.train_transform.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
            )
        )
        self.train_transform.append(transforms.ToTensor())
        self.train_transform.append(
            transforms.Normalize(mean=image_mean, std=image_std)
        )
        self.train_transform = transforms.Compose(self.train_transform)

        self.inference_transform = []
        self.inference_transform.append(transforms.CenterCrop(crop_size))
        self.inference_transform.append(transforms.ToTensor())
        self.inference_transform.append(
            transforms.Normalize(mean=image_mean, std=image_std)
        )
        self.inference_transform = transforms.Compose(self.inference_transform)

    def __call__(self, images, train=False):
        if not isinstance(images, list):
            images = [images]

        processed_images = []
        for image in images:
            # PIL 이미지인 경우 numpy 배열로 변환 (cv2 처리를 위해)
            if isinstance(image, Image.Image):
                image = image.convert("L")
                image = np.array(image, dtype=np.uint8)

            image = resize_img(image, self.resize_size)

            image = Image.fromarray(image).convert("RGB")

            # remove augmentation
            # if train:
            #     processed_images.append(self.train_transform(image))
            # else:
            processed_images.append(self.inference_transform(image))

        processed_images = torch.stack(processed_images)

        return {"pixel_values": processed_images}


# copy from CARZero
def resize_img(img, scale):
    """
    Args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    Return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(
        img, desireable_size[::-1], interpolation=cv2.INTER_AREA
    )  # this flips the desireable_size vector

    # Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(
        resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
    )

    return resized_img


# aspect ratio padding dicon processor
class AspectRatioBlipImageProcessor(BlipImageProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, images, **kwargs):
        if not isinstance(images, list):
            images = [images]

        images = [convert_to_rgb(image) for image in images]

        # processing aspect ratio padding
        padded_images = [self.pad_to_square(image) for image in images]

        return super().__call__(padded_images, **kwargs)

    def pad_to_square(self, image: Image.Image, fill=(0, 0, 0)) -> Image.Image:
        """Pad image to make it square (maintain RGB)"""
        w, h = image.size
        if w == h:
            return image
        target_size = max(w, h)
        pad_left = (target_size - w) // 2
        pad_top = (target_size - h) // 2
        pad_right = target_size - w - pad_left
        pad_bottom = target_size - h - pad_top
        return ImageOps.expand(
            image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill
        )
