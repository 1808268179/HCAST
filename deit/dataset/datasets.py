# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os

from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from . import custom


def _resolve_custom_split_root(data_path, is_train):
    split_candidates = ['train'] if is_train else ['val', 'test']
    for split_name in split_candidates:
        root = os.path.join(data_path, split_name)
        if os.path.isdir(root):
            return root
    raise FileNotFoundError(
        f"Could not find expected split folder under {data_path}. "
        f"Tried: {', '.join(split_candidates)}"
    )


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if args.data_set == 'CUSTOM-HIER':
        root = _resolve_custom_split_root(args.data_path, is_train)
        dataset = custom.ImageFolder(
            root,
            transform=transform,
            is_hier=True,
        )
        num_classes = len(dataset.classes)
        nb_classes = [num_classes, num_classes]

    elif args.data_set == 'CUSTOM-HIER-SUPERPIXEL':
        from . import custom_seeds
        root = _resolve_custom_split_root(args.data_path, is_train)
        dataset = custom_seeds.ImageFolder(
            root,
            transform=transform,
            is_hier=True,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        num_classes = len(dataset.classes)
        nb_classes = [num_classes, num_classes]
    else:
        raise ValueError(
            f"Unsupported data_set={args.data_set}. "
            "Only CUSTOM-HIER and CUSTOM-HIER-SUPERPIXEL are supported."
        )


    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if 'INAT' in args.data_set:
        t.append(transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192]))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
