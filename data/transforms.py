import albumentations as A

def get_transforms(
    size,
    intensity="no_aug",
    mode="train",
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    if mode == "train":
        if intensity == "no_aug":
            return A.Compose(
                [
                    A.Resize(size, size, always_apply=True),
                    A.Normalize(
                        mean=mean, std=std, max_pixel_value=255.0, always_apply=True
                    ),
                ],
                keypoint_params=A.KeypointParams(format="xy"),
            )
        elif intensity == "heavy_aug":
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.5, rotate_limit=10, scale_limit=0.08),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=10,
                        val_shift_limit=10,
                        p=0.7,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7
                    ),
                    A.CLAHE(clip_limit=(1, 4), p=0.5),
                    A.OneOf(
                        [
                            A.GaussNoise(var_limit=[10, 50]),
                            A.GaussianBlur(),
                            A.MotionBlur(),
                            A.MedianBlur(),
                        ],
                        p=0.2,
                    ),
                    A.Resize(size, size, always_apply=True),
                    A.OneOf(
                        [
                            A.JpegCompression(),
                            A.Downscale(scale_min=0.1, scale_max=0.15),
                        ],
                        p=0.2,
                    ),
                    A.IAAPiecewiseAffine(p=0.2),
                    A.IAASharpen(p=0.2),
                    A.Normalize(
                        mean=mean, std=std, max_pixel_value=255.0, always_apply=True
                    ),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )
        elif intensity == "mild_aug":
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.5, rotate_limit=10, scale_limit=0.08),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=10,
                        val_shift_limit=10,
                        p=0.5,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7
                    ),
                    A.Resize(size, size, always_apply=True),
                    A.Normalize(
                        mean=mean, std=std, max_pixel_value=255.0, always_apply=True
                    ),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )
    else:
        return A.Compose(
            [
                A.Resize(size, size, always_apply=True),
                A.Normalize(
                    mean=mean, std=std, max_pixel_value=255.0, always_apply=True
                ),
            ],
            keypoint_params=A.KeypointParams(format="xy"),
        )
