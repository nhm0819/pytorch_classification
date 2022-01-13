import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_transforms(img_size, data):
    if data == 'train':
        return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.GaussNoise(),
            A.NoOp(),
            A.MultiplicativeNoise(),
            A.ISONoise()
        ], p=0.5),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.PiecewiseAffine(p=0.3),
        ], p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            A.RandomGamma()
        ], p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        A.Rotate(limit=15, p=0.3),
        ToTensorV2(always_apply=True)
    ])


    elif data == 'test':
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            ToTensorV2(always_apply=True)
        ])
