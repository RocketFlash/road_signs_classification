import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transformations(aug_name='soft', image_size=448):
    all_transforms = {
        'no_aug' : A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        'super_soft' : A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        'soft_rtsd' : A.Compose([
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3),p=0.5),
            A.Cutout(num_holes=5, max_h_size=image_size//10, max_w_size=image_size//10, p=0.25),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        'medium_rtsd' : A.Compose([
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3),p=0.5),
            A.Blur(p=0.25, blur_limit=(3, 5)),
            A.GaussNoise(p=0.25, var_limit=(10.0, 50.0)),
            A.RGBShift(p=0.25, r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20)),
            A.RandomFog(p=0.1, fog_coef_lower=0.1, fog_coef_upper=0.44, alpha_coef=0.16),
            A.Cutout(num_holes=5, max_h_size=image_size//10, max_w_size=image_size//10, p=0.25),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        'soft' : A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=1.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2,0.2),p=0.5),
            A.Cutout(num_holes=8, max_h_size=image_size//5, max_w_size=image_size//5, p=0.5),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        'transforms_without_aug' : A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    }

    image_transforms = {
        'train': all_transforms[aug_name],
        'valid': all_transforms['transforms_without_aug'],
        'test': all_transforms['transforms_without_aug']
    }
    return image_transforms