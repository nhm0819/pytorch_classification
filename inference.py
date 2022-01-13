import argparse
import os
import pandas as pd
import cv2
import torch
import albumentations as A
import shutil
from dataset import TestDataset

from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from torch.utils.data import Dataset
from model import WireClassifier


parser = argparse.ArgumentParser(description='PyTorch Classification')
parser.add_argument('--model_name', type=str, default="efficientnet_b4", metavar='S',
                    help='model name')
parser.add_argument('--model_path', type=str, default="efficientnet_b4_result_1_4.pt", metavar='S',
                    help='model path')
parser.add_argument('--num_classes', type=int, default=2, metavar='N',
                    help='num classes')
parser.add_argument('--num_workers', type=int, default=6, metavar='N',
                    help='num workers')

parser.add_argument('--dataset_dir', type=str, default="E:\\work\\kesco\\raw_data\\20211008", metavar='S',
                    help='model path')
parser.add_argument('--df_path', type=str, default="segmented_test.csv", metavar='S',
                    help='model path')
parser.add_argument('--save_dir_name', type=str, default="test_all_result_5", metavar='S',
                    help='model path')




def main():
    args = parser.parse_args()
    # use gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model init
    model = WireClassifier(args)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    # test data load
    test_df = pd.read_csv(args.df_path).reset_index(drop=True)

    # transform
    # transform = A.Compose([
    #     A.Resize(height=380, width=380),
    #     A.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]),
    #     ToTensorV2()
    # ])

    #####
    transform = A.Compose([
        A.RandomResizedCrop(height=380, width=380, scale=(0.8, 1.0)),
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
        # A.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]),
        A.Rotate(limit=15, p=0.3),
        ToTensorV2(always_apply=True)
    ])
    #####


    # dataset
    test_dataset = TestDataset(args, test_df, transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers)

    # correct variable for calculate accuracy
    num_correct = 0
    num_data = len(test_df)

    # for idx in tqdm(range(num_data)):
    #     # image read
    #     dataset_dir = "E:\\work\\kesco\\raw_data\\20211008\\segmented_good_data"
    #     file_name = test_df['path'][idx].split("\\")[-1]
    #     img_path = os.path.join(dataset_dir, test_df['path'][idx])
    #     img_path = img_path.replace("/", "\\")
    #     img = cv2.imread(img_path)
    #     image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     image = transform(image=image)['image']
    #
    #     # label read
    #     label = test_df['label'][idx]

    correct_imgs = []
    wrong_imgs = []

    # data load
    for batch_idx, (image, label, img_path) in tqdm(enumerate(test_loader)):
        # inference
        input = image.to(device)
        with torch.no_grad():
            pred = model(input)
        pred_label = pred.max(1)[1]# .to("cpu").numpy()

        pred_res = pred_label.eq(label.to(device)).cpu()


        for i, res in enumerate(pred_res):
            if res == 1:
                correct_imgs.append(img_path[i])
            elif res == 0:
                wrong_imgs.append(img_path[i])

        num_correct += pred_res.sum().item()

    print(f'\nAccuracy : {num_correct / num_data:.2f}\n')


    # image save path
    dataset_name = args.dataset_dir.split("\\")[-1]
    save_folder = args.dataset_dir.replace(dataset_name, args.save_dir_name)
    correct_folder = os.path.join(save_folder, "correct")
    wrong_folder = os.path.join(save_folder, "wrong")

    # make directories
    os.makedirs(correct_folder, exist_ok=True)
    os.makedirs(wrong_folder, exist_ok=True)

    print("copy correct images...")
    for correct_img in correct_imgs:
        file_name = correct_img.split("\\")[-1]
        save_img_path = os.path.join(correct_folder, file_name)
        shutil.copy(correct_img, save_img_path)

    print("\nwrong correct images...")
    for wrong_img in wrong_imgs:
        file_name = wrong_img.split("\\")[-1]
        save_img_path = os.path.join(wrong_folder, file_name)
        shutil.copy(wrong_img, save_img_path)


    # split images into result
    # if pred_label == label:
    #     save_img_path = os.path.join(correct_path, file_name)
    #     shutil.copy2(img_path, save_img_path)
    #     num_correct += 1
    #
    # else:
    #     save_img_path = os.path.join(wrong_path, file_name)
    #     shutil.copy2(img_path, save_img_path)





if __name__ == '__main__':
    main()