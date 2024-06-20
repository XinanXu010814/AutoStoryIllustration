import csv

from dreamsim import dreamsim
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import torch
import numpy as np


def dreamSim_score(image1, image2):
    model, preprocess = dreamsim(pretrained=True)
    img1 = preprocess(image1).to("cuda")
    img2 = preprocess(image2).to("cuda")

    distance = model(img1, img2) # The model takes an RGB image from [0, 1], size batch_sizex3x224x224
    return distance


def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)


if __name__ == '__main__':
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    scores = []
    with open("CLIP_images.csv", newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if any(field.strip() for field in row):
                sd_clip_score = calculate_clip_score(np.array([Image.open(row[0])]), row[1])
                scores.append(sd_clip_score)
                print(f"CLIP score: {sd_clip_score}")
    scores_np = np.array(scores)
    print(f"CLIP score mean: {scores_np.mean()}, CLIP score std: {scores_np.std()}")