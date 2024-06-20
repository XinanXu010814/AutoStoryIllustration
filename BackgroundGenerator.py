from PIL import Image, ImageOps
import torch
from diffusers import StableDiffusionInpaintPipeline
import numpy as np

import Segmentation


def find_bounding_box(image):

    np_image = np.array(image)
    white_pixels = np.where(np_image > 0)

    if len(white_pixels[0]) == 0 or len(white_pixels[1]) == 0:
        return None

    min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
    min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])

    return min_x, min_y, max_x, max_y


def merge_bounding_boxes(bounding_box1, bounding_box2):
    min_x_1, min_y_1, max_x_1, max_y_1 = bounding_box1
    min_x_2, min_y_2, max_x_2, max_y_2 = bounding_box2
    return min(min_x_1, min_x_2), min(min_y_1, min_y_2), max(max_x_1, max_x_2), max(max_y_1, max_y_2)


def mask_outside_bounding_box(bounding_box, image):
    min_x, min_y, max_x, max_y = bounding_box
    np_image = np.array(image)

    np_image[:min_y, :] = 0
    np_image[max_y+1:, :] = 0
    np_image[:, :min_x] = 0
    np_image[:, max_x+1:] = 0

    masked_image = Image.fromarray(np_image)
    return masked_image


def merge_background(bounding_box, origin_image, character_image):
    min_x, min_y, max_x, max_y = bounding_box
    np_origin = np.array(origin_image)
    np_character = np.array(character_image)

    np_character[:min_y, :] = np_origin[:min_y, :]
    np_character[max_y+1:, :] = np_origin[max_y+1:, :]
    np_character[:, :min_x] = np_origin[:, :min_x]
    np_character[:, max_x+1:] = np_origin[:, max_x+1:]

    return Image.fromarray(np_character)

if __name__ == '__main__':

    # image = Image.open("tests/final_test_red_t_shirt/output_consistency.png")
    # for i in range(4):
    #     image_split = Segmentation.crop_image(image, i)
    #     mask_image = Segmentation.background_remove(image_split)
    #     bounding_box = find_bounding_box(mask_image)
    #     print(bounding_box)
    #     mask_image = mask_outside_bounding_box(bounding_box, ImageOps.invert(mask_image))
    generator = torch.manual_seed(128)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
    pipe.to("cuda")
    mask_image = Segmentation.background_remove(Segmentation.crop_image(Image.open(
        "outputs/jack_man_red_T-shirt/output_consistency.png"), 1))
    bounding_box = find_bounding_box(mask_image)
    print(bounding_box)
    mask_image = mask_outside_bounding_box(bounding_box, ImageOps.invert(mask_image))

    # this "i" could be changed
    image_origin = Image.open("outputs/jack_man_red_T-shirt/origin_1.png")
    image_merged = merge_background(bounding_box, image_origin, Segmentation.crop_image(Image.open(
        "outputs/jack_man_red_T-shirt/output_consistency.png"), 1))
    image_out = pipe(prompt="a strong man with red T-shirt, runs on a lawn", image=image_merged, mask_image=mask_image, generator=generator).images[0]
    image_out.save("reports/contrast_background.png")


    # generator = torch.manual_seed(128)
    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2-inpainting",
    #     torch_dtype=torch.float16,
    # )
    # pipe.to("cuda")
    # prompt = "He is standing, swimming pool"
    # #image and mask_image should be PIL images.
    # #The mask structure is white for inpainting and black for keeping as is
    #
    # img = Image.open("tests/jack_man_red_T-shirt/origin_0.png")
    #
    # mask_image = Image.open('segment_mask.png')
    # bounding_box = find_bounding_box(mask_image)
    # mask_image = mask_outside_bounding_box(bounding_box, ImageOps.invert(mask_image))
    #
    # image = pipe(prompt=prompt, image=img, mask_image=mask_image, generator=generator,).images[0]
    # image.save("./yellow_cat_on_park_bench6.png")


