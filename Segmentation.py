from PIL import Image
from rembg import new_session, remove


def crop_image(image: Image, index: int):
    crop_box = (index * 512, 0, (index + 1) * 512, 512)
    return image.crop(crop_box)


def background_remove(image: Image):
    model_name = "isnet-general-use"
    session = new_session(model_name)
    output_mask = remove(image, session=session, only_mask=True)
    return output_mask
