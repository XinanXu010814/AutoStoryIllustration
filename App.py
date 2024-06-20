import gradio as gr
import os

from PIL.Image import Resampling
from controlnet_aux import *
from PIL import Image, ImageOps
from controlnet_aux.open_pose import PoseResult, Keypoint
from diffusers import StableDiffusionPipeline, AutoencoderKL, ControlNetModel, StableDiffusionControlNetPipeline, \
    LCMScheduler, UniPCMultistepScheduler, StableDiffusionInpaintPipeline
import torch
import os.path as path
import numpy as np

from Segmentation import *
from BackgroundGenerator import mask_outside_bounding_box, find_bounding_box, merge_background, merge_bounding_boxes
from TokenExtracter import *
from test import denormalize, pt_to_numpy, numpy_to_pil
from OpenposeAdapter import face_correction, scale_up_openpose, scale_down_character, face_blend, openpose_paint


def iterate_neighbor(depth_map, width, height):
    pixel_sum = 0
    pixel_max = 0
    for i in range(8):
        for j in range(8):
            pixel_val = depth_map.getpixel((width * 8 + i, height * 8 + j))
            if sum(pixel_val) / 3 > pixel_max:
                pixel_max = sum(pixel_val) / 3
            pixel_sum += sum(pixel_val) / 3
    return pixel_sum / 64, pixel_max


def remap(latents_origin, latents_out, depth_maps):
    outputs = []
    for i in range(len(depth_maps)):
        max_val = np.max(np.array(depth_maps[i]))
        out = torch.zeros(latents_origin[i].shape, dtype=torch.float16)
        for j in range(64):
            for k in range(64):
                # pixel_val = depth_maps[i].getpixel((k, j))
                pixel_avg, pixel_max = iterate_neighbor(depth_maps[i], k, j)
                # print(pixel_val)
                if pixel_max < max_val * 0.9:
                    out[:, :, j, k] = latents_origin[i][:, :, j, k]
                else:
                    out[:, :, j, k] = (latents_out[i][:, :, j, k] * (pixel_avg / 255)
                                       + latents_origin[i][:, :, j, k] * (1 - pixel_avg / 255))
        outputs.append(out)
    return outputs


def run_latent_method(main_prompt, sub_dir, pose):
    full_dir = path.join("outputs", sub_dir)

    if not path.exists(full_dir):
        os.mkdir(full_dir)

    ### record prompt ###
    with open(path.join(full_dir, f"prompt.txt"), "w") as file:
        file.write(main_prompt)

    preprocessor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

    preprocessor1 = MidasDetector.from_pretrained("lllyasviel/Annotators")

    generator = torch.manual_seed(128)

    pipe = StableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-7", torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")  # yes, it's a normal LoRA

    actions, descriptions, nouns = extract_tokens(main_prompt)

    prompt_num = len(actions)

    prompts = []
    prompt = ""
    for noun in nouns:
        prompt += noun
        prompt += ", "
    for description in descriptions:
        prompt += description
        prompt += ", "
    concat = ", full body, " if pose else ", "
    for i in range(prompt_num):
        prompts.append(actions[i] + concat + prompt)
    print(prompts)
    main_prompt = "full body, " + prompt

    latents_origin = []
    depths = []
    for i in range(len(prompts)):
        results = pipe(
            prompt=prompts[i],
            num_inference_steps=4,
            guidance_scale=0.0,
            # num_inference_steps=20,
            # guidance_scale=8.0,
            output_type="latent",
            generator=generator,
        )

        ### results : [1, 4, 64, 64]
        ###################
        latents_origin.append(results.images)
        image = \
        pipe.vae.decode(results.images / pipe.vae.config.scaling_factor, return_dict=False, generator=generator)[
            0
        ]
        image = torch.stack(
            [denormalize(image[i]) for i in range(image.shape[0])]
        )
        image = pt_to_numpy(image)
        image = numpy_to_pil(image)[0]

        ######################

        image.save(path.join(full_dir, f"origin_{i}.png"))
        if pose:
            preprocessor(image, include_hand=True, include_face=True).save(path.join(full_dir, f"openpose_{i}.png"))

        depth_map = preprocessor1(image)
        # print("pixel val: ", depth_map.getpixel((250, 250)))
        depths.append(depth_map)
        depth_map.save(path.join(full_dir, f"depth_{i}.png"))
        depth_map.resize((64, 64), resample=Resampling.BILINEAR).save(path.join(full_dir, f"depth_{i}64.png"))

    if pose:
        # Combine images horizontally
        combined_image = Image.new('RGB', (512 * prompt_num, 512))
        for i in range(prompt_num):
            image = Image.open(path.join(full_dir, f'openpose_{i}.png'))
            combined_image.paste(image, (i * 512, 0))

        # Save the result
        combined_image.save(path.join(full_dir, 'combined_openpose.png'))

    # Combine images horizontally
    combined_image = Image.new('RGB', (512 * prompt_num, 512))
    for i in range(prompt_num):
        image = Image.open(path.join(full_dir, f'depth_{i}.png'))
        combined_image.paste(image, (i * 512, 0))

    # Save the result
    combined_image.save(path.join(full_dir, 'combined_depth.png'))

    ''' Merge all images '''
    controlnet = [
        ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_openpose', torch_dtype=torch.float16),
        ControlNetModel.from_pretrained('lllyasviel/control_v11f1p_sd15_depth', torch_dtype=torch.float16),
    ] if pose \
        else [ControlNetModel.from_pretrained('lllyasviel/control_v11f1p_sd15_depth', torch_dtype=torch.float16)]

    control_image = [
        Image.open(path.join(full_dir, 'combined_openpose.png')),
        Image.open(path.join(full_dir, 'combined_depth.png')),
    ] if pose \
        else [Image.open(path.join(full_dir, 'combined_depth.png'))]

    generator = torch.manual_seed(128)

    pipe = StableDiffusionControlNetPipeline.from_pretrained("Lykon/dreamshaper-7", controlnet=controlnet,
                                                             torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")  # yes, it's a normal LoRA

    results = pipe(
        prompt=main_prompt,
        num_inference_steps=4,
        guidance_scale=0.3,
        image=control_image,
        height=512,
        width=prompt_num * 512,
        # height=1024,
        # width=1024,
        generator=generator,
        output_type="latent",
        controlnet_conditioning_scale=[0.5]
    )
    # image = results.images[0]
    image_tensor = results.images[0].cpu()
    image_array = image_tensor.numpy()
    image_array = np.transpose(image_array, (1, 2, 0))
    data_normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min())

    # Convert to uint8 in the range [0, 255]
    image_array = (data_normalized * 255).astype(np.uint8)
    image_array = image_array.astype(np.uint8)
    image = Image.fromarray(image_array)
    image.save(path.join(full_dir, 'output_latent.png'))

    latents_out = []
    for i in range(prompt_num):
        latents_out.append(results.images[:, :, :, i * 64:i * 64 + 64])
    latents = remap(latents_origin=latents_origin, latents_out=latents_out, depth_maps=depths)

    ##### just for display #####
    image = pipe.vae.decode(results.images / pipe.vae.config.scaling_factor, return_dict=False, generator=generator)[
        0
    ]
    image = torch.stack(
        [denormalize(image[i]) for i in range(image.shape[0])]
    )
    image = pt_to_numpy(image)
    image = numpy_to_pil(image)
    image[0].save(path.join(full_dir, 'output_consistency.png'))

    """ Decode latent """
    print("shape A: ", results.images[0].shape)
    print("shape B: ", results.images.shape)

    combined_image = Image.new('RGB', (512 * prompt_num, 512))
    """ decode all latents """
    for j in range(len(latents)):
        image = \
        pipe.vae.decode(latents[j].to("cuda") / pipe.vae.config.scaling_factor, return_dict=False, generator=generator)[
            0
        ]
        print("decoding finish ......")
        image = torch.stack(
            [denormalize(image[i]) for i in range(image.shape[0])]
        )
        image = pt_to_numpy(image)
        image = numpy_to_pil(image)[0]
        image.save(path.join(full_dir, f'output_{j}.png'))
        combined_image.paste(image, (j * 512, 0))

    return combined_image


cur_sub_dir = ""
cur_img_count = 0


def run(character_prompt: str, scene_prompt: str, sub_dir: str, human: bool):
    full_dir = path.join("outputs", sub_dir)

    if not path.exists(full_dir):
        os.mkdir(full_dir)

    start_idx = 0
    if path.exists(path.join(full_dir, "count.txt")):
        with open(path.join(full_dir, "count.txt"), 'r') as file:
            content = file.read().strip()
            start_idx = int(content)

    ### record prompt ###
    with open(path.join(full_dir, f"prompt.txt"), "w") as file:
        file.write(character_prompt)
        file.write("\n")
        file.write(scene_prompt)

    if human:
        preprocessor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    else:
        preprocessor = MidasDetector.from_pretrained("lllyasviel/Annotators")

    generator = torch.manual_seed(128)

    scene_prompts = scene_prompt.split("\n")

    prompts = []
    for prompt in scene_prompts:
        print(prompt)
        if human:
            prompts.append(character_prompt + ", full body, " + prompt)
        else:
            prompts.append(character_prompt + ", " + prompt)

    prompt_num = len(prompts)

    pipe = StableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-7", torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")  # yes, it's a normal LoRA
    # Generate Origin Images
    for i in range(start_idx, prompt_num):
        results = pipe(
            prompt=prompts[i],
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=generator,
        )
        image = results.images[0]
        image.save(path.join(full_dir, f"origin_{i}.png"))
        if human:
            results, success = face_correction(image, path.join(full_dir, f"openpose_{i}.png"), preprocessor)
            if not success:
                openpose_paint(results, path.join(full_dir, f"openpose_{i}.png"))
        else:
            preprocessor(image).save(path.join(full_dir, f"depth_{i}.png"))

    # loading controlnet model and SD ControlNet pipeline
    controlnet = [ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_openpose', torch_dtype=torch.float16)] \
        if human else [
        ControlNetModel.from_pretrained('lllyasviel/control_v11f1p_sd15_depth', torch_dtype=torch.float16)]
    generator = torch.manual_seed(128)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("Lykon/dreamshaper-7", controlnet=controlnet,
                                                             torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")  # yes, it's a normal LoRA

    if start_idx == 0:
        # Concatenate images horizontally
        combined_image = Image.new('RGB', (512 * prompt_num, 512))
        if human:
            for i in range(prompt_num):
                image = Image.open(path.join(full_dir, f'openpose_{i}.png'))
                combined_image.paste(image, (i * 512, 0))

            # Save the result
            combined_image.save(path.join(full_dir, 'combined_openpose.png'))

        else:
            for i in range(prompt_num):
                image = Image.open(path.join(full_dir, f'depth_{i}.png'))
                combined_image.paste(image, (i * 512, 0))

            # Save the result
            combined_image.save(path.join(full_dir, 'combined_depth.png'))

        ''' Consistent Character Generation '''

        control_image = [Image.open(path.join(full_dir, 'combined_openpose.png'))] \
            if human else [Image.open(path.join(full_dir, 'combined_depth.png'))]

        results = pipe(
            prompt=character_prompt,
            num_inference_steps=4,
            guidance_scale=0.3,
            image=control_image,
            height=512,
            width=prompt_num * 512,
            generator=generator,
            controlnet_conditioning_scale=[0.5]
        )
        image = results.images[0]
        image.save(path.join(full_dir, 'output_consistency.png'))
        character_images = [crop_image(image, i) for i in range(prompt_num)]
    else:
        combined_image = Image.open(path.join(full_dir, 'combined_openpose.png')) if human \
            else Image.open(path.join(full_dir, 'combined_depth.png'))
        width, _ = combined_image.size
        character_images = [None] * start_idx
        for i in range(start_idx, prompt_num):
            if human:
                combined_image.paste(Image.open(path.join(full_dir, f"openpose_{i}.png")), (width - 512, 0))
            else:
                combined_image.paste(Image.open(path.join(full_dir, f"depth_{i}.png")), (width - 512, 0))
            results = pipe(
                prompt=character_prompt,
                num_inference_steps=4,
                guidance_scale=0.3,
                image=[combined_image],
                height=512,
                width=width,
                generator=generator,
                controlnet_conditioning_scale=[0.5]
            )
            results.images[0].save(path.join(full_dir, f'output_consistency_for{i}.png'))
            character_images.append(crop_image(results.images[0], width // 512 - 1))

    # background reform
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    ).to("cuda")

    outputs = [Image.open(path.join(full_dir, f"output_{i}.png")) for i in range(start_idx)]
    # combined_image = Image.new('RGB', (512 * prompt_num, 512))
    for i in range(start_idx, prompt_num):
        mask_image = background_remove(character_images[i])
        bounding_box = find_bounding_box(mask_image)
        print(bounding_box)
        mask_image = mask_outside_bounding_box(bounding_box, ImageOps.invert(mask_image))

        # this "i" could be changed
        image_origin = Image.open(path.join(full_dir, f"origin_{i}.png"))
        image_merged = merge_background(bounding_box, image_origin, character_images[i])

        # image and mask_image should be PIL images.
        # The mask structure is white for inpainting and black for keeping as is
        image_out = pipe(prompt=prompts[i], image=image_merged, mask_image=mask_image, generator=generator).images[0]
        image_out.save(path.join(full_dir, f"output_{i}.png"))
        # combined_image.paste(image_out, (i * 512, 0))
        outputs.append(image_out)

    with open(path.join(full_dir, "count.txt"), 'w') as file:
        file.write(str(prompt_num))
    return outputs


def run_v2(character_prompt: str, scene_prompt: str, sub_dir: str, human: bool, face_length: float, face_width: float,
           eyebrow_height: float, eye_height: float, nose_size: float, mouth_size: float):
    full_dir = path.join("outputs", sub_dir)

    if not path.exists(full_dir):
        os.mkdir(full_dir)

    start_idx = 0
    if path.exists(path.join(full_dir, "count.txt")):
        with open(path.join(full_dir, "count.txt"), 'r') as file:
            content = file.read().strip()
            start_idx = int(content)

    ### record prompt ###
    with open(path.join(full_dir, f"prompt.txt"), "w") as file:
        file.write(character_prompt)
        file.write("\n")
        file.write(scene_prompt)

    if human:
        preprocessor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    else:
        preprocessor = MidasDetector.from_pretrained("lllyasviel/Annotators")

    generator = torch.manual_seed(128)

    scene_prompts = scene_prompt.split("\n")

    prompts = []
    for prompt in scene_prompts:
        print(prompt)
        if human:
            prompts.append(character_prompt + ", full body, " + prompt)
        else:
            prompts.append(character_prompt + ", " + prompt)

    prompt_num = len(prompts)

    pipe = StableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-7", torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")  # yes, it's a normal LoRA

    scale_info = [None] * start_idx
    # Generate Origin Images
    for i in range(start_idx, prompt_num):
        results = pipe(
            prompt=prompts[i],
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=generator,
        )
        image = results.images[0]
        image.save(path.join(full_dir, f"origin_{i}.png"))
        if human:
            result_points, face_points \
                = face_correction(image, path.join(full_dir, f"openpose_{i}.png"), preprocessor, points_only=True)
            face = [Keypoint(p[0], p[1]) for p in face_points] if face_points is not None else result_points[0].face
            scale, min_x, min_y \
                = scale_up_openpose([PoseResult(result_points[0].body,
                                                result_points[0].left_hand,
                                                result_points[0].right_hand,
                                                face)],
                                    2, path.join(full_dir, f"openpose_{i}.png"))
            if scale == -1:
                openpose_paint(result_points, path.join(full_dir, f"openpose_{i}.png"))
            scale_info.append((scale, min_x, min_y))
        else:
            preprocessor(image).save(path.join(full_dir, f"depth_{i}.png"))

    # loading controlnet model and SD ControlNet pipeline
    controlnet = [ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_openpose', torch_dtype=torch.float16)] \
        if human else [
        ControlNetModel.from_pretrained('lllyasviel/control_v11f1p_sd15_depth', torch_dtype=torch.float16)]
    generator = torch.manual_seed(128)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("Lykon/dreamshaper-7", controlnet=controlnet,
                                                             torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")  # yes, it's a normal LoRA

    if start_idx == 0:
        # Concatenate images horizontally
        combined_image = Image.new('RGB', (512 * prompt_num, 512))
        if human:
            for i in range(prompt_num):
                image = Image.open(path.join(full_dir, f'openpose_{i}.png'))
                combined_image.paste(image, (i * 512, 0))

            # Save the result
            combined_image.save(path.join(full_dir, 'combined_openpose.png'))

        else:
            for i in range(prompt_num):
                image = Image.open(path.join(full_dir, f'depth_{i}.png'))
                combined_image.paste(image, (i * 512, 0))

            # Save the result
            combined_image.save(path.join(full_dir, 'combined_depth.png'))

        ''' Consistent Character Generation '''

        control_image = [Image.open(path.join(full_dir, 'combined_openpose.png'))] \
            if human else [Image.open(path.join(full_dir, 'combined_depth.png'))]

        results = pipe(
            prompt=character_prompt,
            num_inference_steps=4,
            guidance_scale=0.3,
            image=control_image,
            height=512,
            width=prompt_num * 512,
            generator=generator,
            controlnet_conditioning_scale=[0.5]
        )
        image = results.images[0]
        image.save(path.join(full_dir, 'output_consistency.png'))
        character_images = [crop_image(image, i) for i in range(prompt_num)]
    else:
        combined_image = Image.open(path.join(full_dir, 'combined_openpose.png')) if human \
            else Image.open(path.join(full_dir, 'combined_depth.png'))
        width, _ = combined_image.size
        character_images = [None] * start_idx
        for i in range(start_idx, prompt_num):
            if human:
                combined_image.paste(Image.open(path.join(full_dir, f"openpose_{i}.png")), (width - 512, 0))
            else:
                combined_image.paste(Image.open(path.join(full_dir, f"depth_{i}.png")), (width - 512, 0))
            results = pipe(
                prompt=character_prompt,
                num_inference_steps=4,
                guidance_scale=0.3,
                image=[combined_image],
                height=512,
                width=width,
                generator=generator,
                controlnet_conditioning_scale=[0.5]
            )
            results.images[0].save(path.join(full_dir, f'output_consistency_for{i}.png'))
            character_images.append(crop_image(results.images[0], width // 512 - 1))

    if human:
        for i in range(start_idx, prompt_num):
            assert scale_info[i] is not None
            if scale_info[i][0] == -1:
                continue
            character_images[i] = scale_down_character(character_images[i], *scale_info[i])

    # background reform
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    ).to("cuda")

    outputs = [Image.open(path.join(full_dir, f"output_{i}.png")) for i in range(start_idx)]
    # combined_image = Image.new('RGB', (512 * prompt_num, 512))
    for i in range(start_idx, prompt_num):
        image_origin = Image.open(path.join(full_dir, f"origin_{i}.png"))

        mask_image = background_remove(character_images[i])
        bounding_box1 = find_bounding_box(mask_image)
        bounding_box2 = find_bounding_box(background_remove(image_origin))
        bounding_box = merge_bounding_boxes(bounding_box1, bounding_box2)
        print(bounding_box)
        mask_image = mask_outside_bounding_box(bounding_box, ImageOps.invert(mask_image))

        image_merged = merge_background(bounding_box, image_origin, character_images[i])

        # image and mask_image should be PIL images.
        # The mask structure is white for inpainting and black for keeping as is
        image_out = pipe(prompt=prompts[i], image=image_merged, mask_image=mask_image, generator=generator).images[0]
        if human:
            face_blend(image_src=image_merged, image_out=image_out, preprocessor=preprocessor)

        image_out.save(path.join(full_dir, f"output_{i}.png"))
        # combined_image.paste(image_out, (i * 512, 0))
        outputs.append(image_out)

    with open(path.join(full_dir, "count.txt"), 'w') as file:
        file.write(str(prompt_num))
    return outputs


# demo = gr.Interface(
#     fn=run_v2,
#     # inputs=["text", "checkbox", gr.Slider(0, 100)],
#     inputs=[
#         "text",
#         gr.Textbox(lines=10, placeholder="hhh..."),
#         "text",
#         "checkbox",
#     ],
#     outputs=gr.Gallery(label="Generated Images")  # ["image"]
# )
# demo.launch()

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                input_text1 = gr.Textbox(label="Character Prompt", placeholder="a strong man with red T-shirt")
            with gr.Row():
                input_text2 = gr.Textbox(label="Scene Prompts", lines=10, placeholder="running in forest \njumping "
                                                                                      "in park \nstand in swimming "
                                                                                      "pool")
            with gr.Row():
                input_text3 = gr.Textbox(label="Image Save Directory")
            with gr.Row():
                input_checkbox = gr.Checkbox(label="Human Character")

            with gr.Accordion("Optional Settings", open=False):
                slider1 = gr.Slider(label="Face Length", minimum=0.25, maximum=1, step=0.01, value=0.5)
                slider2 = gr.Slider(label="Face Width", minimum=0.25, maximum=1, step=0.01, value=0.5)
                slider3 = gr.Slider(label="Eyebrow Height", minimum=0.25, maximum=1, step=0.01, value=0.5)
                slider4 = gr.Slider(label="Eye Height", minimum=0.25, maximum=1, step=0.01, value=0.5)
                slider5 = gr.Slider(label="Nose Size", minimum=0.25, maximum=1, step=0.01, value=0.5)
                slider6 = gr.Slider(label="Mouth Size", minimum=0.25, maximum=1, step=0.01, value=0.5)

            submit_button = gr.Button("Submit")

        with gr.Column(scale=2):
            output_gallery = gr.Gallery(label="Generated Images")

    submit_button.click(
        run_v2,
        inputs=[input_text1, input_text2, input_text3, input_checkbox, slider1, slider2, slider3, slider4],
        outputs=output_gallery
    )

demo.launch()
