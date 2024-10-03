import cv2
from PIL import Image
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import torch

# Input Should be image with white background and one object in it




# Function to create a refined mask
def create_object_mask(image):
    image_array = np.array(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY_INV)
    binary_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    inverted_mask = cv2.bitwise_not(binary_mask)
    return Image.fromarray(inverted_mask)

# Resizing and centering images based on scale factor
def resize_and_center_images(init_image, mask_image, scale_factor=0.75):
    item_img = np.array(init_image) if isinstance(init_image, Image.Image) else init_image
    mask_img = np.array(mask_image) if isinstance(mask_image, Image.Image) else mask_image

    if item_img is None or mask_img is None:
        raise ValueError("One or both image arrays are None.")

    background_size = item_img.shape[:2]
    new_size = (int(item_img.shape[1] * scale_factor), int(item_img.shape[0] * scale_factor))
    resized_item = cv2.resize(item_img, new_size)
    resized_mask = cv2.resize(mask_img, new_size)

    # Create blank white backgrounds for both item and mask
    background = np.ones_like(item_img) * 255
    background_mask = np.ones_like(mask_img) * 255

    # Calculate the position to place the resized item (center it)
    start_x = (background.shape[1] - resized_item.shape[1]) // 2
    start_y = (background.shape[0] - resized_item.shape[0]) // 2

    # Place the resized item and mask on the white background
    background[start_y:start_y + resized_item.shape[0], start_x:start_x + resized_item.shape[1]] = resized_item
    background_mask[start_y:start_y + resized_mask.shape[0], start_x:start_x + resized_mask.shape[1]] = resized_mask

    return background, background_mask

# Function to inpaint the image based on a mask and prompt
def inpaint_image(image, mask, prompt):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32
    ).to("cuda")
    
    generated_image = pipe(prompt=prompt, image=image, mask_image=mask, num_inference_steps=50).images[0]
    return generated_image

# Function to generate frames for video with dynamic resizing and background changes
def generate_frames_for_video(video_prompts, init_image, steps=10, scale_factor=0.98, interval=3, repeat_frames=5):
    frames = []
    mask_image = create_object_mask(init_image)

    for prompt in video_prompts:
        for i in range(steps):
            # Create blank white background of the same size as the original image
            blank_background = np.ones_like(np.array(init_image)) * 255

            # Adjust size only on specific intervals (e.g., every `interval` frames)
            if i % interval == 0:
                # Calculate the new size based on the scale factor
                resized_image, resized_mask = resize_and_center_images(init_image, mask_image, scale_factor ** i)
                # Inpaint only on size-change frames
                new_background = inpaint_image(Image.fromarray(resized_image), Image.fromarray(resized_mask), prompt)
            else:
                # Reuse the previous frame if not resizing
                new_background = frames[-1] if frames else Image.fromarray(np.array(init_image))

            # Convert new background to format for video
            combined_frame = cv2.cvtColor(np.array(new_background), cv2.COLOR_RGB2BGR)
            
            # Repeat each frame 'repeat_frames' times to make the video slower
            for _ in range(repeat_frames):
                frames.append(combined_frame)

    return frames

# Function to save frames as video (adjusted frame rate for slower video)
def save_frames_to_video(frames, output_path, fps=5):
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        video.write(frame)
    video.release()

# Main function to allow user interaction for generating image or video
def main():
    choice = input("Do you want to generate a single image (enter 'image') or a video (enter 'video')? ").strip().lower()
    
    if choice == 'image':
        image_path = input("Enter the path of the image: ")
        prompt = input("Enter the prompt for inpainting: ")
        
        init_image = Image.open(image_path).convert("RGB")
        mask_image = create_object_mask(init_image)
        inpainted_image = inpaint_image(init_image, mask_image, prompt)
        inpainted_image = inpainted_image.resize(init_image.size, Image.LANCZOS)
        
        # Save the inpainted image
        output_image_path = "inpainted_image.png"
        inpainted_image.save(output_image_path)
        print(f"Inpainted image saved successfully at: {output_image_path}")

    elif choice == 'video':
        image_path = input("Enter the path of the image: ")
        init_image = Image.open(image_path).convert("RGB")
        
        # Get multiple prompts for video generation
        video_prompts = []
        while True:
            prompt = input("Enter a prompt for the video (or 'done' to finish): ")
            if prompt.lower() == 'done':
                break
            video_prompts.append(prompt)

        frames = generate_frames_for_video(video_prompts, init_image, steps=10, scale_factor=0.98, interval=3, repeat_frames=5)

        # Save the video
        output_video_path = "generated_video.mp4"
        save_frames_to_video(frames, output_video_path, fps=5)
        print(f"Video saved successfully at: {output_video_path}")

    else:
        print("Invalid choice. Please enter 'image' or 'video'.")

# Example usage
if __name__ == "__main__":
    main()


