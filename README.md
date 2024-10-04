<h1 align="center" id="title">Place an object’s image in a text-conditioned scene</h1>

<p id="description">This project focuses on using generative AI to create realistic product photographs by seamlessly integrating an object image (with a white background) into a text-generated scene. The goal is to place the object naturally into various contexts preserving its details while ensuring that the scene aligns with aspects like lighting spatial positioning and realism.</p>

<h2>Project Screenshots:</h2>
<img src="https://drive.google.com/uc?id=1aPS18uvwNnloKzzXmBKE-VmB-OP_nJM7" alt="project-screenshot" width="400" height="400" />
<img src="https://drive.google.com/uc?id=15dVKM5kelXuzxSCZmXXMy47vtX48CmDS" alt="project-screenshot" width="400" height="400" />


[Watch the Video](https://github.com/SahilGoyal098/Avaatar-asignment/blob/master/output_video_3oct_evening.mp4)

<h2>🧐 Features</h2>

Here're some of the project's best features:

*   Realistic Background Generation: Utilizes advanced generative AI (Stable Diffusion Inpainting) to create visually appealing and contextually relevant backgrounds based on text prompts enhancing the overall scene realism.
*   Efficient Object Masking: Implements refined mask creation to accurately separate the product from its background preserving details and ensuring the object remains unaltered in the final output.
*   Dynamic Resizing with Controlled Placement: Supports dynamic resizing of the object while maintaining its center positioning allowing for smooth transitions and natural-looking placements in various scenes.
*   Seamless Integration into Videos: Generates a sequence of frames that create smooth transitions and coherent storytelling in video output enhancing user engagement.

<h2>Flow of the Code</h2>

<img src="https://github.com/user-attachments/assets/2d0656c3-71f5-4fe3-812e-49b6aa8887ac" alt="project-screenshot" width="400" height="400"/>


1. **Start**  
   - Decision: *Do you want to generate a single image or a video?*

### For **Image** Generation:

2. **Input Image Path**  
   - User inputs the path of the image file.

3. **Input Prompt for Inpainting**  
   - User provides a prompt describing what to inpaint.

4. **Generate Object Mask**  
   - The code creates a mask by detecting the object in the image.

5. **Resize and Center Images**  
   - The object is resized and centered on a blank white background.

6. **Inpaint Image**  
   - Using the inpainting pipeline, the image is modified based on the mask and prompt.

7. **Save Inpainted Image**  
   - The inpainted image is saved to disk.

### For **Video** Generation:

2. **Input Image Path**  
   - User inputs the path of the image file.

3. **Input Video Prompts**  
   - User enters multiple prompts for video frame generation.

4. **Generate Object Mask**  
   - The code creates a mask by detecting the object in the image.

5. **Resize and Center Images (Dynamic)**  
   - The object is dynamically resized and centered at different scales.

6. **Inpaint Image**  
   - Inpainting occurs on dynamically resized frames.

7. **Repeat Frames**  
   - Each inpainted frame is repeated multiple times for a slower video effect.

8. **Save Video**  
   - The generated frames are compiled into a video file and saved.

**End**

<h2>🛠️ Installation Steps:</h2>

<p>1. Step 1: Install Python</p>

<p>2. Step 2: Set Up a Virtual Environment (Optional but Recommended)</p>

```
python -m venv myenv
```

<p>3. Step 3: Install Required Libraries</p>

```
pip install opencv-python Pillow numpy diffusers torch torchvision
```

<p>4. Step 4: Ensure CUDA is Installed (Optional for GPU Acceleration)</p>

```
nvcc --version
```

<p>5. Step 5: Download the Model Checkpoints (if required)</p>

<h2>💻 Built with</h2>

Technologies used in the project:

*   Python: The primary programming language used for writing the script.
*   OpenCV: Used for image processing tasks like resizing grayscale conversion thresholding blurring and morphological operations.
*   Pillow (PIL): Used for image manipulation converting images between different formats and handling image file input/output.
*   Diffusers (Stable Diffusion Inpainting): A deep learning library from Hugging Face used for the inpainting process specifically leveraging the Stable Diffusion Inpainting model to fill in areas of the image based on a mask and text prompt.
*   Torch (PyTorch): A deep learning framework used for model inference and interacting with the Stable Diffusion Inpainting model particularly handling tensors and operations on the GPU (CUDA).
