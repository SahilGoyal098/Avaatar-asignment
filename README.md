<h1 align="center" id="title">Place an object’s image in a text-conditioned scene</h1>

<p id="description">This project focuses on using generative AI to create realistic product photographs by seamlessly integrating an object image (with a white background) into a text-generated scene. The goal is to place the object naturally into various contexts preserving its details while ensuring that the scene aligns with aspects like lighting spatial positioning and realism.</p>

<h2>Project Screenshots:</h2>

<img src="https://drive.google.com/file/d/1aPS18uvwNnloKzzXmBKE-VmB-OP_nJM7/preview" width="640" height="480"/>

  
  
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   Realistic Background Generation: Utilizes advanced generative AI (Stable Diffusion Inpainting) to create visually appealing and contextually relevant backgrounds based on text prompts enhancing the overall scene realism.
*   Efficient Object Masking: Implements refined mask creation to accurately separate the product from its background preserving details and ensuring the object remains unaltered in the final output.
*   Dynamic Resizing with Controlled Placement: Supports dynamic resizing of the object while maintaining its center positioning allowing for smooth transitions and natural-looking placements in various scenes.
*   Seamless Integration into Videos: Generates a sequence of frames that create smooth transitions and coherent storytelling in video output enhancing user engagement.

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
