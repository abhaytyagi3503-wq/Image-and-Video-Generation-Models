
# Image and Video Generation Models: Squirrel Detection, Segmentation, Inpainting, and AI Generation

## Project Overview

This project focuses on using modern computer vision and generative AI models to manipulate real-world images and generate visual media. The main task was to identify images containing a squirrel near a birdfeeder, segment the squirrel region, remove the squirrel using image inpainting, and replace the squirrel with a bird using generative models.

The project also compared a local machine-learning pipeline with online multimodal AI tools. The local pipeline used object detection, segmentation, and Stable Diffusion inpainting models, while the online method used multimodal AI assistants for image editing, text-to-image generation, and video generation.

This project demonstrates practical use of AI/ML models for visual editing, object localization, segmentation, image restoration, object replacement, and prompt-based media generation.

---

## Objectives

- Detect images containing a squirrel near a birdfeeder.
- Create a filtered list of selected image files.
- Segment squirrels from selected images.
- Generate slightly enlarged squirrel masks for inpainting.
- Remove squirrels from images using deep learning-based inpainting.
- Replace squirrels with realistic birds using Stable Diffusion inpainting.
- Compare local model-based image editing with online multimodal AI tools.
- Generate photorealistic bird-at-birdfeeder images using text-only prompts.
- Create an 8-second instructional AI-generated video.

---

## Tools and Technologies Used

- Python 3
- PyTorch
- Hugging Face Transformers
- Hugging Face Diffusers
- OWL-ViT Zero-Shot Object Detection
- Segment Anything Model-style segmentation workflow
- Stable Diffusion XL Inpainting
- PIL / Pillow
- NumPy
- Online multimodal AI assistants
- Text-to-image generation tools
- AI video generation tools

---

## Project Files

| File | Purpose |
|---|---|
| `subselectImages.py` | Detects images containing both a squirrel and a birdfeeder |
| `segmentSquirrels.py` | Generates squirrel segmentation masks |
| `removeSquirrels.py` | Removes squirrels using inpainting |
| `replaceSquirrels.py` | Replaces squirrels with birds using inpainting |
| `selectedImages.txt` | Stores the selected squirrel-at-birdfeeder image paths |
| `imageList.txt` | Contains the original input image list |
| `chatTranscriptLinks.txt` | Stores transcript links for online squirrel-to-bird edits |
| `eraseChatTranscriptLinks.txt` | Stores transcript links for online squirrel removal edits |
| `generationChatTranscriptLinks.txt` | Stores transcript links for text-to-image generation |
| `videoPromptGeneration.txt` | Stores the video generation prompt |
| `myVideo.mp4` | Final generated instructional video |

---

## Methodology and Project Execution

The project was completed using two main methods:

1. A local ML pipeline using task-specific models.
2. Online multimodal AI tools for comparison and generation.

---

## Method 1: Local Vision and Generative AI Pipeline

### 1. Image Subselection

The first step was to identify images that contain both a squirrel and a birdfeeder. The program `subselectImages.py` reads a text file containing image paths and writes only the selected images to `selectedImages.txt`.

The script uses OWL-ViT zero-shot object detection with prompts such as:

```text
a squirrel
a gray squirrel
a brown squirrel
a squirrel on a feeder
a bird feeder
a hanging bird feeder
a backyard bird feeder
a feeder full of seeds
````

The detection logic checks for both squirrel-like and birdfeeder-like objects. It then scores the image using confidence values and spatial proximity between the squirrel and feeder bounding boxes. Images are selected only when the squirrel and feeder are both detected close enough to each other. The code uses OWL-ViT to detect squirrel and birdfeeder prompts and keeps images whose pair score crosses the selection threshold. 

### Command

```bash
python subselectImages.py imageList.txt selectedImages.txt
```

### Output

```text
selectedImages.txt
```

---

### 2. Squirrel Segmentation

After selecting the correct images, the next step was to generate a mask around the squirrel. The script `segmentSquirrels.py` detects the squirrel and creates a segmentation mask.

For each input image:

```text
imageName.jpg
```

the output mask is saved as:

```text
imageName-sqMask.png
```

The mask was intentionally enlarged so that squirrel fur, tail edges, and boundary pixels are included. This is important because a mask that is too tight can leave visible squirrel artifacts after inpainting. The segmentation script detects the squirrel box, expands it using padding, applies SAM-based masking, and then dilates and smooths the mask before saving it. 

### Command

```bash
python segmentSquirrels.py selectedImages.txt
```

---

### 3. Squirrel Removal Using Inpainting

The script `removeSquirrels.py` loads each selected image and its corresponding squirrel mask. It then uses a Stable Diffusion XL inpainting pipeline to remove the squirrel and fill the masked region with a natural-looking background.

The model used was:

```text
diffusers/stable-diffusion-xl-1.0-inpainting-0.1
```

The prompt was designed to preserve the original outdoor birdfeeder scene:

```text
natural outdoor wildlife photo of a birdfeeder and surrounding branches,
no squirrel, photorealistic, seamless background, realistic lighting, high detail
```

A negative prompt was also used to avoid squirrel-like or unrealistic outputs:

```text
squirrel, rodent, extra animals, blurry, painting, cartoon, distorted, artifacts, duplicate
```

The removal script loads the image and corresponding mask, applies the inpainting model, and saves the output as `imageName-squirrelRemoved.jpg`. 

### Command

```bash
python removeSquirrels.py selectedImages.txt
```

### Output

```text
imageName-squirrelRemoved.jpg
```

---

### 4. Replacing Squirrels with Birds

The script `replaceSquirrels.py` uses the same selected images and masks, but instead of removing the squirrel, it replaces the masked region with a realistic bird.

The model used was:

```text
diffusers/stable-diffusion-xl-1.0-inpainting-0.1
```

The replacement prompt was:

```text
photorealistic small songbird perched naturally next to the birdfeeder,
realistic feathers, matching camera angle, matching daylight, wildlife photography
```

The negative prompt was:

```text
squirrel, rodent, cartoon, painting, multiple birds, deformed bird, blurry, artifacts, unrealistic
```

The script saves the final edited image as:

```text
imageName-squirrelReplaced.jpg
```

The replacement script uses Stable Diffusion XL inpainting with a bird-focused prompt and saves the generated output as `imageName-squirrelReplaced.jpg`. 

### Command

```bash
python replaceSquirrels.py selectedImages.txt
```

---

## Method 2: Online Multimodal AI Assistants

The second method used online AI assistants to perform similar image editing tasks. At least three selected images were uploaded to an online assistant with prompts to either remove the squirrel or replace it with a bird.

The assignment required:

* At least 3 images where squirrels were replaced by birds.
* At least 3 images where squirrels were removed.
* At least 3 text-only generated images of birds eating at birdfeeders.
* Chat transcript links for the online editing and generation steps.
* An 8-second AI-generated instructional video.

The required transcript files were:

```text
chatTranscriptLinks.txt
eraseChatTranscriptLinks.txt
generationChatTranscriptLinks.txt
videoPromptGeneration.txt
```

The assignment specifically required these transcript-link files and online AI outputs as part of the final submission. 

---

## Video Generation

For the video generation task, an 8-second instructional video prompt was created for a cinematic cooking video. The selected concept was an avocado toast preparation sequence.

The prompt specified:

* 8-second real-time video
* 9:16 vertical format
* normal-speed pacing
* macro close-up shots
* soft daylight
* modern kitchen setting
* cinematic instructional style

The video prompt described step-by-step visuals such as placing bread in a toaster, toast popping up, cutting avocado, mashing avocado, adding salt and lemon, spreading avocado on toast, and final plating. 

---

## Pipeline Workflow

```text
Input Image List
        ↓
Object Detection with OWL-ViT
        ↓
Select Images with Squirrel + Birdfeeder
        ↓
Save selectedImages.txt
        ↓
Detect Squirrel Region
        ↓
Generate and Expand Squirrel Mask
        ↓
Inpaint Masked Region
        ↓
Remove Squirrel OR Replace Squirrel with Bird
        ↓
Save Final Edited Images
        ↓
Compare with Online AI-Generated Outputs
```

---

## How to Run the Project

### 1. Install Required Packages

```bash
pip install torch pillow numpy transformers diffusers accelerate safetensors
```

Depending on your environment, additional dependencies may be needed for model loading and GPU acceleration.

---

### 2. Run Image Selection

```bash
python subselectImages.py imageList.txt selectedImages.txt
```

This creates a list of images that contain a squirrel near a birdfeeder.

---

### 3. Generate Squirrel Masks

```bash
python segmentSquirrels.py selectedImages.txt
```

This creates mask files such as:

```text
imageName-sqMask.png
```

---

### 4. Remove Squirrels

```bash
python removeSquirrels.py selectedImages.txt
```

This creates edited images such as:

```text
imageName-squirrelRemoved.jpg
```

---

### 5. Replace Squirrels with Birds

```bash
python replaceSquirrels.py selectedImages.txt
```

This creates edited images such as:

```text
imageName-squirrelReplaced.jpg
```

---

## Results and Outcomes

The project successfully implemented a complete image-editing pipeline using computer vision and generative AI.

Major outcomes included:

* Built an object-detection pipeline to identify squirrel-at-birdfeeder images.
* Generated `selectedImages.txt` containing filtered image paths.
* Created squirrel segmentation masks using object detection and segmentation.
* Enlarged masks to reduce leftover squirrel artifacts.
* Removed squirrels from selected images using Stable Diffusion XL inpainting.
* Replaced squirrels with realistic birds using prompt-guided inpainting.
* Compared local model-based outputs with online multimodal AI outputs.
* Generated photorealistic birds-at-birdfeeders images using text-only prompts.
* Created an 8-second AI-generated instructional video.

---

## Key Features

* Zero-shot object detection
* Squirrel and birdfeeder image filtering
* Segmentation mask generation
* Mask dilation and smoothing
* Stable Diffusion XL inpainting
* Object removal
* Object replacement
* Prompt engineering
* Online AI comparison
* Text-to-image generation
* AI video generation

---

## Output Files

| Output File              | Description                                        |
| ------------------------ | -------------------------------------------------- |
| `selectedImages.txt`     | Selected images containing squirrel and birdfeeder |
| `*-sqMask.png`           | Squirrel segmentation masks                        |
| `*-squirrelRemoved.jpg`  | Images with squirrel removed                       |
| `*-squirrelReplaced.jpg` | Images with squirrel replaced by bird              |
| `onlineModedN.jpg`       | Online AI squirrel-to-bird edited images           |
| `onlineErasedN.jpg`      | Online AI squirrel removal images                  |
| `onlineGeneratedN.jpg`   | Text-to-image bird-at-birdfeeder images            |
| `myVideo.mp4`            | AI-generated instructional video                   |

---

## Prompt Engineering Strategy

### Squirrel Removal Prompt

```text
natural outdoor wildlife photo of a birdfeeder and surrounding branches,
no squirrel, photorealistic, seamless background, realistic lighting, high detail
```

### Squirrel Replacement Prompt

```text
photorealistic small songbird perched naturally next to the birdfeeder,
realistic feathers, matching camera angle, matching daylight, wildlife photography
```

### Negative Prompt

```text
squirrel, rodent, cartoon, painting, multiple birds, deformed, blurry, artifacts, unrealistic
```

Prompt quality was important because final outputs were judged based on realism, background consistency, and how well the edited area blended with the original scene.

---

## Challenges Faced

### 1. Selecting the Correct Images

Some images contained only birds, only squirrels, or birdfeeders without squirrels. To improve selection, the detection script used both object confidence and spatial relationship scoring.

### 2. Creating Accurate Masks

A tight squirrel mask can leave fur or tail artifacts. To solve this, the mask was expanded and smoothed before inpainting.

### 3. Preserving Background Realism

Inpainting sometimes changed the birdfeeder or surrounding background. Neutral prompts and negative prompts were used to keep the scene realistic.

### 4. Replacing the Squirrel with a Believable Bird

Bird replacement required careful prompting so that the bird matched the original lighting, camera angle, scale, and scene context.

---

## Repository Structure

```text
.
├── README.md
├── assign 5.pdf
├── imageList.txt
├── selectedImages.txt
├── subselectImages.py
├── segmentSquirrels.py
├── removeSquirrels.py
├── replaceSquirrels.py
├── chatTranscriptLinks.txt
├── eraseChatTranscriptLinks.txt
├── generationChatTranscriptLinks.txt
├── videoPromptGeneration.txt
├── myVideo.mp4
└── outputs/
    ├── masks/
    ├── squirrel_removed/
    ├── squirrel_replaced/
    ├── online_modified/
    ├── online_erased/
    └── online_generated/
```

---

## Major Learnings

Through this project, I gained practical experience with:

* Applied image generation models
* Object detection using zero-shot vision-language models
* Segmentation-based image editing
* Stable Diffusion inpainting
* Hugging Face model integration
* Prompt engineering for realistic image manipulation
* Image masking and mask expansion
* Local AI pipelines vs online multimodal AI tools
* Text-to-image generation
* AI video prompt design

---

## Conclusion

This project demonstrated how multiple AI/ML models can be combined to solve a realistic image manipulation problem. The local pipeline used object detection to select squirrel-at-birdfeeder images, segmentation to create masks, and Stable Diffusion XL inpainting to remove or replace squirrels.

The online AI workflow provided a comparison against general-purpose multimodal assistants. Together, both methods showed the strengths and limitations of modern image and video generation systems.

Overall, this project provided hands-on experience in computer vision, generative AI, image editing, segmentation, inpainting, and prompt-based media generation.

```
```
