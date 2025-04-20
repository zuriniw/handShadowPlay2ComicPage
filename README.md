# HandShadow2ComicPanel

*A Character-Aware Hand Shadow Play with Lasting + Printable Narratives*

## Setup Instructions

**Step 1:** Install the things the code needs.

Open your computer's command line and run:
`pip install -r requirements.txt`

**Step 2:** Set up StreamDiffusion.

Follow the instructions in this file: `streamdiffusion\README.md`

**Step 3:** Make sure your webcam is plugged in.

**Step 4:** Run the first part of the code.

Open your computer's command line, go to the `src` folder, and run:
`python main.py`

You will see three new windows pop up. These show what your camera sees and how the code is getting ready to make the video.

**Step 5:** Run the second part of the code.

Open another command line, go to the folder where `test_connect_ani.py` is, and run it. You will see a new window called "image viewer." This shows the video that is being created.

**Step 6:** Get the materials for the comic page.

To get the comic page, you will need:

* The `json` file in this folder. You can feed it into LLM to get the captions.
* The pictures that were saved in the `screen/img` folder inside the `streamdiffusion` folder. These pictures are the "keypanels."





## Description

### Concept

Traditional hand shadow play is a cherished childhood memory, but it comes with inherent limitations. First, creative play can be difficult for children, as it’s hard to maintain precise hand shapes to represent specific animals—especially when performing various actions. Second, the experience is fleeting; memories of the play often fade quickly.

This project reimagines hand shadow play as an engaging, collaborative, and easy-to-perform storytelling system that generates a printable comic page documenting the storylines formed during play. It detects hand gestures, classifies animal characters, and generates vivid six-panel comics to reflect the interaction.

### Input System

- MediaPipe is used for motion capture and joint recognition, implemented via a TouchDesigner integration by Torin Blankenship.

### Output System

- StreamDiffusion handles real-time image generation using captured base maps and updated prompts. Integrated via the plugin by dotsimulate.

### Image-to-Image Models

- Base model: `stabilityai/sd-turbo`
- LoRA: `SDXL_CrayonPainting 2.0_2.0`
- LoRA weight: `0.8`

### LLM for Captioning

- GPT-4o is used to generate captions for each comic panel, based on the character sequences and keyframes.

## Strategies

### Strategy 1 | Define and Log Keyframes

Four keyframe types:

- Character Add
- Character Quit
- Characters Are Far Enough
- Characters Are Close Enough

Each keyframe is logged in a JSON file (`keyframes.json`) with:

- `timestamp`
- `name`
- `current_characters`
- `all_characters`

### Strategy 2 | Responsive Prompt

Each new keyframe triggers a prompt formatted based on the scene (e.g., character identities and positions), sent via socket and appended to the base prompt for image generation.

### Strategy 3 | Palette

Background colors are varied per keyframe. Characters like rabbits and spiders are assigned consistent colors from a curated palette to ensure distinction and harmony with the LoRA’s aesthetic.

## Reflection

### What Worked and Main Limitation

- Fast and responsive interaction
- High-quality stylized outputs using LoRA
- Clear narrative structure via keyframe logging
- Currently only supports rabbit and spider gestures
- Style and diversity are limited by the LoRA model

## Result and Impact

- A real-time, collaborative, and playful system that transforms hand shadows into vivid animal characters
- A method to generate printable six-panel comic pages reflecting the user’s hand shadow performance

This project bridges physical interaction and generative AI to create a new form of real-time storytelling. It supports embodied play, tangible creativity, and collaborative co-creation. Its educational and artistic potential spans gesture-based learning, performance, and STEAM instruction.

## Tech & Tools

- (legacy method for segmentation) ITAP Robotica Medica. Lightweight Hand Segmentation, GitHub Repository, 2024. [Online]. Available: https://github.com/itap-robotica-medica/lightweight-hand-segmentation/blob/main/hand_segmentation.py

- Google. MediaPipe: Cross-platform, customizable ML solutions for live and streaming media, 2024. [Online]. Available: https://mediapipe.dev/

- Cumulo-autumn. StreamDiffusion: Real-time Diffusion Image Generation Pipeline, GitHub Repository, 2024. [Online]. Available: https://github.com/cumulo-autumn/StreamDiffusion

- Stability AI. SD-Turbo: High-speed Text-to-Image Generation Model, 2024. [Online]. Available: https://stability.ai/

- Civitai. SDXL_CrayonPainting 2.0_2.0: Crayon-style Fine-tuned LoRA for SDXL, 2024. [Online]. Available: https://civitai.com/models/439493/crayonpaiting-sdxl-20-or


