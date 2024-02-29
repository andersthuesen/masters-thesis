# Masters thesis - Project plan

## Project title

3D human interaction synthesis for action recognition data augmentation

## Author

Anders Bredgaard Thuesen (andersbthuesen[that weird "a" here]gmail.com)

## Project period

January 29th - June 29th (~5 Months)

## Supervisors

DTU: Morten Rieger Hannemose (mohan[that weird "a" here]dtu.dk)

Teton.ai: Frederik Warburg (frederik[that weird "a" here]teton.ai)

## Introduction

Care homes and hospitals are currently grappling with significant challenges in providing high-quality care due to a persistent shortage of healthcare professionals. This shortage, fueled by an aging population's increasing demand for healthcare services and a declining number of new entrants into the healthcare profession, is straining these institutions. Furthermore, as the demand for documentation in healthcare is growing, professionals have less time for patient interaction limiting face-to-face engagement and the ability to address patients' needs effectively.

Addressing this critical issue requires innovative solutions and a concerted effort to improve working conditions and attract more individuals to the healthcare profession.

Teton.ai is enhancing the capabilities of healthcare professionals by equipping them with advanced tools for improved patient monitoring. The company achieves this by installing intelligent camera sensors in the hospital rooms. These sensors feature on-device processing, enabling the system to alert staff about patient falls promptly, all while ensuring patient privacy is not compromised.

It is an open question whether it is possible to automate some of the documentation burden using deep learning and computer vision, requiring vast amounts of training data which is often inaccessible and expensive to label. One posed solution is to first construct a simulator of the environment enabling the sampling of synthetic data, ideally capturing the ditribution of real-life hospital and care home scenarios.

In recent years generative models like Variational Auto Encoders (VAEs), Generative Adverserial Networks (GANs) and Denoising Diffusion Models (DDMs) have shown incredible results in generating highly realistic images and videos from text. One might ask the question whether these models could enable the learning of this kind of simulator. However, due to the high dimensionality of images and videos on might need large amounts of data to train such models, defeating its own purpose. Instead, representing the humans in the scene explicitly using e.g. the SMPL model combined with realistic computer rendering could provide a feasible alternative.

## Plan

- **February:** Reading papers, data exploration, filtering, smoothing and visualisation and general preparation
- **March:** Training diffusion model and condition on action
- **April:** Condition diffusion model on the scene
- **May & June**: Write thesis

## Engineering plan

- Generate dataset by filtering pseudo ground truth on different criteria

  - Movement in scene
  - Multiple people preset (classified as patient and staff)

- Train an unconditional diffusion model on pseudo ground truth 3D SMPL poses
  - Maybe include public data
- Condition diffusion model on actions using classifier free guidance
- Condition diffusion model on scene
  - Maybe using pretrained depth anything model
- Render poses in novel scene reconstructions

## Other ideas / obs

### Initial pose

Initial pose could have any rotation. What is a good way to parameterise this?
Subsequent changes in pose is relative to the viewing/forward direction?

### How to handle out-of-view poses?

Poses might dynamically come in and out of the scene. Might just assume that poses are always just in scene.

### Noisy pseudo labels

Probably have to do some smoothing

## Links

https://khanhha.github.io/posts/SMPL-model-introduction/
http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf
https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf
