# Project plan

## Author
Anders Bredgaard Thuesen (andersbthuesen[that weird "a" here]gmail.com)


## Project period
January 29th - June 29th (~5 Months)

## Supervisors
DTU: Morten Rieger Hannemose (mohan[that weird "a" here]dtu.dk)

Teton.ai: Frederik Warburg (frederik[that weird "a" here]teton.ai)


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

###  How to handle out-of-view poses?
Poses might dynamically come in and out of the scene. Might just assume that poses are always just in scene.

### Noisy pseudo labels
Probably have to do some smoothing



## Links
https://khanhha.github.io/posts/SMPL-model-introduction/
http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf

