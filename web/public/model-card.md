# Sightline model card

The model the Sightline page runs: `models/sightline.onnx`, a SightlineNet trained on the blocked
split of drive2018 (run `drive2018_sightlinenet_blocked_s2`). The page also ships its twin,
`models/sightline-shuffled.onnx` (run `drive2018_sightlinenet_leaky_s2`): the same network, recipe
and seed trained on shuffled frames, there only to show what a shuffled test split hides.

Every measured number in this card is read from the project's reports, and a test fails if the card
and the reports disagree. Source code, reports and the README: <https://github.com/safdar-hussain1/sightline>.

## Intended use

- **For** research and teaching: seeing how the way a steering model is tested changes how good it
  looks, and seeing where a network looks when it estimates a steering angle.
- **Never** for steering, assisting or controlling a vehicle, in any form, and never as a safety
  component. The model estimates, from recorded footage, the angle a human applied; it has never
  steered anything.
- **Not** for footage from other cameras, other places or other conditions without testing it there
  first (see [Limitations](#limitations)).

## Model details

| field | value |
|---|---|
| architecture | SightlineNet: five convolutions with batch normalisation and SiLU on PilotNet's convolution geometry, a flatten that keeps left–right position, a small fully connected head with a linear output in degrees |
| parameters | 289,289 |
| input `frames` | uint8, shape (N, 66, 200, 6): two RGB frames stacked along the channels, frame t first and frame t−2 second |
| each frame | the camera's 455 × 256 frame cut to its bottom 150 rows and resized to 200 × 66, RGB, as raw bytes; normalisation is inside the graph |
| output `steering_deg` | float32, shape (N, 1): the steering-wheel angle in degrees, positive clockwise |
| output `saliency` | float32, shape (N, 66, 200): a VisualBackProp map of where the network looked, scaled to [0, 1] per frame stack |
| format | ONNX, opset 17, float32, 1,170,713 bytes; its metadata names the frame lags (`[2]`) and the run |
| runtime on the page | onnxruntime-web, WebAssembly, one thread |

Before it was written, onnxruntime reproduced PyTorch on 848 frame stacks, every clip frame and
self-test input the page uses: the largest angle difference was 1.8e-04° (gate 0.0005°) and the
largest saliency difference 7.5e-06 (gate 0.0001).

## Training data

- **drive2018** of the Sully Chen driving datasets (<https://github.com/SullyChen/driving-datasets>,
  MIT License): 63,825 frames over 55.9 minutes from one dashcam, a timestamp on every frame, a
  median 57.0 ms apart.
- **Labels**: the human's steering-wheel angle in degrees. 368 single-frame sensor dropouts, a
  logged zero between two large steering angles of the same sign, are repaired by interpolating
  their neighbours.
- **Split**: blocked. The drive is cut into blocks of 1,500 frames dealt by the repeating pattern
  `TTVTTET` (T train, V validation, E test), and 250 frames are dropped on each side of every
  boundary between a training block and an evaluation block, so every evaluation frame is at least
  501 frames from a training frame: 39,825 training, 6,000 validation and 6,000 test frames.
- **Recipe**: 20 epochs of AdamW on a cosine schedule from a learning rate of 0.0003, weight decay
  0.0001, batches of 128, a Huber loss on the angle, dropout 0.3, and strong augmentation (flips
  with the angle negated, brightness, contrast, shadows, position, colour, noise, occlusion), every
  draw but the per-pixel noise applied to both frames alike. The checkpoint with the lowest
  validation error is kept.
- **Seed choice**: three seeds were trained; the one with the lowest validation error ships (seed 2,
  7.15°). Test and another-day scores took no part in the choice.

## Evaluation

All errors are mean absolute errors in degrees, computed on the CPU, beside baselines scored on the
same frames.

- **Four rungs**: *shuffled* (frames shuffled, then dealt 70%, 15% and 15%), *blocked* (above),
  *later in the drive* (train on the first 70% of the drive, test on its end) and *another day*
  (drive2017, a different day, every frame from frame 2 on).
- **The paired set**: the test frames the blocked split and a seed's shuffled split have in common,
  893, 889 and 943 frames for the three seeds, on which the two models of each seed are compared.
- **Baselines**: always steering straight (0°) and always the training mean; and two oracles that
  read the human's own previous angles, which no camera model sees: they are bounds, not
  competitors.

## Results

Means ± standard deviations over three seeds unless the line names the shipped run.

- **The paired leakage measurement.** On identical held-out frames, the model trained on the
  blocked split is off by 3.05 ± 0.61 times as much as the one trained on shuffled frames
  (PilotNet: 3.17 ± 0.14). Testing on shuffled frames makes the same kind of model look about three
  times better than it is on road it has not seen nearby.
- **Blocked test frames**: 10.24° ± 0.68°, where always steering straight scores 17.48°: 41% ± 4%
  better than always straight.
- **Another day**: 10.64° ± 1.30°, where always steering straight scores 15.52°: 31% ± 8% better.
- **The shipped run alone**: 10.12° on the blocked test frames and 9.77° on another day.
- **By steering regime**, shipped run, blocked test frames: 6.09° on nearly straight frames, where
  always steering straight scores 2.76° and wins; 5.52° on gentle turns (12.60°); 36.05° on sharp
  ones (78.41°).
- **Trained on shuffled frames**, as the twin is, SightlineNet scores 2.90° ± 0.09° on its own
  shuffled test frames and 7.96° ± 0.13° on another day. The defect is the score the shuffled split
  reports, not the model it trains: on another day it is the more accurate of the two, in every
  seed.

## The multi-frame caveat

The model reads motion between two frames, so its input depends on timing. It learned frame pairs
114 ms apart on drive2018; drive2017 has no timestamps and is taken to run at the same rate.
Footage from another camera or frame rate, a dropped frame, or a cut breaks that assumption, and
the model cannot tell. The first two frames of any sequence have no history: evaluation starts at
frame 2, and the page's clips carry the two frames before their first scored frame. On the page,
the model reads the clips' frames from image sheets made from the drive, not from the playing
video, and its expected outputs were computed on those same decoded frames.

## Limitations

- One car and one camera: two drives in daylight around Rancho Palos Verdes and San Pedro in
  Southern California. Nothing here measures night, rain, other cars or other places.
- Open loop only: every score compares the model with what the human did, frame by frame. The model
  never steered, so nothing here says how it would drive.
- On frames steered 30° or more, the shipped run is still off by 36.05°.
- The same network on one frame is only level with always steering straight on another day
  (+2% ± 4%, against +31% ± 8% with two frames), so most of what this model carries to another
  day depends on the second frame, and so on its timing.
- The paired set is small (893, 889 and 943 frames, three seeds).
- On the page's 2018 clip, the twin trained on 71% of the frames and 97% sit within one frame of a
  frame it trained on, so it looks better there than the shipped model (5.63° against 14.26° on this
  clip). The clips are chosen for having the most steering, so their errors describe the clip.
- VisualBackProp shows where activations survive through the network, not why it chose an angle.

## Ethical considerations

- **Safety.** A steering estimate that looks good on a screen invites being trusted on a road. This
  one must never be: it is open loop, trained on one car, and its error on sharp turns is large.
  The dataset's author asks that no one ever pilot a car with software trained on it.
- **Measurement.** The project exists because a shuffled split overstates accuracy by about three
  times here; any number quoted from this model should name the protocol it was measured under.
- **Footage.** The drives are public dashcam footage of public roads; other vehicles and buildings
  appear in them. The page shows short excerpts at the drive's own low resolution.

## Versions

- Training and evaluation: Python 3.12.13, PyTorch 2.13.0, NumPy 2.5.1. Training ran on an Apple
  GPU through PyTorch's `mps` device; every evaluation ran on the `cpu`.
- Export: onnx 1.22.0 and onnxruntime 1.28.0.
- Code: trained at commit `493b3b8`, evaluated at commit `ed7ca37`.

## Licence

The model files are released under the MIT License, with the project's code, for research use only,
since they were trained on the Sully Chen driving datasets (MIT License, Copyright (c) 2018 Sully
Chen), whose author asks that the data be used purely for research and statistics and never to
pilot a vehicle.
