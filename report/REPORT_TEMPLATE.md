# Assignment 2 Report – Generative AI

## Q1: Denoising Autoencoder
1. **Dataset & preprocessing** – BSDS300, 200 train (160/40 train/val) + 100 test, resized 128x128, scaled to [0,1].
2. **Noise models** – Gaussian (sigma), salt-and-pepper (amount).
3. **Architecture** – 4 stride-2 conv layers 3-32-64-128-256 (BN+ReLU), 1x1 conv to bottleneck (default 16x8x8 = 1024 values), mirrored transposed-conv decoder, Sigmoid output. Loss: MSE. Adam 1e-3.
4. **Results** – paste figure of clean/noisy/denoised; table of MSE/PSNR/SSIM per noise type.
5. **Experimental study** – paste `q1_noise_results.csv`, `q1_bottleneck_results.csv`, `q1_experiments.png`; discuss noise level and bottleneck effects.

## Q2: CycleGAN
1. **Method** – generators (ResNet, 9 blocks), PatchGAN discriminators, LSGAN + cycle (10) + identity (5) losses, image pool.
2. **Training setup** – epochs, batch size, learning rate and schedule, hardware, any dataset subsampling; checkpoint/resume strategy.
3. **Results** – `q2_losses.png`; face->sketch and sketch->face examples; failure cases.
4. **Hyperparameter tuning** – what was tried and observed.
5. **User interface** – screenshots of the Flask app, description of the sketch/face detection heuristic.

## Conclusion & limitations
