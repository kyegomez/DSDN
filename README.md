# DSDN
My implementation of "Dual-Stream Diffusion Net "

## Model Architecture:

The DSDN consists of the following main components:

1. Encoder (E): Encodes input video frames x into latent space as z0 = E(x). Uses a pre-trained VQ-VAE encoder which is frozen during training.

2. Motion Decomposer: Decomposes z0 into content features z0 and motion features z~0. Uses a lightweight architecture with 1x1 convolutions and channel reduction. 

3. Forward Diffusion: Applies separate forward diffusion processes on z0 and z~0 to obtain priors zT and z~T. Uses the same noise schedule for both.

4. Personalized Content Generation Stream: Denoises zT to get z0|T. Uses a content base unit (pre-trained text-to-image model) and content increment unit.

5. Personalized Motion Generation Stream: Denoises z~T to get z~0|T. Uses a 3D UNet architecture. 

6. Dual-Stream Interaction: Aligns the generated z0|T and z~0|T using cross-attention transformers.

7. Motion Combiner: Compensates motion information into content features to obtain final video latent code z^0. Inverse of motion decomposer.

8. Decoder (D): Decodes z^0 to generate video frames x.

## Key Components:

1. Content Base Unit: Pre-trained text-to-image model (Stable Diffusion) that remains frozen during training.

2. Content Increment Unit: Small tunable network to refine content generation. Uses low-rank decomposition for efficiency.

3. Personalized Motion Generation Stream: 3D UNet diffusion model to generate coherent motion latent features.

4. Dual-Stream Interaction: Cross-attention modules between content and motion streams for alignment.

## Algorithm:

1. Encode input video x into z0 using VQ-VAE E

2. Decompose z0 into content z0 and motion z~0 features 

3. Perform forward diffusion on z0 and z~0 to get priors zT, z~T

4. Denoise zT in content stream to get z0|T
     - Use content base unit (frozen) 
     - Refine with trainable content increment unit

5. Denoise z~T in motion stream to get z~0|T
     - Use 3D UNet architecture

6. Align z0|T and z~0|T using dual-stream interaction
     - Apply cross-attention transformers between streams

7. Combine aligned z0|T and z~0|T to get final latent code z^0 

8. Decode z^0 to generate video frames x using decoder D

This analysis covers the key components and algorithm for implementing the DSDN model in PyTorch. The dual-stream architecture with separate content and motion diffusion along with the interaction module are critical for generating smooth and consistent videos from text descriptions.
