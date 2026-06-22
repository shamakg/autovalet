## How does SimLingo work?

Each view (front camera, topdown) is split into one or more 448x448 tiles via dynamic_preprocess, not squished into a single image.

Front camera: since its cropped at the bottom (the car obstructs the view partially) it lands on a 2x1 grid --> 2 tiles
Topdown is square, so it's 1 tile

So each sample has 3 tiles, each resized to 448x448

Per-tile ViT forward:
Patch embed (1024 patches),
    - A 448×448×3 image gets cut into a non-overlapping grid of 14×14×3 patches → 32×32 = 1024 patches
    - A learned CLS token is prepended, making 1025 tokens total entering the transformer stack

Each patch gets positional encoding via attnetion

Shrink 1024 tokens to 256 (glue every 2x2 block of 4 neighboring patches together into one token)

