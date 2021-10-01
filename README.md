# WFS-Net
<p>Unofficial implementation of "Face inpainting network for large missing regions based on weighted facial similarity"</p>

<p>[<a href="https://www.sciencedirect.com/science/article/abs/pii/S0925231219317941">Paper</a>]</p>

## Usage
usage: train.py [-h] [--original ORIGINAL] [--masked MASKED] [--k1 K1] [--database DATABASE] [--k2 K2] [--n N] [--epoch EPOCH]
                [--checkpoints CHECKPOINTS] [--batch BATCH]

optional arguments:
  
  - --original default="original". <p>The name of a folder which includes the original images.</p> <p>The images will be resized as the longer edge become 128 px conserving its aspect, then k1 pieces of 128x128 regions will be randomly cropped.</p> <p style="color: red">IMPORTANT: The images under this folder must be named in order. ex) 12.jpg.</p>
  - --masked default="masked". <p>The name of a folder which includes the masked images.</p> <p>The images will be resized as the longer edge become 128 px conserving its aspect, then k1 pieces of 128x128 regions will be randomly cropped.</p> <p style="color: red">IMPORTANT: The images under this folder must be named in order. ex) 12.jpg.</p>

  - --k1 default=4. <p>The number of regions cropped from original and masked images.</p>

  - --database default"database". <p>The name of a folder which includes the dataset images used as WSFS.</p> <p>The images will be resized as the longer edge become 128 px conserving its aspect, then k pieces of 128x128 regions will be randomly cropped.</p>
  --k2 default=1. <p>The number of regions cropped from original and masked images.</p>
  --n default=10. <p>The size of WSFS</p>
  --epoch default=100. <p>The number of epoch</p>
  --checkpoints default="checkpoints". <p>The name of a folder in which checkpoints files will be created.</p>
  --batch default=1. <p>The batch size</p>

## LICENSE
MIT License. See LICENSE for details.

## Author
<a href="https://twitter.com/manatoy_jpn">Manato Yo</a>
