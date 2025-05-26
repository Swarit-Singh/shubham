let’s chart out the trade-off between capacity and PSNR to help visualize how embedding more watermark bits affects image quality.

For each threshold T from 1 to 10:

Capacity = Number of pixels where prediction error ∈ [-T, T] (i.e., watermark-embeddable pixels)

PSNR = Quality of the image after embedding & restoring using threshold T

✅ What You’ll See
Threshold vs Capacity: Capacity increases sharply as threshold increases

Threshold vs PSNR: PSNR decreases (image quality drops) as threshold increases

Capacity vs PSNR: A curve showing the trade-off — more capacity, lower quality