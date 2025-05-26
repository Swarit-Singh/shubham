import numpy as np

def embed_histogram_shift(image, payload_bits):
    """
    Embed a list of payload bits into the image using histogram shifting.
    Returns the watermarked image and parameters for extraction.
    """
    img = image.copy()
    params = {"peaks": [], "zeros": [], "dirs": []}
    payload_index = 0
    while payload_index < len(payload_bits):
        # Compute histogram of pixel values (0-255)
        hist = np.bincount(img.flatten(), minlength=256)
        # Find peak intensity and a zero-frequency intensity
        peak_val = np.argmax(hist)                # value with max frequency
        peak_count = hist[peak_val]
        zero_val = None; direction = None
        # Search for a zero count on the right side
        for v in range(peak_val+1, 256):
            if hist[v] == 0:
                zero_val = v
                direction = +1
                break
        # If none on right, search left side
        if zero_val is None:
            for v in range(peak_val-1, -1, -1):
                if hist[v] == 0:
                    zero_val = v
                    direction = -1
                    break
        if zero_val is None:
            break  # No available zero-bin, capacity reached
        # Shift pixel intensities to create gap
        if direction == +1:
            for v in range(zero_val-1, peak_val, -1):
                img[img == v] = v + 1
        else:
            for v in range(zero_val+1, peak_val):
                img[img == v] = v - 1
        # Embed the next bit in pixels at peak_val
        bit = payload_bits[payload_index]
        if bit == 1:
            if direction == +1:
                img[img == peak_val] = peak_val + 1
            else:
                img[img == peak_val] = peak_val - 1
        # Record parameters for extraction
        params["peaks"].append(peak_val)
        params["zeros"].append(zero_val)
        params["dirs"].append(direction)
        payload_index += 1
    params["payload_length"] = payload_index
    return img, params

def extract_histogram_shift(wm_image, params):
    """
    Extract embedded bits and recover original image using recorded parameters.
    """
    img = wm_image.copy()
    extracted_bits = []
    # Reverse the embedding process
    for i in range(params["payload_length"]-1, -1, -1):
        peak_val = params["peaks"][i]
        zero_val = params["zeros"][i]
        direction = params["dirs"][i]
        # Determine embedded bit by checking shifted peak
        if direction == +1:
            bit = 1 if np.any(img == peak_val+1) else 0
            # Restore pixels that were peak_val+1 back to peak_val
            img[img == peak_val+1] = peak_val
            # Reverse the histogram shift (shift values back down)
            for v in range(peak_val+2, zero_val+1):
                img[img == v] = v - 1
        else:
            bit = 1 if np.any(img == peak_val-1) else 0
            img[img == peak_val-1] = peak_val
            for v in range(peak_val-2, zero_val-1, -1):
                img[img == v] = v + 1
        extracted_bits.append(bit)
    extracted_bits.reverse()
    return img, extracted_bits
