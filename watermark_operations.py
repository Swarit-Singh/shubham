import numpy as np

# --- Prediction Error Expansion (PEE) ---
def _pee_embed_core(image_channel, payload_bits, T):
    channel = image_channel.copy().astype(np.int32) # Use int32 for intermediate calcs
    payload_idx = 0
    params = {'T': T, 'original_shape': image_channel.shape}
    loc_map = np.zeros(channel.shape, dtype=bool)

    rows, cols = channel.shape
    for r in range(rows):
        for c in range(1, cols): # Predict from left neighbor
            if payload_idx >= len(payload_bits):
                break
            
            prediction = channel[r, c-1]
            error = channel[r, c] - prediction
            
            if -T <= error < T: # Expandable error
                if payload_idx < len(payload_bits):
                    bit_to_embed = payload_bits[payload_idx]
                    modified_error = 2 * error + bit_to_embed
                    channel[r, c] = prediction + modified_error
                    loc_map[r, c] = True # Mark as changed for data hiding
                    payload_idx += 1
            elif error >= T:
                channel[r, c] += T
            elif error < -T: # error <= -T effectively
                channel[r, c] -= T
        if payload_idx >= len(payload_bits):
            break
    
    # Clip to ensure pixel values are valid
    channel = np.clip(channel, 0, 255)
    params['embedded_count'] = payload_idx
    return channel.astype(np.uint8), params, loc_map

def _pee_extract_core(watermarked_channel, payload_length, T, loc_map_if_available=None):
    channel = watermarked_channel.copy().astype(np.int32)
    rec_channel = watermarked_channel.copy().astype(np.int32) # For recovery
    
    extracted_bits = []
    payload_idx = 0

    rows, cols = channel.shape
    for r in range(rows):
        for c in range(1, cols):
            if payload_idx >= payload_length:
                break

            prediction = channel[r, c-1] # Use watermarked for prediction during extraction
                                         # More robust PEE might use original neighbors if available
                                         # or iterate in reverse if scan order allows perfect recovery.
                                         # For simplicity here, we use the watermarked left neighbor.

            # If location map is available and this location was not used for embedding, skip.
            # This is a simplification. True PEE might not need explicit loc_map for extraction if T is well-chosen
            # or if shifts are perfectly reversible.
            # if loc_map_if_available is not None and not loc_map_if_available[r,c]:
            #    rec_channel[r,c] = channel[r,c] # No change
            #    continue

            watermarked_error = channel[r, c] - prediction
            
            # Check if this pixel was used for embedding based on modified error range
            # The range for 2e+b is [-2T+0, 2(T-1)+1] = [-2T, 2T-1]
            if -2*T <= watermarked_error < 2*T : # Potential embedded location
                if payload_idx < payload_length:
                    extracted_bit = watermarked_error % 2
                    extracted_bits.append(str(abs(extracted_bit))) # abs for safety with neg results of %

                    original_error = watermarked_error // 2
                    rec_channel[r, c] = prediction + original_error
                    payload_idx += 1
                else: # No more bits to extract, just recover
                    original_error = watermarked_error // 2
                    rec_channel[r, c] = prediction + original_error
            elif watermarked_error >= 2*T: # Was shifted right by T
                rec_channel[r, c] = channel[r, c] - T
            elif watermarked_error < -2*T +1 : # Was shifted left by T (approx)
                                              # Note: the condition might need to be more precise based on embed logic.
                                              # For e.g. watermarked_error <= -2T (original error was <= -T)
                rec_channel[r, c] = channel[r, c] + T
            # else: pixel was not modified or is out of typical PEE modification range

        if payload_idx >= payload_length:
            break
            
    rec_channel = np.clip(rec_channel, 0, 255)
    return ''.join(extracted_bits), rec_channel.astype(np.uint8)


def embed_pred_error(image, payload_bits_str, T=1):
    payload_bits_int = [int(b) for b in payload_bits_str]
    watermarked_image, params, _ = _pee_embed_core(image, payload_bits_int, T)
    # params will contain 'embedded_count' and 'T'
    return watermarked_image, params

def extract_pred_error(watermarked_image, payload_length, T=1, params_from_embedding=None):
    # loc_map might be passed via params_from_embedding if generated and needed
    extracted_bits_str, recovered_image = _pee_extract_core(watermarked_image, payload_length, T)
    return extracted_bits_str, recovered_image


# --- Histogram Shifting (HS) ---
def _find_hs_params(image_channel):
    hist, bins = np.histogram(image_channel.flatten(), bins=np.arange(257))
    peak_val = np.argmax(hist)
    
    # Find first zero/min to the right of the peak
    zero_val_right = -1
    for i in range(peak_val + 1, 256):
        if hist[i] == 0:
            zero_val_right = i
            break
    if zero_val_right == -1: # If no zero, find min
        if peak_val + 1 < 256:
            zero_val_right = peak_val + 1 + np.argmin(hist[peak_val+1:])
        else: # Peak is at 255, cannot shift right
            zero_val_right = 255


    # Find first zero/min to the left of the peak
    zero_val_left = -1
    for i in range(peak_val - 1, -1, -1):
        if hist[i] == 0:
            zero_val_left = i
            break
    if zero_val_left == -1: # If no zero, find min
        if peak_val -1 >= 0:
            zero_val_left = np.argmin(hist[:peak_val])
        else: # Peak is at 0, cannot shift left
            zero_val_left = 0
            
    # Choose the zero/min that gives larger capacity (larger distance from peak)
    # Or a simpler strategy: prefer right shift if possible
    if peak_val < 255 and (zero_val_right != -1 ): # and (peak_val - zero_val_left < zero_val_right - peak_val)
        # Shift (peak_val, zero_val_right) to the right
        return peak_val, zero_val_right, 'right'
    elif peak_val > 0 and (zero_val_left != -1):
        # Shift (zero_val_left, peak_val) to the left
        return peak_val, zero_val_left, 'left'
    else: # Fallback if no suitable zero/min found, or peak is at boundary
        if peak_val < 128 and peak_val < 255: # Try to shift right
             return peak_val, min(255, peak_val + 1 + np.argmin(hist[peak_val+1:]) if peak_val+1 < 256 else 255), 'right'
        elif peak_val > 0: # Try to shift left
             return peak_val, max(0, np.argmin(hist[:peak_val]) if peak_val > 0 else 0), 'left'
        else: # Cannot shift (e.g. flat image)
            return peak_val, peak_val, 'none'


def embed_hist_shift(image, payload_bits_str):
    img_arr = image.copy().astype(np.int16)
    payload_bits = [int(b) for b in payload_bits_str]
    
    p, z, direction = _find_hs_params(img_arr)
    params = {'p': p, 'z': z, 'direction': direction, 'embedded_count': 0}

    if direction == 'none' or p == z: # No capacity
        return img_arr.astype(np.uint8), params

    idx = 0
    for r in range(img_arr.shape[0]):
        for c in range(img_arr.shape[1]):
            if idx >= len(payload_bits):
                break
            
            val = img_arr[r, c]
            if direction == 'right':
                if val == p:
                    if payload_bits[idx] == 0:
                        img_arr[r, c] = p # Stays p
                    else: # bit is 1
                        img_arr[r, c] = p + 1
                    idx += 1
                elif p < val < z:
                    img_arr[r, c] = val + 1
            elif direction == 'left':
                if val == p:
                    if payload_bits[idx] == 0:
                        img_arr[r, c] = p # Stays p
                    else: # bit is 1
                        img_arr[r, c] = p - 1
                    idx += 1
                elif z < val < p:
                    img_arr[r, c] = val - 1
        if idx >= len(payload_bits):
            break
            
    params['embedded_count'] = idx
    return np.clip(img_arr, 0, 255).astype(np.uint8), params

def extract_hist_shift(watermarked_image, payload_length, p, z, direction):
    img_arr = watermarked_image.copy().astype(np.int16)
    rec_img_arr = watermarked_image.copy().astype(np.int16)
    extracted_bits = []
    idx = 0

    for r in range(img_arr.shape[0]):
        for c in range(img_arr.shape[1]):
            if idx >= payload_length:
                break
            
            val = img_arr[r, c]
            rec_val = val # Default recovered value is current value

            if direction == 'right':
                if val == p:
                    extracted_bits.append('0')
                    rec_val = p
                    idx += 1
                elif val == p + 1:
                    extracted_bits.append('1')
                    rec_val = p
                    idx += 1
                elif p + 1 < val <= z: # Shifted from val-1
                    rec_val = val - 1
                # else: no change or outside shifting range
            
            elif direction == 'left':
                if val == p:
                    extracted_bits.append('0')
                    rec_val = p
                    idx += 1
                elif val == p - 1:
                    extracted_bits.append('1')
                    rec_val = p
                    idx += 1
                elif z <= val < p -1: # Shifted from val+1
                    rec_val = val + 1
                # else: no change or outside shifting range
            
            rec_img_arr[r,c] = rec_val
        if idx >= payload_length:
            break
            
    return "".join(extracted_bits), np.clip(rec_img_arr, 0, 255).astype(np.uint8)


# --- ML-Assisted (Classic - Placeholder, uses PEE T=1) ---
def embed_ml_assisted(image, payload_bits_str, T_predicted=1): # T_predicted could come from another ML model
    # For demonstration, this is a simple PEE.
    # A true ML-assisted method might use features.py and ml_optimization.py
    # to determine optimal T or other parameters.
    return embed_pred_error(image, payload_bits_str, T=T_predicted)

def extract_ml_assisted(watermarked_image, payload_length, T_predicted=1, params_from_embedding=None):
    return extract_pred_error(watermarked_image, payload_length, T=T_predicted, params_from_embedding=params_from_embedding)


# --- ML-Assisted (CNN+XGBoost for PEE Threshold T) ---
def embed_ml_cnn_xgboost(image, payload_bits_str, T=1): # T is predicted by CNN+XGBoost in app.py
    return embed_pred_error(image, payload_bits_str, T=T)

def extract_ml_cnn_xgboost(watermarked_image, payload_length, T=1, params_from_embedding=None):
    return extract_pred_error(watermarked_image, payload_length, T=T, params_from_embedding=params_from_embedding)
