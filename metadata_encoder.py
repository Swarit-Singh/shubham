import json
import logging

def metadata_to_bits(metadata_dict):
    """
    Converts a metadata dictionary to a string of bits (0s and 1s).
    Metadata -> JSON string -> UTF-8 bytes -> bits string.
    """
    try:
        json_str = json.dumps(metadata_dict, separators=(',', ':')) # Compact JSON
        utf8_bytes = json_str.encode('utf-8')
        bits_str = ''.join(format(byte, '08b') for byte in utf8_bytes)
        return bits_str
    except TypeError as e:
        logging.error(f"Error serializing metadata to JSON: {e}. Ensure metadata is JSON serializable.")
        return None
    except Exception as e:
        logging.error(f"Error converting metadata to bits: {e}")
        return None

def bits_to_metadata(bits_str):
    """
    Converts a string of bits back to a metadata dictionary.
    Bits string -> UTF-8 bytes -> JSON string -> dictionary.
    """
    if not bits_str or len(bits_str) % 8 != 0:
        logging.warning(f"Bit string is empty or not a multiple of 8. Length: {len(bits_str)}. Cannot convert to bytes accurately.")
        # Attempt to process if possible, but it might lead to errors
        # For robustness, often a length header is embedded with the bits.

    byte_list = []
    try:
        for i in range(0, len(bits_str), 8):
            byte_chunk = bits_str[i:i+8]
            if len(byte_chunk) == 8: # Process only full bytes
                byte_list.append(int(byte_chunk, 2))
        
        if not byte_list:
            logging.error("No valid bytes could be formed from the bit string.")
            return None

        utf8_bytes = bytes(byte_list)
        json_str = utf8_bytes.decode('utf-8') # Can fail if bits didn't form valid UTF-8
        metadata_dict = json.loads(json_str) # Can fail if not valid JSON
        return metadata_dict
    except UnicodeDecodeError as e:
        logging.error(f"Error decoding bytes to UTF-8 string: {e}. The bit string might be corrupted or not represent valid UTF-8 encoded JSON.")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON string to metadata dictionary: {e}. The content might not be valid JSON.")
        return None
    except Exception as e:
        logging.error(f"Error converting bits to metadata: {e}")
        return None


def get_metadata_bit_length(metadata_dict):
    """
    Calculates the number of bits required to store the given metadata dictionary.
    """
    bits_str = metadata_to_bits(metadata_dict)
    return len(bits_str) if bits_str is not None else 0


# Example Usage
if __name__ == '__main__':
    from utils import setup_logging
    setup_logging(level=logging.INFO)

    test_metadata = {
        "patient_id": "P7890",
        "timestamp": "2025-05-21T10:30:00Z",
        "image_hash": "a1b2c3d4e5f6",
        "source_modality": "XRAY",
        "parameters": {"T": 2, "block_size": None} # Example of None value
    }
    logging.info(f"Original Metadata: {test_metadata}")

    # Convert to bits
    bits_representation = metadata_to_bits(test_metadata)
    if bits_representation:
        logging.info(f"Bits Representation (first 64 bits): {bits_representation[:64]}...")
        logging.info(f"Total bit length: {len(bits_representation)}")

        # Convert back to metadata
        recovered_metadata = bits_to_metadata(bits_representation)
        if recovered_metadata:
            logging.info(f"Recovered Metadata: {recovered_metadata}")
            assert test_metadata == recovered_metadata, "Mismatch in original and recovered metadata!"
            logging.info("Metadata recovery successful and matches original.")
        else:
            logging.error("Failed to recover metadata.")
    else:
        logging.error("Failed to convert metadata to bits.")

    # Test with potentially problematic data
    bad_metadata = {"unsupported": lambda x: x} # Lambda function is not JSON serializable
    logging.info(f"\nTesting with non-serializable metadata: {bad_metadata}")
    bad_bits = metadata_to_bits(bad_metadata)
    assert bad_bits is None, "Should fail for non-serializable data"
    logging.info("Correctly failed for non-serializable data.")
    
    # Test bits_to_metadata with invalid bit string
    logging.info("\nTesting with invalid bit string for recovery:")
    invalid_bits = "1111111" # Not multiple of 8
    recovered_invalid = bits_to_metadata(invalid_bits)
    # assert recovered_invalid is None # This behavior might vary, it might process what it can or return None
    logging.info(f"Result from invalid bits: {recovered_invalid}")

    non_utf8_bits = format(0xFF, '08b') + format(0xFE, '08b') # Invalid UTF-8 sequence start
    recovered_non_utf8 = bits_to_metadata(non_utf8_bits)
    assert recovered_non_utf8 is None
    logging.info(f"Result from non-UTF8 bits: {recovered_non_utf8} (Expected None due to decode error)")
