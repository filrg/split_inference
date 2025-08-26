# Full module: quant_delta_codec.py
import numpy as np
import struct
import math

def pack_bits(q_vals, num_bits):
    max_q = (1 << num_bits) - 1
    bitstring = 0
    bits_filled = 0
    packed = []

    for val in q_vals:
        val = int(val)
        assert 0 <= val <= max_q
        bitstring |= val << bits_filled
        bits_filled += num_bits
        while bits_filled >= 8:
            packed.append(bitstring & 0xFF)
            bitstring >>= 8
            bits_filled -= 8
    if bits_filled > 0:
        packed.append(bitstring & 0xFF)
    return bytes(packed)

def unpack_bits(data, num_bits, count):
    q_vals = []
    bitstring = 0
    bits_collected = 0
    byte_idx = 0

    for _ in range(count):
        while bits_collected < num_bits:
            bitstring |= data[byte_idx] << bits_collected
            bits_collected += 8
            byte_idx += 1
        q_vals.append(bitstring & ((1 << num_bits) - 1))
        bitstring >>= num_bits
        bits_collected -= num_bits

    return np.array(q_vals, dtype=np.uint16)

def encode_quant_delta(batches, num_bits=4):
    assert 1 <= num_bits <= 16
    max_q = (1 << num_bits) - 1

    vec_len = len(batches[0])
    arrs = [np.asarray(b, dtype=np.float32).ravel() for b in batches]

    global_min = min(a.min() for a in arrs)
    global_max = max(a.max() for a in arrs)
    if global_max == global_min:
        global_max += 1e-6
    scale = (global_max - global_min) / max_q

    out = bytearray()
    out.append(num_bits)
    out.extend(struct.pack('<I', vec_len))
    out.extend(struct.pack('<f', global_min))
    out.extend(struct.pack('<f', scale))

    q_prev = np.round((arrs[0] - global_min) / scale).astype(np.uint16)
    if num_bits < 8:
        out.extend(pack_bits(q_prev, num_bits))
    else:
        dtype = np.uint8 if num_bits == 8 else np.uint16
        out.extend(q_prev.astype(dtype).tobytes())

    mask_bytes_len = math.ceil(vec_len / 8)
    for arr in arrs[1:]:
        q = np.round((arr - global_min) / scale).astype(np.uint16)
        diff = q != q_prev
        mask = np.packbits(diff.astype(np.uint8), bitorder='little')
        mask = mask[:mask_bytes_len]
        out.extend(mask.tobytes())
        if num_bits < 8:
            out.extend(pack_bits(q[diff], num_bits))
        else:
            dtype = np.uint8 if num_bits == 8 else np.uint16
            out.extend(q[diff].astype(dtype).tobytes())
        q_prev = q
    return bytes(out)

def decode_quant_delta(buf):
    mv = memoryview(buf)
    idx = 0

    num_bits = mv[idx]
    idx += 1
    vec_len, = struct.unpack_from('<I', mv, idx); idx += 4
    global_min, = struct.unpack_from('<f', mv, idx); idx += 4
    scale, = struct.unpack_from('<f', mv, idx); idx += 4
    max_q = (1 << num_bits) - 1

    mask_bytes_len = math.ceil(vec_len / 8)
    batches = []

    if num_bits < 8:
        byte_len = math.ceil(num_bits * vec_len / 8)
        q = unpack_bits(mv[idx:idx+byte_len], num_bits, vec_len)
        idx += byte_len
    else:
        dtype = np.uint8 if num_bits == 8 else np.uint16
        itemsize = np.dtype(dtype).itemsize
        q = np.frombuffer(mv[idx:idx + vec_len * itemsize], dtype=dtype)
        idx += vec_len * itemsize
    arr = (q.astype(np.float32) * scale) + global_min
    batches.append(arr.copy())
    q_prev = q.copy()

    while idx < len(mv):
        mask = np.frombuffer(mv[idx:idx + mask_bytes_len], dtype=np.uint8)
        idx += mask_bytes_len
        diff_bits = np.unpackbits(mask, count=vec_len, bitorder='little')
        num_changed = diff_bits.sum()

        if num_bits < 8:
            byte_len = math.ceil(num_bits * num_changed / 8)
            q_changed = unpack_bits(mv[idx:idx+byte_len], num_bits, num_changed)
            idx += byte_len
        else:
            dtype = np.uint8 if num_bits == 8 else np.uint16
            itemsize = np.dtype(dtype).itemsize
            q_changed = np.frombuffer(mv[idx:idx + int(num_changed * itemsize)], dtype=dtype)
            idx += int(num_changed * itemsize)

        q_curr = q_prev.copy()
        q_curr[diff_bits.astype(bool)] = q_changed
        batches.append((q_curr.astype(np.float32) * scale) + global_min)
        q_prev = q_curr

    return batches

def Encoder(data_output, num_bits=4):
    shape_data = []
    encoded_data = []
    for output in data_output:
        if output is None:
            encoded_data.append(None)
            shape_data.append(0)
        else:
            shape = output.shape
            shape_data.append(shape)
            output = output.reshape(output.shape[0], -1)
            encoded_data.append(encode_quant_delta(output, num_bits))
    return encoded_data, shape_data

def Decoder(encoded_data, shape_data):
    decoded_data = []
    for i, output in enumerate(encoded_data):
        if output is None:
            decoded_data.append(None)
        else:
            shape = shape_data[i]
            data_decode = decode_quant_delta(output)
            data_decode = np.stack([arr.reshape(shape[1:]) for arr in data_decode])
            decoded_data.append(data_decode)
    return decoded_data
