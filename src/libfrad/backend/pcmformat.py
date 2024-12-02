import numpy as np
import sys

def ff_format_to_numpy_type(x: str) -> np.dtype:

    match x.lower():
        case 'u8': return np.dtype('u1')
        case 'u16be': return np.dtype('>u2')
        case 'u16le': return np.dtype('<u2')
        case 'u32be': return np.dtype('>u4')
        case 'u32le': return np.dtype('<u4')
        case 'u64be': return np.dtype('>u8')
        case 'u64le': return np.dtype('<u8')

        case 's8': return np.dtype('i1')
        case 's16be': return np.dtype('>i2')
        case 's16le': return np.dtype('<i2')
        case 's32be': return np.dtype('>i4')
        case 's32le': return np.dtype('<i4')
        case 's64be': return np.dtype('>i8')
        case 's64le': return np.dtype('<i8')

        case 'f16be': return np.dtype('>f2')
        case 'f16le': return np.dtype('<f2')
        case 'f32be': return np.dtype('>f4')
        case 'f32le': return np.dtype('<f4')
        case 'f64be': return np.dtype('>f8')
        case 'f64le': return np.dtype('<f8')

        case _:
            print(f'Invalid format: {x}', file=sys.stderr)
            exit(1)

def to_f64(pcm: np.ndarray, pcm_format: np.dtype) -> np.ndarray:
    if pcm_format in [np.float16, np.float32, np.float64]: return pcm

    if pcm_format == np.int8: pcm = pcm.astype(np.float64) / 128
    if pcm_format == np.int16: pcm = pcm.astype(np.float64) / 32768
    if pcm_format == np.int32: pcm = pcm.astype(np.float64) / 2147483648
    if pcm_format == np.int64: pcm = pcm.astype(np.float64) / 9223372036854775808

    if pcm_format == np.uint8: pcm = (pcm.astype(np.float64) / 128) - 1
    if pcm_format == np.uint16: pcm = (pcm.astype(np.float64) / 32768) - 1
    if pcm_format == np.uint32: pcm = (pcm.astype(np.float64) / 2147483648) - 1
    if pcm_format == np.uint64: pcm = (pcm.astype(np.float64) / 9223372036854775808) - 1

    return pcm

def from_f64(pcm: np.ndarray, pcm_format: np.dtype) -> np.ndarray:
    if pcm_format in [np.float16, np.float32, np.float64]: return pcm

    if pcm_format == np.int8: pcm = (pcm * 128).astype(np.int8)
    if pcm_format == np.int16: pcm = (pcm * 32768).astype(np.int16)
    if pcm_format == np.int32: pcm = (pcm * 2147483648).astype(np.int32)
    if pcm_format == np.int64: pcm = (pcm * 9223372036854775808).astype(np.int64)

    if pcm_format == np.uint8: pcm = ((pcm + 1) * 128).astype(np.uint8)
    if pcm_format == np.uint16: pcm = ((pcm + 1) * 32768).astype(np.uint16)
    if pcm_format == np.uint32: pcm = ((pcm + 1) * 2147483648).astype(np.uint32)
    if pcm_format == np.uint64: pcm = ((pcm + 1) * 9223372036854775808).astype(np.uint64)

    return pcm