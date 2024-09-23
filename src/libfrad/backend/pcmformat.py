import numpy as np

def ff_format_to_numpy_type(x: str) -> np.dtype:

    match x:
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