from pathlib import Path
import struct

# Historical DXGI ASTC extension formats as defined by Microsoft.
ASTC_FORMATS = {
    133: (4, 4, False, "ASTC_4X4_TYPELESS"),
    134: (4, 4, False, "ASTC_4X4_UNORM"),
    135: (4, 4, True,  "ASTC_4X4_UNORM_SRGB"),
    137: (5, 4, False, "ASTC_5X4_TYPELESS"),
    138: (5, 4, False, "ASTC_5X4_UNORM"),
    139: (5, 4, True,  "ASTC_5X4_UNORM_SRGB"),
    141: (5, 5, False, "ASTC_5X5_TYPELESS"),
    142: (5, 5, False, "ASTC_5X5_UNORM"),
    143: (5, 5, True,  "ASTC_5X5_UNORM_SRGB"),
    145: (6, 5, False, "ASTC_6X5_TYPELESS"),
    146: (6, 5, False, "ASTC_6X5_UNORM"),
    147: (6, 5, True,  "ASTC_6X5_UNORM_SRGB"),
    149: (6, 6, False, "ASTC_6X6_TYPELESS"),
    150: (6, 6, False, "ASTC_6X6_UNORM"),
    151: (6, 6, True,  "ASTC_6X6_UNORM_SRGB"),
    153: (8, 5, False, "ASTC_8X5_TYPELESS"),
    154: (8, 5, False, "ASTC_8X5_UNORM"),
    155: (8, 5, True,  "ASTC_8X5_UNORM_SRGB"),
    157: (8, 6, False, "ASTC_8X6_TYPELESS"),
    158: (8, 6, False, "ASTC_8X6_UNORM"),
    159: (8, 6, True,  "ASTC_8X6_UNORM_SRGB"),
    161: (8, 8, False, "ASTC_8X8_TYPELESS"),
    162: (8, 8, False, "ASTC_8X8_UNORM"),
    163: (8, 8, True,  "ASTC_8X8_UNORM_SRGB"),
    165: (10, 5, False, "ASTC_10X5_TYPELESS"),
    166: (10, 5, False, "ASTC_10X5_UNORM"),
    167: (10, 5, True,  "ASTC_10X5_UNORM_SRGB"),
    169: (10, 6, False, "ASTC_10X6_TYPELESS"),
    170: (10, 6, False, "ASTC_10X6_UNORM"),
    171: (10, 6, True, "ASTC_10X6_UNORM_SRGB"),
    173: (10, 8, False, "ASTC_10X8_TYPELESS"),
    174: (10, 8, False, "ASTC_10X8_UNORM"),
    175: (10, 8, True, "ASTC_10X8_UNORM_SRGB"),
    177: (10, 10, False, "ASTC_10X10_TYPELESS"),
    178: (10, 10, False, "ASTC_10X10_UNORM"),
    179: (10, 10, True, "ASTC_10X10_UNORM_SRGB"),
    181: (12, 10, False, "ASTC_12X10_TYPELESS"),
    182: (12, 10, False, "ASTC_12X10_UNORM"),
    183: (12, 10, True, "ASTC_12X10_UNORM_SRGB"),
    185: (12, 12, False, "ASTC_12X12_TYPELESS"),
    186: (12, 12, False, "ASTC_12X12_UNORM"),
    187: (12, 12, True, "ASTC_12X12_UNORM_SRGB"),
}

def div_up(n: int, d: int) -> int:
    return (n + d - 1) // d

def round_up(n: int, a: int) -> int:
    return div_up(n, a) * a

def block_height_gobs(height_in_blocks: int) -> int:
    # Switch/Tegra block-linear textures use a power-of-two GOB block height. The smallest useful unit is one 8-row GOB, cap at 16 GOBs.
    gob_rows = max(1, div_up(height_in_blocks, 8))
    h = 1
    while h < gob_rows:
        h <<= 1
    return min(h, 16)

def surface_size(width_blocks: int, height_blocks: int, bytes_per_block: int) -> int:
    pitch = round_up(width_blocks * bytes_per_block, 64)
    bh = block_height_gobs(height_blocks)
    return pitch * round_up(height_blocks, 8 * bh)

def astc_mip_layout(width: int, height: int, mip_count: int, block_x: int, block_y: int):
    result = []
    w, h = width, height
    for mip in range(mip_count):
        wb = div_up(w, block_x)
        hb = div_up(h, block_y)
        size = surface_size(wb, hb, 16)
        result.append((mip, w, h, wb, hb, size))
        w = max(1, w // 2)
        h = max(1, h // 2)
    return result

def infer_astc_header_size(data: bytes, layout) -> int:
    expected_payload = sum(row[5] for row in layout)
    for header_size in range(148, min(204, len(data)) + 1, 2):
        if len(data) - header_size == expected_payload:
            return header_size
    raise ValueError(f"could not infer the SDFTOC header length.\nfile={len(data)} bytes, expected ASTC payload={expected_payload}")

def parse_dds(data: bytes):
    if len(data) < 148 or data[:4] != b"DDS ":
        raise ValueError("not a standard DDS file")

    height = struct.unpack_from("<I", data, 12)[0]
    width = struct.unpack_from("<I", data, 16)[0]
    mip_count = struct.unpack_from("<I", data, 28)[0] or 1
    fourcc = data[84:88]

    dxgi = None
    if fourcc == b"DX10":
        if len(data) < 148:
            raise ValueError("DDS says DX10 but file is too short")
        dxgi = struct.unpack_from("<I", data, 128)[0]

    return width, height, mip_count, fourcc, dxgi

def convert_astc(path: Path, out_path: Path):
    data = path.read_bytes()
    width, height, mip_count, fourcc, dxgi = parse_dds(data)

    bx, by, srgb, fmt_name = ASTC_FORMATS[dxgi]

    layout = astc_mip_layout(width, height, mip_count, bx, by)
    header_size = infer_astc_header_size(data, layout)

    # The first 148 bytes are the actual DDS/DX10 container header. The remaining bytes up to header_size are extra Snowdrop/Switch metadata.
    out_path.write_bytes(data[:148] + data[header_size:])


inputPath = "./"
inp = Path(inputPath)
files = inp.glob("*.dds")

outputPath = "./"
out_dir = Path(outputPath)
out_dir.mkdir(parents=True, exist_ok=True)

for path in files:
    out = Path(str(path.stem) + "_fixed").with_suffix(".dds")

    data = path.read_bytes()
    _, _, _, fourcc, dxgi = parse_dds(data)
    if fourcc == b"DX10" and dxgi in ASTC_FORMATS:
        convert_astc(path, out)

