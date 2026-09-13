"""Create an honest semantic-map authoring skeleton from a source PNG.

The map catalogs used by this project expose rendered images, not collision data.  This tool
captures the facts the image *can* establish without guessing semantics: byte checksum, native
pixel dimensions, and the declared pixel-to-cell transform.  Every terrain cell is emitted as
``?`` (unknown / fail-closed) for a human or a separately validated decoder to classify.

It never downloads or writes an artifact.  Redirect or copy the JSON only after reviewing it:

    PYTHONPATH=backend python backend/scripts/bootstrap_map_geometry.py map.png \
      --map-id 15000072 --name "Bridge Too Far" --mode Heist \
      --revision catalog-v1 --source-uri https://cdn.example/map.png --pixels-per-cell 30

The output is one ``maps[]`` entry for the atomic world-model bundle documented in
``docs/spatial-world-model.md``.  It is intentionally unusable for reasoning until its unknown
cells and objective/spawn coordinates have been verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import zlib
from pathlib import Path

from bsdraft.world.schema import (
    MAX_GRID_CELLS,
    MAX_GRID_DIMENSION,
    MIN_CELL_SIZE_WORLD,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_DECODED_BYTES = 64 * 1024 * 1024

_PNG_BIT_DEPTHS = {
    0: frozenset((1, 2, 4, 8, 16)),  # grayscale
    2: frozenset((8, 16)),           # truecolour
    3: frozenset((1, 2, 4, 8)),      # indexed colour
    4: frozenset((8, 16)),           # grayscale + alpha
    6: frozenset((8, 16)),           # truecolour + alpha
}
_PNG_SAMPLES_PER_PIXEL = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def png_dimensions(path: Path) -> tuple[int, int]:
    """Validate the PNG chunk envelope and read dimensions without an image dependency.

    This is intentionally not semantic image decoding, but it does reject truncated bytes, bad
    chunk CRCs, missing image data, and malformed chunk order before recording source provenance.
    """
    size = path.stat().st_size
    if size <= 0 or size > MAX_SOURCE_BYTES:
        raise ValueError(f"PNG size must be in 1..{MAX_SOURCE_BYTES} bytes (got {size})")
    with path.open("rb") as fh:
        if fh.read(8) != PNG_SIGNATURE:
            raise ValueError("source does not have the PNG signature")
        dimensions = None
        expected_decoded_bytes = None
        row_bytes = None
        colour_type = None
        saw_palette = False
        idat_parts = []
        idat_closed = False
        chunk_index = 0
        while True:
            raw_length = fh.read(4)
            if len(raw_length) != 4:
                raise ValueError("PNG ended before an IEND chunk")
            length = struct.unpack(">I", raw_length)[0]
            if length > MAX_SOURCE_BYTES:
                raise ValueError(f"PNG chunk is too large ({length} bytes)")
            chunk_type = fh.read(4)
            data = fh.read(length)
            raw_crc = fh.read(4)
            if len(chunk_type) != 4 or len(data) != length or len(raw_crc) != 4:
                raise ValueError("PNG contains a truncated chunk")
            expected_crc = struct.unpack(">I", raw_crc)[0]
            actual_crc = zlib.crc32(chunk_type)
            actual_crc = zlib.crc32(data, actual_crc) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                raise ValueError(f"PNG chunk {chunk_type!r} has an invalid CRC")

            if chunk_index == 0 and chunk_type != b"IHDR":
                raise ValueError("PNG's first chunk must be IHDR")
            if chunk_type == b"IHDR":
                if chunk_index != 0 or length != 13 or dimensions is not None:
                    raise ValueError("PNG has an invalid IHDR chunk")
                (
                    width,
                    height,
                    bit_depth,
                    colour_type,
                    compression_method,
                    filter_method,
                    interlace_method,
                ) = struct.unpack(">IIBBBBB", data)
                if width <= 0 or height <= 0:
                    raise ValueError("PNG dimensions must be positive")
                if bit_depth not in _PNG_BIT_DEPTHS.get(colour_type, ()):
                    raise ValueError("PNG has an invalid bit-depth/colour-type combination")
                if compression_method != 0 or filter_method != 0:
                    raise ValueError("PNG uses an unsupported compression or filter method")
                if interlace_method != 0:
                    raise ValueError("interlaced PNG sources are not supported by this bootstrap")
                bits_per_pixel = bit_depth * _PNG_SAMPLES_PER_PIXEL[colour_type]
                row_bytes = (width * bits_per_pixel + 7) // 8
                expected_decoded_bytes = height * (1 + row_bytes)
                if expected_decoded_bytes > MAX_DECODED_BYTES:
                    raise ValueError(
                        "decoded PNG exceeds the source-image memory limit "
                        f"({expected_decoded_bytes} > {MAX_DECODED_BYTES} bytes)"
                    )
                dimensions = (width, height)
            elif chunk_type == b"PLTE":
                if idat_parts:
                    raise ValueError("PNG palette must precede image data")
                if length == 0 or length % 3 or length > 768:
                    raise ValueError("PNG palette has an invalid length")
                saw_palette = True
            elif chunk_type == b"IDAT":
                if idat_closed:
                    raise ValueError("PNG image-data chunks must be consecutive")
                idat_parts.append(data)
            elif chunk_type == b"IEND":
                if length != 0:
                    raise ValueError("PNG IEND chunk must be empty")
                if fh.read(1):
                    raise ValueError("PNG has trailing bytes after IEND")
                break
            elif idat_parts:
                idat_closed = True
            chunk_index += 1

    if dimensions is None or not idat_parts or not any(idat_parts):
        raise ValueError("PNG must contain IHDR and non-empty IDAT data")
    if colour_type == 3 and not saw_palette:
        raise ValueError("indexed-colour PNG must contain a palette before image data")

    compressed = b"".join(idat_parts)
    assert expected_decoded_bytes is not None  # established by the mandatory first IHDR
    try:
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(compressed, expected_decoded_bytes + 1)
        if decoder.unconsumed_tail or len(decoded) > expected_decoded_bytes:
            raise ValueError("PNG image data expands beyond its declared dimensions")
        remaining = expected_decoded_bytes + 1 - len(decoded)
        if remaining > 0:
            decoded += decoder.flush(remaining)
    except zlib.error as exc:
        raise ValueError("PNG contains invalid compressed image data") from exc
    if (
        len(decoded) != expected_decoded_bytes
        or not decoder.eof
        or decoder.unused_data
    ):
        raise ValueError("PNG image data does not match its declared dimensions")
    assert row_bytes is not None
    stride = row_bytes + 1
    if any(decoded[offset] > 4 for offset in range(0, len(decoded), stride)):
        raise ValueError("PNG image data contains an invalid scanline filter")
    return dimensions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_map_entry(
    path: Path,
    *,
    map_id: int,
    name: str,
    mode: str,
    revision: str,
    source_uri: str,
    pixels_per_cell: int,
    cell_size_world: float,
) -> dict:
    """Return a map entry whose terrain is wholly unknown until separately verified."""
    if not isinstance(map_id, int) or isinstance(map_id, bool) or map_id <= 0:
        raise ValueError("map id must be positive")
    if not name.strip() or not mode.strip() or not revision.strip() or not source_uri.strip():
        raise ValueError("name, mode, revision, and source URI must be non-empty")
    if (
        not isinstance(pixels_per_cell, int)
        or isinstance(pixels_per_cell, bool)
        or pixels_per_cell <= 0
    ):
        raise ValueError("pixels per cell must be positive")
    if (
        not isinstance(cell_size_world, (int, float))
        or isinstance(cell_size_world, bool)
        or not math.isfinite(float(cell_size_world))
        or cell_size_world < MIN_CELL_SIZE_WORLD
    ):
        raise ValueError(
            f"world cell size must be finite and >= {MIN_CELL_SIZE_WORLD}"
        )

    width_px, height_px = png_dimensions(path)
    if width_px % pixels_per_cell or height_px % pixels_per_cell:
        raise ValueError(
            f"{width_px}x{height_px} is not divisible by {pixels_per_cell} pixels per cell"
        )
    width = width_px // pixels_per_cell
    height = height_px // pixels_per_cell
    if (
        width > MAX_GRID_DIMENSION
        or height > MAX_GRID_DIMENSION
        or width * height > MAX_GRID_CELLS
    ):
        raise ValueError(
            f"derived grid {width}x{height} exceeds the v1 semantic-grid limits "
            f"({MAX_GRID_DIMENSION} per side, {MAX_GRID_CELLS} total cells)"
        )

    return {
        "map_id": map_id,
        "name": name.strip(),
        "mode": mode.strip(),
        "revision": revision.strip(),
        "annotation_method": "unverified",
        "verified_at": None,
        "cell_size_world": cell_size_world,
        "width": width,
        "height": height,
        "terrain": ["?" * width for _ in range(height)],
        "objective_cells": [],
        "ally_spawn_cells": [],
        "enemy_spawn_cells": [],
        "source": {
            "uri": source_uri.strip(),
            "sha256": sha256_file(path),
            "width_px": width_px,
            "height_px": height_px,
            "grid_origin_x_px": 0,
            "grid_origin_y_px": 0,
            "pixels_per_cell": pixels_per_cell,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit a fail-closed semantic-map skeleton from a local source PNG."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--map-id", required=True, type=int)
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--pixels-per-cell", required=True, type=int)
    parser.add_argument("--cell-size-world", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        entry = build_map_entry(
            args.image,
            map_id=args.map_id,
            name=args.name,
            mode=args.mode,
            revision=args.revision,
            source_uri=args.source_uri,
            pixels_per_cell=args.pixels_per_cell,
            cell_size_world=args.cell_size_world,
        )
    except (OSError, ValueError) as exc:
        print(f"map bootstrap failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(entry, indent=2))
    print(
        "warning: every terrain cell is '?' and therefore fail-closed; verify semantics before use",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
