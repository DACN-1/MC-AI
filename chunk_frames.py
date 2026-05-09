"""
Frame Chunking Utilities - Convert PNG frames to HDF5 format

This module provides utilities to convert extracted PNG frames into
HDF5 chunked format, significantly reducing inode usage and improving I/O performance.

Expected improvements:
- Inode reduction: 6,007 → 61 per trajectory (99%)
- Storage reduction: 600 MB → 80 MB per trajectory (87%)
- I/O improvement: 50-100ms → 10-20ms per batch (5x faster)
"""

import h5py
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
from typing import Optional
import argparse


_VALID_COMPRESSION = {"gzip", "none"}


class FrameChunker:
    """Convert extracted PNG frames or MP4 videos into HDF5 chunks for efficient storage and access."""

    def __init__(self, frames_per_chunk: int = 100, compression: str = "gzip"):
        """
        Initialize FrameChunker.

        Args:
            frames_per_chunk: Number of frames per HDF5 chunk (default: 100)
            compression: Compression algorithm ('gzip' or 'none').
        """
        if compression not in _VALID_COMPRESSION:
            raise ValueError(
                f"Unsupported compression {compression!r}. Choose one of {sorted(_VALID_COMPRESSION)}."
            )
        self.frames_per_chunk = frames_per_chunk
        self.compression = compression

    def _h5_compression_kwargs(self) -> dict:
        if self.compression == "gzip":
            return {"compression": "gzip", "compression_opts": 4}
        return {}

    def chunk_video(
        self,
        video_path: Path,
        output_file: Path,
        verbose: bool = True
    ) -> dict:
        """
        Convert MP4 video directly to HDF5 file without extracting PNG frames.
        This avoids inode explosion by processing frames one at a time.

        Args:
            video_path: Path to MP4 video file
            output_file: Output HDF5 file path
            verbose: Print progress messages

        Returns:
            dict with conversion statistics
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            if verbose:
                print(f"Converting video: {video_path.name}")
                print(f"  Frames: {total_frames}")
                print(f"  Dimensions: {frame_height}x{frame_width}")
                print(f"  FPS: {fps}")
                print(f"  Chunk size: {self.frames_per_chunk} frames")
                print(f"  Compression: {self.compression}")

            # Create output directory if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Get video file size for stats
            video_size = video_path.stat().st_size

            # Create HDF5 file with chunked dataset
            with h5py.File(output_file, 'w') as f:
                dset = f.create_dataset(
                    'frames',
                    shape=(total_frames, frame_height, frame_width, 3),
                    maxshape=(None, frame_height, frame_width, 3),
                    dtype=np.uint8,
                    chunks=(self.frames_per_chunk, frame_height, frame_width, 3),
                    **self._h5_compression_kwargs(),
                )

                # Buffer one chunk in RAM before writing to avoid re-encoding
                # the same chunk on every per-frame write.
                buffer = np.empty(
                    (self.frames_per_chunk, frame_height, frame_width, 3), dtype=np.uint8
                )
                buffer_fill = 0
                buffer_start = 0
                frames_written = 0

                iterator = range(total_frames)
                if verbose:
                    iterator = tqdm(iterator, desc="  Writing frames")

                for _ in iterator:
                    ret, frame = cap.read()
                    if not ret:
                        if verbose:
                            print(f"\n  Warning: Failed to read frame {frames_written}/{total_frames}")
                        break
                    buffer[buffer_fill] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    buffer_fill += 1
                    frames_written += 1
                    if buffer_fill == self.frames_per_chunk:
                        dset[buffer_start : buffer_start + buffer_fill] = buffer[:buffer_fill]
                        buffer_start += buffer_fill
                        buffer_fill = 0

                if buffer_fill > 0:
                    dset[buffer_start : buffer_start + buffer_fill] = buffer[:buffer_fill]
                    frames_written = buffer_start + buffer_fill

                # Trim dataset if the source video was shorter than reported
                if frames_written < total_frames:
                    dset.resize((frames_written, frame_height, frame_width, 3))
                    total_frames = frames_written

                # Store metadata
                f.attrs['total_frames'] = total_frames
                f.attrs['frame_height'] = frame_height
                f.attrs['frame_width'] = frame_width
                f.attrs['channels'] = 3
                f.attrs['fps'] = fps
                f.attrs['frames_per_chunk'] = self.frames_per_chunk
                f.attrs['compression'] = self.compression
                f.attrs['source_video'] = str(video_path)

            # Get HDF5 file size
            h5_size = output_file.stat().st_size

            stats = {
                'total_frames': total_frames,
                'video_size': video_size,
                'h5_file_size': h5_size,
                'compression_ratio': video_size / h5_size if h5_size > 0 else 0,
                'storage_change': ((h5_size / video_size) - 1) * 100 if video_size > 0 else 0,
            }

            if verbose:
                print(f"  ✓ Conversion complete!")
                print(f"    Video size: {video_size / 1024**2:.1f} MB")
                print(f"    HDF5 size: {h5_size / 1024**2:.1f} MB")
                print(f"    Storage change: {stats['storage_change']:+.1f}%")

            return stats

        finally:
            cap.release()

    def chunk_trajectory(
        self,
        frames_dir: Path,
        output_file: Path,
        frame_height: int = 360,
        frame_width: int = 640,
        verbose: bool = True
    ) -> dict:
        """
        Convert frames/video_stem/ directory into HDF5 file.

        Input: frames/video_stem/frame_000000.png ... frame_005999.png
        Output: frames_chunked/video_stem.h5

        HDF5 Structure:
          h5file['frames'][0] = np.array([H, W, 3], dtype=uint8)  # frame 0
          h5file['frames'][1] = np.array([H, W, 3], dtype=uint8)  # frame 1
          ...
          h5file.attrs['total_frames'] = 6000
          h5file.attrs['frame_shape'] = (360, 640, 3)

        Args:
            frames_dir: Directory containing frame_*.png files
            output_file: Output HDF5 file path
            frame_height: Expected frame height (default: 360)
            frame_width: Expected frame width (default: 640)
            verbose: Print progress messages

        Returns:
            dict with conversion statistics
        """
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        frame_files = sorted(frames_dir.glob("frame_*.png"))

        if len(frame_files) == 0:
            raise ValueError(f"No frame files found in {frames_dir}")

        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Get actual frame dimensions from first frame
        first_frame = cv2.imread(str(frame_files[0]))
        actual_height, actual_width = first_frame.shape[:2]

        if verbose:
            print(f"Converting {len(frame_files)} frames from {frames_dir.name}")
            print(f"  Frame dimensions: {actual_height}x{actual_width}")
            print(f"  Chunk size: {self.frames_per_chunk} frames")
            print(f"  Compression: {self.compression}")

        # Calculate storage sizes
        png_size = sum(f.stat().st_size for f in frame_files)

        with h5py.File(output_file, 'w') as f:
            dset = f.create_dataset(
                'frames',
                shape=(len(frame_files), actual_height, actual_width, 3),
                dtype=np.uint8,
                chunks=(self.frames_per_chunk, actual_height, actual_width, 3),
                **self._h5_compression_kwargs(),
            )

            buffer = np.empty(
                (self.frames_per_chunk, actual_height, actual_width, 3), dtype=np.uint8
            )
            buffer_fill = 0
            buffer_start = 0

            iterator = enumerate(frame_files)
            if verbose:
                iterator = tqdm(iterator, total=len(frame_files), desc="  Writing frames")

            for _, frame_path in iterator:
                img = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
                buffer[buffer_fill] = img
                buffer_fill += 1
                if buffer_fill == self.frames_per_chunk:
                    dset[buffer_start : buffer_start + buffer_fill] = buffer[:buffer_fill]
                    buffer_start += buffer_fill
                    buffer_fill = 0

            if buffer_fill > 0:
                dset[buffer_start : buffer_start + buffer_fill] = buffer[:buffer_fill]

            # Store metadata
            f.attrs['total_frames'] = len(frame_files)
            f.attrs['frame_height'] = actual_height
            f.attrs['frame_width'] = actual_width
            f.attrs['channels'] = 3
            f.attrs['fps'] = 20
            f.attrs['frames_per_chunk'] = self.frames_per_chunk
            f.attrs['compression'] = self.compression
            f.attrs['source_dir'] = str(frames_dir)

        # Get HDF5 file size
        h5_size = output_file.stat().st_size

        stats = {
            'total_frames': len(frame_files),
            'png_total_size': png_size,
            'h5_file_size': h5_size,
            'compression_ratio': png_size / h5_size if h5_size > 0 else 0,
            'storage_reduction': (1 - h5_size / png_size) * 100 if png_size > 0 else 0,
            'inode_reduction': (1 - 1 / len(frame_files)) * 100 if len(frame_files) > 0 else 0,
        }

        if verbose:
            print(f"  ✓ Conversion complete!")
            print(f"    PNG total: {png_size / 1024**2:.1f} MB ({len(frame_files)} files)")
            print(f"    HDF5 size: {h5_size / 1024**2:.1f} MB (1 file)")
            print(f"    Compression ratio: {stats['compression_ratio']:.2f}x")
            print(f"    Storage reduction: {stats['storage_reduction']:.1f}%")
            print(f"    Inode reduction: {stats['inode_reduction']:.1f}%")

        return stats

    def chunk_all_trajectories(
        self,
        data_dir: Path,
        output_dir: Optional[Path] = None,
        skip_existing: bool = True,
        verbose: bool = True
    ) -> dict:
        """
        Convert all trajectory frames to HDF5 chunks.

        Args:
            data_dir: Root directory containing trajectory_task_* folders
            output_dir: Output directory for HDF5 files (default: data_dir/frames_chunked)
            skip_existing: Skip trajectories that are already converted
            verbose: Print progress messages

        Returns:
            dict with overall statistics
        """
        if output_dir is None:
            output_dir = data_dir / "frames_chunked"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all trajectory directories
        traj_dirs = sorted(data_dir.glob("trajectory_task_*"))

        if len(traj_dirs) == 0:
            raise ValueError(f"No trajectory directories found in {data_dir}")

        if verbose:
            print(f"\n{'='*70}")
            print(f"Frame Chunking: Converting trajectories to HDF5 format")
            print(f"{'='*70}")
            print(f"Input directory: {data_dir}")
            print(f"Output directory: {output_dir}")
            print(f"Found {len(traj_dirs)} trajectory directories")
            print(f"{'='*70}\n")

        overall_stats = {
            'total_trajectories': len(traj_dirs),
            'converted': 0,
            'skipped': 0,
            'failed': 0,
            'total_png_size': 0,
            'total_h5_size': 0,
            'total_frames': 0,
        }

        for traj_dir in traj_dirs:
            frames_dir = traj_dir / "frames"

            if not frames_dir.exists():
                if verbose:
                    print(f"⚠ Skipping {traj_dir.name}: no frames directory")
                overall_stats['skipped'] += 1
                continue

            # Process each video directory
            video_dirs = list(frames_dir.glob("video_*"))

            for video_dir in video_dirs:
                output_file = output_dir / f"{video_dir.name}.h5"

                if skip_existing and output_file.exists():
                    if verbose:
                        print(f"⊘ Skipping {video_dir.name} (already exists)")
                    overall_stats['skipped'] += 1
                    continue

                try:
                    stats = self.chunk_trajectory(video_dir, output_file, verbose=verbose)
                    overall_stats['converted'] += 1
                    overall_stats['total_png_size'] += stats['png_total_size']
                    overall_stats['total_h5_size'] += stats['h5_file_size']
                    overall_stats['total_frames'] += stats['total_frames']
                except Exception as e:
                    if verbose:
                        print(f"✗ Failed to convert {video_dir.name}: {e}")
                    overall_stats['failed'] += 1

        # Calculate overall statistics
        if overall_stats['total_png_size'] > 0:
            overall_stats['compression_ratio'] = (
                overall_stats['total_png_size'] / overall_stats['total_h5_size']
            )
            overall_stats['storage_reduction'] = (
                1 - overall_stats['total_h5_size'] / overall_stats['total_png_size']
            ) * 100

        if verbose:
            print(f"\n{'='*70}")
            print(f"Conversion Summary")
            print(f"{'='*70}")
            print(f"Total trajectories: {overall_stats['total_trajectories']}")
            print(f"  Converted: {overall_stats['converted']}")
            print(f"  Skipped: {overall_stats['skipped']}")
            print(f"  Failed: {overall_stats['failed']}")
            print(f"\nStorage Statistics:")
            print(f"  Total frames: {overall_stats['total_frames']:,}")
            print(f"  PNG total: {overall_stats['total_png_size'] / 1024**3:.2f} GB")
            print(f"  HDF5 total: {overall_stats['total_h5_size'] / 1024**3:.2f} GB")
            print(f"  Compression ratio: {overall_stats.get('compression_ratio', 0):.2f}x")
            print(f"  Storage reduction: {overall_stats.get('storage_reduction', 0):.1f}%")
            print(f"{'='*70}\n")

        return overall_stats


def verify_h5_file(h5_path: Path, sample_frames: int = 5, verbose: bool = True) -> bool:
    """
    Verify HDF5 file integrity and display sample information.

    Args:
        h5_path: Path to HDF5 file
        sample_frames: Number of sample frames to check
        verbose: Print verification details

    Returns:
        True if file is valid, False otherwise
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            # Check required dataset
            if 'frames' not in f:
                if verbose:
                    print(f"✗ Missing 'frames' dataset in {h5_path}")
                return False

            dset = f['frames']
            total_frames = f.attrs.get('total_frames', dset.shape[0])

            if verbose:
                print(f"\n{'='*70}")
                print(f"HDF5 File Verification: {h5_path.name}")
                print(f"{'='*70}")
                print(f"Total frames: {total_frames}")
                print(f"Frame shape: {dset.shape[1:]}")
                print(f"Data type: {dset.dtype}")
                print(f"Chunk size: {dset.chunks}")
                print(f"Compression: {f.attrs.get('compression', 'none')}")
                print(f"File size: {h5_path.stat().st_size / 1024**2:.1f} MB")

                # Sample random frames
                print(f"\nSampling {sample_frames} random frames...")
                indices = np.random.choice(total_frames, min(sample_frames, total_frames), replace=False)
                for idx in sorted(indices):
                    frame = dset[idx]
                    print(f"  Frame {idx}: shape={frame.shape}, "
                          f"min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")

                print(f"{'='*70}\n")
                print(f"✓ File is valid and readable")

            return True

    except Exception as e:
        if verbose:
            print(f"✗ Verification failed for {h5_path}: {e}")
        return False


def main():
    """Command-line interface for frame chunking."""
    parser = argparse.ArgumentParser(
        description="Convert PNG frames to HDF5 chunks for efficient storage"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Root directory containing trajectory_task_* folders'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for HDF5 files (default: data-dir/frames_chunked)'
    )
    parser.add_argument(
        '--frames-per-chunk',
        type=int,
        default=100,
        help='Number of frames per HDF5 chunk (default: 100)'
    )
    parser.add_argument(
        '--compression',
        choices=['gzip', 'none'],
        default='gzip',
        help='Compression algorithm (default: gzip)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip trajectories that are already converted'
    )
    parser.add_argument(
        '--verify',
        type=Path,
        help='Verify a specific HDF5 file instead of converting'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    # Verify mode
    if args.verify:
        verify_h5_file(args.verify, verbose=not args.quiet)
        return

    # Conversion mode
    chunker = FrameChunker(
        frames_per_chunk=args.frames_per_chunk,
        compression=args.compression
    )

    try:
        stats = chunker.chunk_all_trajectories(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
            verbose=not args.quiet
        )

        # Exit with non-zero status if there were failures
        if stats['failed'] > 0:
            exit(1)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
