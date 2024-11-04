import cv2
import numpy as np
from typing import Optional, List, Callable, Tuple
from pathlib import Path
from moviepy.editor import VideoFileClip
import shutil
from src.utils.profiler import profile, profile_section
from src.utils.file_manager import FileManager


class VideoProcessor:
    """Handles video processing operations including batch processing and I/O"""

    def __init__(self, file_manager: FileManager, frame_buffer_size: int = 32):
        """
        Initialize the video processor.

        Args:
            file_manager: FileManager instance for file operations
            frame_buffer_size: Size of the frame processing buffer
        """
        self.file_manager = file_manager
        self.frame_buffer_size = frame_buffer_size

        # Video properties
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.fps: Optional[int] = None
        self.total_frames: Optional[int] = None

    @profile
    def process_video(self,
                      input_video_name: str,
                      frame_processor: Callable[[np.ndarray, int], np.ndarray],
                      output_path: Path,
                      add_audio: bool = True) -> None:
        """
        Process a video file with batch processing and optional audio.

        Args:
            input_video_name: Name of the input video file
            frame_processor: Function that processes each frame
            output_path: Path where to save the output video
            add_audio: Whether to add audio from original video
        """
        input_video_path = self.file_manager.get_input_video_path(input_video_name)
        if not input_video_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        temp_output = self.file_manager.get_temp_file_path("temp_output.mp4")

        cap = None
        out = None

        try:
            with profile_section("Video Open"):
                cap = cv2.VideoCapture(str(input_video_path))
                if not cap.isOpened():
                    raise ValueError(f"Unable to open input video: {input_video_path}")

                self._initialize_video_properties(cap)
                out = self._create_video_writer(str(temp_output))

                self._log_processing_start(output_path)

            self._process_frames(cap, out, frame_processor)

            # Finalize video
            if out is not None:
                out.release()

            if not temp_output.exists():
                raise FileNotFoundError(f"Failed to create temporary video file: {temp_output}")

            # Handle audio if requested
            if add_audio:
                self._add_audio(input_video_path, temp_output, output_path)
            else:
                shutil.copy2(str(temp_output), str(output_path))

        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")

        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()

    def _initialize_video_properties(self, cap: cv2.VideoCapture) -> None:
        """Initialize video properties from capture object"""
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _create_video_writer(self, output_path: str) -> cv2.VideoWriter:
        """Create video writer with current properties"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )

    def _log_processing_start(self, output_path: Path) -> None:
        """Log processing start information"""
        print(f"Starting video processing...")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Buffer size: {self.frame_buffer_size} frames")
        print(f"Output will be saved as: {output_path.name}")

    @profile
    def _process_frames(self,
                        cap: cv2.VideoCapture,
                        out: cv2.VideoWriter,
                        frame_processor: Callable[[np.ndarray, int], np.ndarray]) -> None:
        """
        Process video frames in batches.

        Args:
            cap: OpenCV VideoCapture object
            out: OpenCV VideoWriter object
            frame_processor: Function to process each frame
        """
        frame_number = 0
        frames_processed = 0
        last_progress_update = 0

        while True:
            # Read batch of frames
            with profile_section("Read Frames Batch"):
                frames_batch = self._read_frames_batch(cap)
                if not frames_batch:
                    break

            # Process frames
            with profile_section("Process Frames Batch"):
                processed_frames = [
                    frame_processor(frame.copy(), frame_number + i)
                    for i, frame in enumerate(frames_batch)
                ]
                frame_number += len(frames_batch)

            # Write processed frames
            with profile_section("Write Frames Batch"):
                for frame in processed_frames:
                    out.write(frame)
                    frames_processed += 1

            # Update progress
            self._update_progress(frames_processed)

        print("\nFrame processing completed!")

    def _read_frames_batch(self, cap: cv2.VideoCapture) -> List[np.ndarray]:
        """Read a batch of frames from video"""
        frames = []
        for _ in range(self.frame_buffer_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def _update_progress(self, frames_processed: int) -> None:
        """Update processing progress information"""
        current_time = frames_processed / self.fps
        progress = (frames_processed / self.total_frames) * 100
        print(f"Processed {frames_processed}/{self.total_frames} frames "
              f"({progress:.1f}%) - {current_time:.1f} seconds")

    def _add_audio(self,
                   input_path: Path,
                   temp_path: Path,
                   output_path: Path) -> None:
        """Add audio from original video to processed video"""
        try:
            print("\nCombining video with original audio...")
            original_video = VideoFileClip(str(input_path))
            processed_video = VideoFileClip(str(temp_path))

            # Copy original audio
            final_video = processed_video.set_audio(original_video.audio)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write final video
            print("Writing final video with audio...")
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )

            # Close videos in reverse order
            final_video.close()
            processed_video.close()
            original_video.close()

            print(f"Video successfully saved to: {output_path}")

        except Exception as e:
            print(f"Warning: Error during audio processing: {e}")
            print("Saving video without audio...")
            if temp_path.exists():
                shutil.copy2(str(temp_path), str(output_path))
            else:
                raise FileNotFoundError("Temporary video file not found")
