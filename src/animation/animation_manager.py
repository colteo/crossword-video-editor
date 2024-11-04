from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from src.types.crossword_types import TimingInfo
from src.utils.file_manager import FileManager
import json
import random


class AnimationType(Enum):
    """Types of animations supported by the system"""
    INITIAL_GRID = "initial_grid"
    SHOW_GRID_EMPTY = "show_grid_empty"
    SHOW_GRID_WORD = "show_grid_word"
    SHOW_CLUE = "show_clue"
    RANDOM_PHRASES = "random_phrases"

@dataclass
class PhraseAnimation:
    """Manages animations related to random phrases"""
    phrase_type: str
    phrase_text: str
    timing: TimingInfo
    style: Dict

@dataclass
class WordAnimation:
    """Manages animations related to a specific word"""
    word_index: int
    animations: List[Dict]


class AnimationManager:
    """Manages animation sequences and timing calculations"""

    def __init__(self, fps: int, file_manager: Optional[FileManager] = None):
        self.fps = fps
        self.file_manager = file_manager or FileManager()
        self._cached_timings: Optional[Dict] = None
        self._cached_phrases: Optional[Dict[str, List[str]]] = None

    def calculate_animation_timings(self, template_data: Dict) -> Dict:
        """
        Pre-calculate animation timings to avoid repeated calculations.

        Args:
            template_data: Template configuration containing animation sequences

        Returns:
            Dictionary mapping frame numbers to animation data
        """
        timings = {}

        for sequence in template_data['animation_sequence']:
            if sequence['type'] == AnimationType.INITIAL_GRID.value:
                start_frame = self._seconds_to_frames(sequence['start'])
                end_frame = self._seconds_to_frames(sequence['end'])
                timings[start_frame] = {
                    'type': AnimationType.INITIAL_GRID.value,
                    'end_frame': end_frame
                }
            elif sequence['type'] == AnimationType.RANDOM_PHRASES.value:
                # Gestione delle frasi random
                for phrase_config in sequence['settings']['phrases_to_show']:
                    start_frame = self._seconds_to_frames(phrase_config['start'])
                    end_frame = self._seconds_to_frames(phrase_config['end'])

                    # Seleziona una frase random del tipo specificato
                    phrase_text = self.get_random_phrase(
                        phrase_config['type'],
                        sequence['settings'].get('data_file', 'phrases.json')
                    )

                    if start_frame not in timings:
                        timings[start_frame] = []

                    timings[start_frame].append({
                        'type': AnimationType.RANDOM_PHRASES.value,
                        'end_frame': end_frame,
                        'phrase_text': phrase_text,
                        'style': {
                            'pos_x': phrase_config.get('pos_x', 'center'),
                            'pos_y': phrase_config.get('pos_y', 100),
                            'max_text_width': phrase_config.get('max_text_width', 800),
                            'text_color': phrase_config.get('text_color', [0, 0, 0]),
                            'text_align': phrase_config.get('text_align', 'center')
                        }
                    })
            elif sequence['type'] == 'word_reveal':
                for word_data in sequence['sequence']:
                    word_index = word_data['word_index']
                    for anim in word_data['animations']:
                        start_frame = self._seconds_to_frames(anim['start'])
                        end_frame = self._seconds_to_frames(anim['end'])
                        if start_frame not in timings:
                            timings[start_frame] = []
                        timings[start_frame].append({
                            'type': anim['type'],
                            'word_index': word_index,
                            'end_frame': end_frame,
                            'data': anim
                        })

        self._cached_timings = timings
        return timings

    def calculate_letter_visibility(self,
                                    frame_number: int,
                                    timing: TimingInfo,
                                    letter_index: int,
                                    total_letters: int,
                                    animation_config: Optional[Dict] = None) -> bool:
        """
        Determine if a letter should be visible based on animation configuration.

        Args:
            frame_number: Current frame number
            timing: Timing information for the animation
            letter_index: Index of the letter in sequence
            total_letters: Total number of letters
            animation_config: Optional animation configuration

        Returns:
            True if letter should be visible, False otherwise
        """
        total_duration = timing.end_frame - timing.start_frame

        # Use default configuration if none provided
        if not animation_config:
            animation_config = {
                "type": "sequential",
                "time_percentage": 100
            }

        animation_type = animation_config.get('type', 'sequential')

        if animation_type == 'sequential':
            return self._calculate_sequential_visibility(
                frame_number, timing.start_frame,
                letter_index, total_letters,
                total_duration, animation_config
            )
        elif animation_type == 'fixed_delay':
            return self._calculate_fixed_delay_visibility(
                frame_number, timing.start_frame,
                letter_index, animation_config
            )
        elif animation_type == 'groups':
            return self._calculate_group_visibility(
                frame_number, timing.start_frame,
                letter_index, total_letters,
                total_duration, animation_config
            )
        elif animation_type == 'instant':
            return frame_number >= timing.start_frame

        return False

    def _calculate_sequential_visibility(self,
                                         frame_number: int,
                                         start_frame: int,
                                         letter_index: int,
                                         total_letters: int,
                                         total_duration: int,
                                         config: Dict) -> bool:
        """Calculate visibility for sequential animation"""
        time_percentage = config.get('time_percentage', 100) / 100
        frames_per_letter = (total_duration * time_percentage) / total_letters
        letter_appears_at = start_frame + (letter_index * frames_per_letter)
        return frame_number >= letter_appears_at

    def _calculate_fixed_delay_visibility(self,
                                          frame_number: int,
                                          start_frame: int,
                                          letter_index: int,
                                          config: Dict) -> bool:
        """Calculate visibility for fixed delay animation"""
        delay_frames = config.get('delay_frames', 3)
        letter_appears_at = start_frame + (letter_index * delay_frames)
        return frame_number >= letter_appears_at

    def _calculate_group_visibility(self,
                                    frame_number: int,
                                    start_frame: int,
                                    letter_index: int,
                                    total_letters: int,
                                    total_duration: int,
                                    config: Dict) -> bool:
        """Calculate visibility for group animation"""
        group_size = config.get('group_size', 2)
        time_percentage = config.get('time_percentage', 30) / 100
        group_index = letter_index // group_size
        total_groups = (total_letters + group_size - 1) // group_size
        frames_per_group = total_duration * time_percentage / total_groups
        letter_appears_at = start_frame + (group_index * frames_per_group)
        return frame_number >= letter_appears_at

    def _seconds_to_frames(self, seconds: float) -> int:
        """Convert seconds to frame number"""
        return int(seconds * self.fps)

    def get_active_animations(self, frame_number: int) -> List[Dict]:
        """
        Get all active animations for the current frame.

        Args:
            frame_number: Current frame number

        Returns:
            List of active animation data
        """
        if not self._cached_timings:
            return []

        active_animations = []

        # Check animations starting at this frame
        if frame_number in self._cached_timings:
            animations = self._cached_timings[frame_number]
            if isinstance(animations, dict):  # initial_grid
                if animations['type'] == AnimationType.INITIAL_GRID.value and \
                        frame_number <= animations['end_frame']:
                    active_animations.append(animations)
            else:  # word_reveal animations
                for anim in animations:
                    if frame_number <= anim['end_frame']:
                        active_animations.append(anim)

        # Check ongoing animations
        for start_frame, animations in self._cached_timings.items():
            if start_frame < frame_number:
                if isinstance(animations, dict):  # initial_grid
                    if animations['type'] == AnimationType.INITIAL_GRID.value and \
                            frame_number <= animations['end_frame']:
                        active_animations.append(animations)
                else:  # word_reveal animations
                    for anim in animations:
                        if frame_number <= anim['end_frame']:
                            active_animations.append(anim)

        return active_animations

    def reset_cache(self):
        """Clear cached animation timings"""
        self._cached_timings = None

    def _load_phrases(self, filename: str) -> Dict[str, List[str]]:
        """Load and cache phrases from JSON file"""
        if self._cached_phrases is None:
            try:
                file_path = self.file_manager.get_data_path(filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Organizziamo le frasi per tipo
                phrases_by_type = {}
                for phrase_data in data['phrases']:
                    phrase_type = phrase_data['type']
                    if phrase_type not in phrases_by_type:
                        phrases_by_type[phrase_type] = []
                    phrases_by_type[phrase_type].append(phrase_data['phrase'])

                self._cached_phrases = phrases_by_type
            except Exception as e:
                raise Exception(f"Error loading phrases from {filename}: {str(e)}")

        return self._cached_phrases

    def get_random_phrase(self, phrase_type: str, filename: str = "phrases.json") -> str:
        """Get a random phrase of specified type"""
        phrases = self._load_phrases(filename)
        if phrase_type not in phrases:
            raise ValueError(f"Invalid phrase type: {phrase_type}")
        return random.choice(phrases[phrase_type])