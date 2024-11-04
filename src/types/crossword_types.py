from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

class CrosswordType(Enum):
    """Enumeration of supported crossword types"""
    STANDARD = "standard"
    HIDDEN_WORD = "hidden_word"

@dataclass
class HiddenWordInfo:
    """Information about the hidden word in a hidden word crossword"""
    word: str
    column: int

@dataclass
class WordIntersection:
    """Information about word intersections"""
    position: int
    letter: str

@dataclass
class CrosswordMetadata:
    """Metadata for a crossword puzzle"""
    guid: Optional[str] = None
    timestamp: Optional[str] = None
    grid_size: Optional[int] = None
    cell_size: Optional[int] = None
    crossword_type: CrosswordType = CrosswordType.STANDARD

@dataclass
class CrosswordConfig:
    """Configuration settings for crossword video generation"""
    font_path: Optional[str] = None
    default_cell_size: int = 75
    default_line_spacing: float = 1.5
    default_clue_distance: int = 100

    # Font settings
    clue_font_size: int = 20
    grid_font_size: int = 46

    # Colors (RGB)
    default_text_color: Tuple[int, int, int] = (0, 0, 0)
    default_border_color: Tuple[int, int, int] = (0, 0, 0)
    default_background_color: Tuple[int, int, int] = (255, 255, 255)
    default_highlight_color: Tuple[int, int, int] = (0, 255, 0)

@dataclass
class FontConfig:
    """Font configuration settings"""
    file_path: str
    clue_size: int
    clue_color: Tuple[int, int, int]
    grid_size: int
    grid_color: Tuple[int, int, int]
    vertical_adjustment: int
    horizontal_adjustment: int

    @classmethod
    def from_template(cls, font_settings: dict, file_path: str) -> 'FontConfig':
        """Create a FontConfig instance from template data"""
        settings = font_settings['settings']
        return cls(
            file_path=file_path,
            clue_size=settings['clue']['size'],
            clue_color=tuple(settings['clue']['color']),
            grid_size=settings['grid']['size'],
            grid_color=tuple(settings['grid']['color']),
            vertical_adjustment=settings['grid'].get('vertical_adjustment', 0),
            horizontal_adjustment=settings['grid'].get('horizontal_adjustment', 0)
        )

@dataclass
class TimingInfo:
    """Timing information for animations"""
    start_frame: int
    end_frame: int