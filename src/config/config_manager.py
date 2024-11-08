import json
from pathlib import Path
from typing import Dict, Any, Optional
from src.utils.file_manager import FileManager
from src.types.crossword_types import CrosswordConfig

class ConfigManager:
    """Manages configuration loading and validation for crossword video generation"""

    def __init__(self, config: Optional[CrosswordConfig] = None, file_manager: Optional[FileManager] = None):
        self.config = config or CrosswordConfig()
        self.file_manager = file_manager or FileManager()

    def load_json_file(self, filename: str) -> Dict[str, Any]:
        """Load and validate a JSON file"""
        try:
            # Determina il tipo di file e usa il percorso appropriato
            if filename.startswith('template'):
                path = self.file_manager.get_template_path(filename)
            else:
                path = self.file_manager.get_data_path(filename)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading {path}: {str(e)}")

    def load_json_file_by_path(self, filename: str) -> Dict[str, Any]:
        """Load and validate a JSON file"""
        try:
            # Determina il tipo di file e usa il percorso appropriato
            path = self.file_manager.get_data_path(filename)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading {path}: {str(e)}")

    def validate_crossword_data(self, data: Dict[str, Any]) -> bool:
        """Validate crossword data structure for both standard and hidden word types"""
        # Check if metadata exists to determine crossword type
        if 'metadata' in data and data['metadata'].get('type') == 'hidden_word':
            return self._validate_hidden_word_crossword(data)
        return self._validate_standard_crossword(data)

    def _validate_hidden_word_crossword(self, data: Dict[str, Any]) -> bool:
        """Validate hidden word crossword structure"""
        # Validate required sections
        required_sections = ['metadata', 'hidden_word', 'grid', 'words']
        if not all(section in data for section in required_sections):
            missing = [s for s in required_sections if s not in data]
            raise ValueError(f"Missing required sections in hidden word crossword: {missing}")

        # Validate metadata
        required_metadata = ['guid', 'timestamp', 'grid_size', 'cell_size', 'type']
        metadata = data['metadata']
        if not all(field in metadata for field in required_metadata):
            missing = [f for f in required_metadata if f not in metadata]
            raise ValueError(f"Missing required metadata fields: {missing}")

        # Validate hidden word data
        hidden_word = data['hidden_word']
        if not all(field in hidden_word for field in ['word', 'column']):
            raise ValueError("Hidden word data must contain 'word' and 'column'")

        # Validate grid
        if not isinstance(data['grid'], list) or not all(isinstance(row, list) for row in data['grid']):
            raise ValueError("Grid must be a 2D array")

        # Validate words and intersections
        for word in data['words']:
            required_word_fields = ['text', 'x', 'y', 'is_horizontal', 'clue', 'intersection']
            if not all(field in word for field in required_word_fields):
                raise ValueError(f"Word missing required fields: {word}")

            # Validate intersection data
            intersection = word['intersection']
            if not all(field in intersection for field in ['position', 'letter']):
                raise ValueError(f"Invalid intersection data in word: {word}")

        return True

    def _validate_standard_crossword(self, data: Dict[str, Any]) -> bool:
        """Validate standard crossword structure"""
        required_fields = ['grid', 'words']

        if not all(field in data for field in required_fields):
            missing = [f for f in required_fields if f not in data]
            raise ValueError(f"Missing required fields in crossword data: {missing}")

        # Validate grid
        if not isinstance(data['grid'], list) or not all(isinstance(row, list) for row in data['grid']):
            raise ValueError("Grid must be a 2D array")

        # Validate words
        for word in data['words']:
            required_word_fields = ['text', 'x', 'y', 'is_horizontal', 'clue']
            if not all(field in word for field in required_word_fields):
                raise ValueError(f"Word missing required fields: {word}")

        return True

    def validate_template(self, data: Dict[str, Any]) -> bool:
        """Validate template structure"""
        required_fields = ['template_type', 'animation_sequence', 'style_settings']

        if not all(field in data for field in required_fields):
            missing = [f for f in required_fields if f not in data]
            raise ValueError(f"Missing required fields in template: {missing}")

        # Validate animation sequence
        if not isinstance(data['animation_sequence'], list):
            raise ValueError("Animation sequence must be a list")

        return True

    def update_config(self, **kwargs):
        """Update configuration settings"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

    def get_font_path(self) -> Optional[str]:
        """Get font path with validation"""
        if self.config.font_path:
            path = Path(self.config.font_path)
            if not path.exists():
                print(f"Warning: Font file not found at {self.config.font_path}")
                return None
            return str(path)
        return None

    def update_font_path(self):
        """Aggiorna il percorso del font nella configurazione"""
        if self.config.font_path:
            font_filename = Path(self.config.font_path).name
            self.config.font_path = str(self.file_manager.get_font_path(font_filename))