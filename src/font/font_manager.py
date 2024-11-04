from typing import Dict, Any, Optional
from pathlib import Path
from src.utils.file_manager import FileManager
from src.types.crossword_types import FontConfig


class FontManager:
    """Manages font loading and configuration"""

    def __init__(self, file_manager: FileManager):
        """
        Initialize the font manager.

        Args:
            file_manager: FileManager instance for handling file operations
        """
        self.file_manager = file_manager
        self.current_font: Optional[FontConfig] = None

    def load_font_config(self, template_data: Dict[str, Any]) -> FontConfig:
        """
        Load font configuration from template.

        Args:
            template_data: Dictionary containing template configuration

        Returns:
            FontConfig object with loaded configuration

        Raises:
            ValueError: If font configuration is missing or invalid
        """
        if 'fonts' not in template_data:
            raise ValueError("Font configuration not found in template")

        fonts_config = template_data['fonts']
        main_font = fonts_config.get('main')

        if not main_font or 'file' not in main_font:
            # If no main font specified, use fallback
            fallback = fonts_config.get('fallback', 'Arial')
            return self._create_default_config(fallback)

        # Check if font file exists
        font_path = self.file_manager.get_font_path(main_font['file'])
        if not font_path.exists():
            print(f"Warning: Font file {font_path} not found, using fallback font")
            return self._create_default_config(fonts_config.get('fallback', 'Arial'))

        # Create configuration from template
        return FontConfig.from_template(main_font, str(font_path))

    def _create_default_config(self, font_name: str) -> FontConfig:
        """
        Create default font configuration.

        Args:
            font_name: Name of the font to use (system font or file path)

        Returns:
            FontConfig object with default settings
        """
        return FontConfig(
            file_path=font_name,  # For system fonts, use only the name
            clue_size=24,
            clue_color=(0, 0, 0),
            grid_size=46,
            grid_color=(0, 0, 0),
            vertical_adjustment=0,
            horizontal_adjustment=0
        )

    def get_current_font(self) -> Optional[FontConfig]:
        """
        Get the currently loaded font configuration.

        Returns:
            Current FontConfig object or None if no font is loaded
        """
        return self.current_font

    def set_current_font(self, font_config: FontConfig):
        """
        Set the current font configuration.

        Args:
            font_config: FontConfig object to set as current
        """
        self.current_font = font_config

    def validate_font_path(self, font_path: str) -> bool:
        """
        Validate that a font file exists and is accessible.

        Args:
            font_path: Path to the font file

        Returns:
            True if font file exists and is accessible
        """
        path = Path(font_path)
        return path.exists() and path.is_file()