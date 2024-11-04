import cv2
import numpy as np
from typing import Dict, List, Tuple, Set
from enum import Enum, auto
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import time
import functools
from contextlib import contextmanager
from typing import Dict, Optional
import statistics
import os
from pathlib import Path
import shutil
import tempfile
import uuid
from datetime import datetime

class CrosswordType(Enum):
    STANDARD = "standard"
    HIDDEN_WORD = "hidden_word"

@dataclass
class HiddenWordInfo:
    word: str
    column: int

@dataclass
class WordIntersection:
    position: int
    letter: str

@dataclass
class CrosswordMetadata:
    guid: Optional[str] = None
    timestamp: Optional[str] = None
    grid_size: Optional[int] = None
    cell_size: Optional[int] = None
    crossword_type: CrosswordType = CrosswordType.STANDARD


class FileManager:
    """Gestisce i percorsi dei file e le cartelle del progetto"""

    def __init__(self, base_dir: str = None):
        """
        Inizializza il gestore dei file

        Args:
            base_dir: Directory base del progetto. Se None, usa la directory corrente.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

        # Definisce le cartelle principali
        self.input_dir = self.base_dir / 'input'
        self.output_dir = self.base_dir / 'output'

        # Sottocartelle della directory input
        self.video_input_dir = self.input_dir / 'videos'
        self.templates_dir = self.input_dir / 'templates'
        self.data_dir = self.input_dir / 'data'
        self.fonts_dir = self.input_dir / 'fonts'

        # Crea le cartelle se non esistono
        self._create_directories()

        # Directory temporanea
        self.temp_dir = None

    def generate_output_filename(self,
                                 crossword_type: str,
                                 template_type: str,
                                 extension: str = "mp4",
                                 add_guid: bool = False) -> str:
        """
        Genera un nome file per il video di output usando il formato specificato.

        Args:
            crossword_type: Tipo del cruciverba
            template_type: Tipo del template
            extension: Estensione del file (default: mp4)
            add_guid: Se True, aggiunge un GUID al nome del file

        Returns:
            str: Nome del file formattato
        """
        # Genera la data corrente nel formato YYYYMMDD
        current_date = datetime.now().strftime("%Y%m%d")

        # Pulisce i nomi dei tipi mantenendo il nome completo
        clean_crossword_type = self._clean_type_name(crossword_type)
        clean_template_type = self._clean_type_name(template_type)

        # Costruisce il nome base del file
        filename_parts = [
            current_date,
            clean_crossword_type,
            clean_template_type
        ]

        # Aggiunge un GUID se richiesto
        if add_guid:
            guid = str(uuid.uuid4())[:8]  # Usa solo i primi 8 caratteri del GUID
            filename_parts.append(guid)

        # Unisce le parti con underscore e aggiunge l'estensione
        return f"{'_'.join(filename_parts)}.{extension}"

    def _clean_type_name(self, type_name: str) -> str:
        """
        Pulisce il nome del tipo mantenendo il nome completo.
        Rimuove solo i caratteri non validi per i nomi file.

        Args:
            type_name: Nome del tipo da pulire

        Returns:
            str: Nome pulito
        """
        # Sostituisce eventuali caratteri non validi per i nomi file con underscore
        import re
        # Rimuove caratteri non validi per i nomi file, mantenendo lettere, numeri,
        # trattini, underscore e spazi
        clean_name = re.sub(r'[^\w\-\s]', '', type_name)

        # Sostituisce spazi multipli con singolo underscore
        clean_name = re.sub(r'\s+', '_', clean_name.strip())

        return clean_name

    def _create_directories(self):
        """Crea le cartelle necessarie se non esistono"""
        directories = [
            self.input_dir,
            self.output_dir,
            self.video_input_dir,
            self.templates_dir,
            self.data_dir,
            self.fonts_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def create_temp_dir(self) -> Path:
        """Crea una directory temporanea"""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(dir=self.output_dir))
        return self.temp_dir

    def get_temp_file_path(self, filename: str) -> Path:
        """Ottiene il percorso per un file temporaneo"""
        if self.temp_dir is None:
            self.create_temp_dir()
        return self.temp_dir / filename

    def get_input_video_path(self, filename: str) -> Path:
        """Restituisce il percorso completo per un video di input"""
        return self.video_input_dir / filename

    def get_output_video_path(self, template_data: dict, crossword_data: dict,
                              add_guid: bool = False) -> Path:
        """
        Genera il percorso completo per il file video di output basato sui dati forniti.

        Args:
            template_data: Dati del template
            crossword_data: Dati del cruciverba
            add_guid: Se True, aggiunge un GUID al nome del file

        Returns:
            Path: Percorso completo del file di output
        """
        filename = self.generate_output_filename(
            crossword_type=crossword_data.get('crossword_type', 'unknown'),
            template_type=template_data.get('template_type', 'unknown'),
            add_guid=add_guid
        )
        return self.output_dir / filename

    def get_template_path(self, filename: str) -> Path:
        """Restituisce il percorso completo per un file template"""
        return self.templates_dir / filename

    def get_data_path(self, filename: str) -> Path:
        """Restituisce il percorso completo per un file di dati"""
        return self.data_dir / filename

    def get_font_path(self, filename: str) -> Path:
        """Restituisce il percorso completo per un file font"""
        return self.fonts_dir / filename

    def ensure_temp_dir(self) -> Path:
        """Crea e restituisce il percorso della cartella temporanea"""
        temp_dir = self.output_dir / 'temp'
        temp_dir.mkdir(exist_ok=True)
        return temp_dir

    def cleanup_temp_files(self):
        """Pulisce i file temporanei in modo sicuro"""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(str(self.temp_dir))
                self.temp_dir = None
            except Exception as e:
                print(f"Warning: Error cleaning temporary files: {e}")


class SimpleProfiler:
    """Simple profiler to track execution times"""

    def __init__(self):
        self.times: Dict[str, list] = {}

    def add_time(self, name: str, elapsed: float):
        """Add execution time for a specific section"""
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(elapsed)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked sections"""
        stats = {}
        for name, times in self.times.items():
            if times:
                stats[name] = {
                    'avg': statistics.mean(times),
                    'min': min(times),
                    'max': max(times),
                    'total': sum(times),
                    'calls': len(times)
                }
        return stats

    def print_stats(self):
        """Print formatted statistics"""
        stats = self.get_stats()
        if not stats:
            print("No profiling data available")
            return

        print("\n=== Profiling Results ===")
        # Find longest name for formatting
        max_name = max(len(name) for name in stats.keys())

        # Print header
        header = f"{'Section':<{max_name}} | {'Avg (ms)':>10} | {'Min (ms)':>10} | {'Max (ms)':>10} | {'Total (s)':>10} | {'Calls':>8}"
        print(header)
        print("-" * len(header))

        # Print each section's stats
        for name, data in sorted(stats.items()):
            print(f"{name:<{max_name}} | {data['avg'] * 1000:10.2f} | {data['min'] * 1000:10.2f} | "
                  f"{data['max'] * 1000:10.2f} | {data['total']:10.2f} | {data['calls']:8d}")


# Create a global profiler instance
profiler = SimpleProfiler()


def profile(func=None, section_name: Optional[str] = None):
    """Decorator to profile function execution time"""
    if func is None:
        return lambda f: profile(f, section_name)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = section_name or func.__name__
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time
            profiler.add_time(name, elapsed)

    return wrapper


@contextmanager
def profile_section(name: str):
    """Context manager to profile a code section"""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        profiler.add_time(name, elapsed)

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
    default_text_color: tuple = (0, 0, 0)
    default_border_color: tuple = (0, 0, 0)
    default_background_color: tuple = (255, 255, 255)
    default_highlight_color: tuple = (0, 255, 0)

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

class AnimationType(Enum):
    INITIAL_GRID = "initial_grid"
    SHOW_GRID_EMPTY = "show_grid_empty"
    SHOW_GRID_WORD = "show_grid_word"
    SHOW_CLUE = "show_clue"

class WordAnimation:
    """Classe per gestire le animazioni relative a una parola specifica"""
    def __init__(self, word_index: int, animations: List[Dict]):
        self.word_index = word_index
        self.animations = animations

@dataclass
class TimingInfo:
    """Classe per gestire le informazioni di timing"""
    start_frame: int
    end_frame: int


@dataclass
class FontConfig:
    """Configurazione per un font"""
    file_path: str
    clue_size: int
    clue_color: Tuple[int, int, int]
    grid_size: int
    grid_color: Tuple[int, int, int]
    vertical_adjustment: int
    horizontal_adjustment: int

    @classmethod
    def from_template(cls, font_settings: Dict[str, Any], file_path: str) -> 'FontConfig':
        """Crea una configurazione font dal template"""
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


class FontManager:
    """Gestisce il caricamento e la configurazione dei font"""

    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.current_font: Optional[FontConfig] = None

    def load_font_config(self, template_data: Dict[str, Any]) -> FontConfig:
        """Carica la configurazione del font dal template"""
        if 'fonts' not in template_data:
            raise ValueError("Font configuration not found in template")

        fonts_config = template_data['fonts']
        main_font = fonts_config.get('main')

        if not main_font or 'file' not in main_font:
            # Se non è specificato un font principale, usa il fallback
            fallback = fonts_config.get('fallback', 'Arial')
            return self._create_default_config(fallback)

        # Controlla che il file del font esista
        font_path = self.file_manager.get_font_path(main_font['file'])
        if not font_path.exists():
            print(f"Warning: Font file {font_path} not found, using fallback font")
            return self._create_default_config(fonts_config.get('fallback', 'Arial'))

        # Crea la configurazione dal template
        return FontConfig.from_template(main_font, str(font_path))

    def _create_default_config(self, font_name: str) -> FontConfig:
        """Crea una configurazione di default per il font specificato"""
        return FontConfig(
            file_path=font_name,  # Per font di sistema, usa solo il nome
            clue_size=24,
            clue_color=(0, 0, 0),
            grid_size=46,
            grid_color=(0, 0, 0),
            vertical_adjustment=0,
            horizontal_adjustment=0
        )


class CrosswordVideoGenerator:
    @profile
    def __init__(self, template_data: Dict, crossword_data: Dict,
                 config_manager: Optional[ConfigManager] = None,
                 file_manager: Optional[FileManager] = None):
        """
        Inizializza il generatore del video con la corretta sequenza di inizializzazione

        Args:
            template_data: Dictionary contenente la configurazione del template
            crossword_data: Dictionary contenente i dati del cruciverba
            config_manager: Optional ConfigManager per la configurazione personalizzata
            file_manager: Optional FileManager per la gestione dei file
        """
        # Salva subito i dati di input come attributi della classe
        self.template_data = template_data
        self.crossword_data = crossword_data

        # Inizializza gli attributi base
        self._frame_buffer_size = 32
        self._frame_buffer = []
        self.fps = None
        self.width = None
        self.height = None
        self._animation_timings = None

        # Inizializza i managers
        self.file_manager = file_manager or FileManager()
        self.config_manager = config_manager or ConfigManager()
        self.font_manager = FontManager(self.file_manager)

        # Determina e inizializza il tipo di cruciverba
        self.crossword_type = CrosswordType.HIDDEN_WORD if 'metadata' in crossword_data and \
            crossword_data['metadata'].get('type') == 'hidden_word' else CrosswordType.STANDARD

        # Inizializza metadata
        if 'metadata' in crossword_data:
            metadata = crossword_data['metadata']
            self.metadata = CrosswordMetadata(
                guid=metadata.get('guid'),
                timestamp=metadata.get('timestamp'),
                grid_size=metadata.get('grid_size'),
                cell_size=metadata.get('cell_size'),
                crossword_type=self.crossword_type
            )
        else:
            # Metadata di default per cruciverba standard
            self.metadata = CrosswordMetadata(
                crossword_type=self.crossword_type
            )

        # Inizializza hidden word info se necessario
        self.hidden_word_info = None
        if self.crossword_type == CrosswordType.HIDDEN_WORD and 'hidden_word' in crossword_data:
            self.hidden_word_info = HiddenWordInfo(
                word=crossword_data['hidden_word']['word'],
                column=crossword_data['hidden_word']['column']
            )

        # Inizializza le cache come attributi vuoti
        self._cell_cache = {}
        self._letter_cache = {}
        self._composite_cache = {}
        self._cell_positions = {}
        self._cached_colors = {}

        # Salva i dati di input elaborati
        with profile_section("Init Config"):
            self.grid = np.array(crossword_data['grid'])
            self.words = crossword_data['words']
            # Nota: usiamo direttamente template_data invece di salvarlo come self.template
            self.style = template_data['style_settings']['crossword']
            self.layout = template_data['layout']

            # Carica la configurazione del font dal template
            self.font_config = self.font_manager.load_font_config(template_data)

        # Inizializza i parametri di testo
        with profile_section("Init Text Params"):
            self.max_text_width = self.style.get('max_text_width', 500)
            self.line_spacing = self.style.get('line_spacing',
                                             self.config_manager.config.default_line_spacing)
            # Inizializza i font
            self._initialize_fonts()

        # Inizializza le dimensioni e i parametri della griglia
        with profile_section("Init Grid"):
            self.valid_cells = self._get_valid_cells()
            self.bounds = self._calculate_bounds()
            min_x, min_y, max_x, max_y = self.bounds
            self.grid_width = (max_x - min_x + 1) * self._get_cell_size()
            self.grid_height = (max_y - min_y + 1) * self._get_cell_size()

        # Ora possiamo inizializzare le cache
        with profile_section("Init Cache"):
            self._initialize_cache()

            # Calcola le posizioni delle celle
            for y, x in self.valid_cells:
                min_x, min_y, _, _ = self.bounds
                cell_size = self._get_cell_size()
                self._cell_positions[(y, x)] = (
                    (y - min_y) * cell_size,  # py
                    (x - min_x) * cell_size  # px
                )

        # Parametri di debug e profiling
        self.debug_mode = False
        self.profile_enabled = True

        if self.debug_mode:
            print("Initialization completed:")
            print(f"Grid size: {self.grid_width}x{self.grid_height}")
            print(f"Cell cache entries: {len(self._cell_cache)}")
            print(f"Composite cache entries: {len(self._composite_cache)}")
            print(f"Crossword type: {self.crossword_type}")
            if self.hidden_word_info:
                print(f"Hidden word: {self.hidden_word_info.word}")

    def _get_cell_size(self) -> int:
        """Get cell size with support for metadata override"""
        if self.metadata.cell_size is not None:
            return self.metadata.cell_size
        return self.style.get('cell_size', self.config_manager.config.default_cell_size)

    def _process_hidden_word_highlight(self, frame: np.ndarray, word_index: int,
                                       highlight_intersection: bool = False):
        """Process highlighting for hidden word crosswords"""
        if self.crossword_type != CrosswordType.HIDDEN_WORD:
            return

        word = self.words[word_index]
        if 'intersection' not in word:
            return

        intersection = word['intersection']
        # Highlight the intersection cell
        if highlight_intersection:
            cell_size = self._get_cell_size()
            x = self.hidden_word_info.column * cell_size
            y = word['y'] * cell_size

            # Create highlight overlay
            highlight_color = self.style.get('intersection_highlight_color',
                                             self.style.get('highlight_color', (255, 255, 0)))

            overlay = np.zeros((cell_size, cell_size, 4), dtype=np.uint8)
            overlay[:, :, :3] = highlight_color
            overlay[:, :, 3] = 128  # Semi-transparent

            self._overlay_image(frame, overlay, (x, y))

    def _initialize_fonts(self):
        """Inizializza i font usando la configurazione dal template"""
        try:
            # Carica il font principale
            if os.path.isfile(self.font_config.file_path):
                self.clue_font = ImageFont.truetype(
                    self.font_config.file_path,
                    self.font_config.clue_size
                )
                self.grid_font = ImageFont.truetype(
                    self.font_config.file_path,
                    self.font_config.grid_size
                )
                print(f"Loaded font: {self.font_config.file_path}")
            else:
                # Per font di sistema
                print(f"Using system font: {self.font_config.file_path}")
                default_font = ImageFont.load_default()
                self.clue_font = default_font
                self.grid_font = default_font

        except Exception as e:
            print(f"Font initialization error: {str(e)}")
            default_font = ImageFont.load_default()
            self.clue_font = default_font
            self.grid_font = default_font

    def _get_style_color(self, key: str, default_key: str) -> tuple:
        """Get color from style settings or default configuration"""
        color = self.style.get(key)
        if color is None:
            return getattr(self.config_manager.config, default_key)
        return tuple(color)

    def _get_font_height(self, font: ImageFont.FreeTypeFont) -> int:
        """
        Ottiene l'altezza del font usando get_bbox
        """
        bbox = font.getbbox("Aj")  # Usa lettere alte e basse per ottenere l'altezza completa
        return bbox[3] - bbox[1]

    def _create_clue_overlay(self, clue_text: str, pattern: Dict) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crea l'overlay per l'indizio usando PIL per il rendering del font

        Args:
            clue_text: Testo dell'indizio
            pattern: Dictionary con le impostazioni di stile e posizionamento

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: Overlay dell'indizio e le sue dimensioni
        """
        # Converti il testo in maiuscolo
        clue_text = clue_text.upper()

        # Usa i colori dalla configurazione del font
        text_color = pattern.get('text_color', self.font_config.clue_color)

        padding = pattern.get('padding', 20)
        max_width = pattern.get('max_text_width', self.max_text_width)
        line_spacing = pattern.get('line_spacing', self.line_spacing)
        text_color = pattern.get('text_color', (0, 0, 0))
        # Aggiungi supporto per l'allineamento, default a 'left'
        text_align = pattern.get('text_align', 'left')

        # Crea un'immagine temporanea per misurare il testo
        temp_img = Image.new('RGBA', (max_width + padding * 2, 1000), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)

        # Wrapping del testo
        words = clue_text.split()
        lines = []
        current_line = []
        current_width = 0

        for word in words:
            word_width = draw.textlength(word, font=self.clue_font)
            space_width = draw.textlength(" ", font=self.clue_font)

            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width + space_width
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width + space_width

        if current_line:
            lines.append(" ".join(current_line))

        # Calcola l'altezza totale necessaria
        line_height = self._get_font_height(self.clue_font)
        total_height = len(lines) * line_height * line_spacing

        # Crea l'immagine finale con le dimensioni corrette
        img = Image.new('RGBA',
                        (max_width + padding * 2,
                         int(total_height) + padding * 2),
                        (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Disegna il testo con l'allineamento specificato
        y = padding
        for line in lines:
            # Calcola la posizione x in base all'allineamento
            line_width = draw.textlength(line, font=self.clue_font)
            if text_align == 'center':
                x = (max_width - line_width) / 2 + padding
            elif text_align == 'right':
                x = max_width - line_width + padding
            else:  # 'left' o qualsiasi altro valore
                x = padding

            draw.text((x, y), line,
                      font=self.clue_font,
                      fill=(*text_color, 255))
            y += line_height * line_spacing

        # Converti l'immagine PIL in array numpy per OpenCV
        overlay = np.array(img)

        return overlay, img.size

    def _get_clue_position(self, pattern: Dict, clue_size: Tuple[int, int], frame_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calcola la posizione dell'indizio basandosi sulle coordinate specificate nel pattern

        Args:
            pattern: Pattern di animazione con le coordinate di posizionamento
            clue_size: Dimensioni dell'overlay dell'indizio (width, height)
            frame_size: Dimensioni del frame (width, height)

        Returns:
            Tuple[int, int]: Coordinate (x, y) per il posizionamento dell'indizio
        """
        clue_width, clue_height = clue_size
        frame_width, frame_height = frame_size

        # Gestisce il posizionamento orizzontale
        if isinstance(pattern.get('pos_x'), str) and pattern['pos_x'] == 'center':
            x = (frame_width - clue_width) // 2
        else:
            try:
                # Usa la coordinata x specificata
                x = int(pattern['pos_x'])
                # Assicura che l'indizio non esca dal frame
                x = max(0, min(x, frame_width - clue_width))
            except (ValueError, KeyError):
                # Fallback al centro se c'è un errore
                x = (frame_width - clue_width) // 2

        # Gestisce il posizionamento verticale
        # Prima controlla pos_y, poi pos_v per retrocompatibilità
        vertical_pos = pattern.get('pos_y', pattern.get('pos_v'))

        if isinstance(vertical_pos, str) and vertical_pos == 'center':
            y = (frame_height - clue_height) // 2
        else:
            try:
                # Usa la coordinata verticale specificata
                y = int(vertical_pos)
                # Assicura che l'indizio non esca dal frame
                y = max(0, min(y, frame_height - clue_height))
            except (ValueError, TypeError):
                # Fallback: posiziona vicino al fondo del frame
                y = frame_height - clue_height - self.layout.get('clue_distance_from_bottom', 100)

        return (x, y)

    def _get_word_cells(self, word: Dict) -> Set[Tuple[int, int]]:
        """
        Ottiene le coordinate delle celle per una specifica parola

        Args:
            word: Dizionario contenente le informazioni della parola
                (deve contenere 'x', 'y', 'text', 'is_horizontal')

        Returns:
            Set di tuple (y, x) contenenti le coordinate delle celle della parola
        """
        cells = set()
        x, y = word['x'], word['y']
        text = word['text']
        is_horizontal = word['is_horizontal']

        for i, _ in enumerate(text):
            if is_horizontal:
                cells.add((y, x + i))
            else:
                cells.add((y + i, x))

        return cells

    def _seconds_to_frames(self, seconds: float, fps: int) -> int:
        """Converte secondi in frames"""
        return int(seconds * fps)

    def _is_frame_in_timing(self, frame_number: int, timing: TimingInfo) -> bool:
        """Verifica se il frame corrente è all'interno del timing specificato"""
        return timing.start_frame <= frame_number < timing.end_frame

    def _get_valid_cells(self) -> Set[Tuple[int, int]]:
        """Identifica le celle valide nel cruciverba"""
        valid_cells = set()
        for word in self.words:
            x, y = word['x'], word['y']
            text = word['text']
            is_horizontal = word['is_horizontal']

            for i, _ in enumerate(text):
                if is_horizontal:
                    valid_cells.add((y, x + i))
                else:
                    valid_cells.add((y + i, x))
        return valid_cells

    def _calculate_bounds(self) -> Tuple[int, int, int, int]:
        """Calcola i limiti della griglia"""
        if not self.valid_cells:
            return (0, 0, 0, 0)

        cells = list(self.valid_cells)
        min_y = min(y for y, x in cells)
        max_y = max(y for y, x in cells)
        min_x = min(x for y, x in cells)
        max_x = max(x for y, x in cells)
        return (min_x, min_y, max_x, max_y)

    @profile
    def _create_grid_overlay(self, highlight_word_index: int = None,
                             show_letters: bool = False,
                             is_initial: bool = False,
                             frame_number: int = None,
                             timing: TimingInfo = None,
                             letter_animation_config: dict = None,
                             show_intersections: bool = False) -> np.ndarray:
        """
        Create grid overlay with support for both standard and hidden word crosswords.
        """
        cell_size = self._get_cell_size()
        grid_width = (self.bounds[2] - self.bounds[0] + 1) * cell_size
        grid_height = (self.bounds[3] - self.bounds[1] + 1) * cell_size
        overlay = np.zeros((grid_height, grid_width, 4), dtype=np.uint8)

        # Determina se è un cruciverba con parola nascosta e ottiene la colonna
        is_hidden_word = self.crossword_type == CrosswordType.HIDDEN_WORD
        hidden_column = None
        if is_hidden_word and self.hidden_word_info:
            hidden_column = self.hidden_word_info.column - self.bounds[0]

        # Calcola le celle da evidenziare
        highlighted_cells = set()
        if highlight_word_index is not None:
            highlighted_cells = self._get_word_cells(self.words[highlight_word_index])

        # Calcola le celle rivelate
        revealed_cells = set()
        if not is_initial and highlight_word_index is not None:
            for i in range(highlight_word_index):
                revealed_cells.update(self._get_word_cells(self.words[i]))

        # Calcola le lettere visibili per l'animazione corrente
        visible_letters = set()
        if highlight_word_index is not None and show_letters and frame_number is not None and timing is not None:
            current_word_letters = self._get_word_letters_sequence(self.words[highlight_word_index])
            for i, (ly, lx, _) in enumerate(current_word_letters):
                if self._calculate_letter_visibility(frame_number, timing, i,
                                                     len(current_word_letters),
                                                     letter_animation_config):
                    visible_letters.add((ly, lx))

        # Processa tutte le celle valide
        min_x, min_y, _, _ = self.bounds
        for y, x in self.valid_cells:
            rel_y = y - min_y
            rel_x = x - min_x
            px = rel_x * cell_size
            py = rel_y * cell_size

            letter = self.grid[y][x]
            is_highlighted = (y, x) in highlighted_cells
            should_show_letter = (y, x) in revealed_cells or (y, x) in visible_letters
            is_hidden_column = is_hidden_word and (rel_x == hidden_column)

            if is_hidden_column:
                # Usa le celle speciali per la colonna nascosta
                if is_initial:
                    cell = self._cell_cache['hidden_initial_6']
                else:
                    border_type = 'highlight' if is_highlighted else 'inactive'
                    if should_show_letter and letter != '_':
                        cell = self._composite_cache[f'hidden_{border_type}_{letter}']
                    else:
                        cell = self._cell_cache[f'hidden_{border_type}_6']
            else:
                # Usa le celle normali per il resto della griglia
                if is_initial:
                    cell = self._cell_cache['initial_6']
                else:
                    border_type = 'highlight' if is_highlighted else 'inactive'
                    if should_show_letter and letter != '_':
                        cell = self._composite_cache[f'{border_type}_{letter}']
                    else:
                        cell = self._cell_cache[f'{border_type}_6']

            if cell is not None:
                overlay[py:py + cell_size, px:px + cell_size] = cell

        return overlay

    def _create_intersection_cell_cache(self):
        """
        Crea la cache per le celle di intersezione con stili specifici
        """
        cell_size = self._get_cell_size()
        intersection_color = self.style.get('intersection_color', (255, 255, 0))  # Default giallo

        for base_type in ['inactive', 'highlight']:
            # Crea una nuova immagine per la cella di intersezione
            cell_img = Image.new('RGBA', (cell_size, cell_size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(cell_img)

            # Colore del bordo base
            border_color = {
                'inactive': self._cached_colors['inactive'],
                'highlight': self._cached_colors['highlight']
            }[base_type]

            # Disegna il bordo base
            thickness = 6
            half_thickness = thickness / 2
            borders = [
                half_thickness,  # left
                half_thickness,  # top
                cell_size - half_thickness,  # right
                cell_size - half_thickness  # bottom
            ]

            draw.rectangle(
                borders,
                outline=tuple(border_color[:3]),
                width=thickness
            )

            # Aggiungi l'evidenziazione per l'intersezione
            padding = thickness + 2
            inner_borders = [
                padding,  # left
                padding,  # top
                cell_size - padding,  # right
                cell_size - padding  # bottom
            ]

            # Disegna un rettangolo semi-trasparente per l'evidenziazione
            draw.rectangle(
                inner_borders,
                fill=(*intersection_color, 64),  # Alpha 64 per semi-trasparenza
                outline=None
            )

            # Salva nella cache
            key = f'intersection_{base_type}_6'
            self._cell_cache[key] = np.array(cell_img)

    def _calculate_positions(self, frame_width: int, frame_height: int) -> Dict[str, Tuple[int, int]]:
        """Calcola le posizioni degli elementi nel frame"""
        # Calcola la posizione centrale per il cruciverba
        grid_x = (frame_width - self.grid_width) // 2
        grid_y = (frame_height - self.grid_height) // 2

        return {
            'grid': (grid_x, grid_y)
        }

    def _overlay_image(self, background: np.ndarray, overlay: np.ndarray,
                       position: Tuple[int, int]):
        """Sovrappone un'immagine RGBA su uno sfondo"""
        x, y = position
        h, w = overlay.shape[:2]

        if y + h > background.shape[0] or x + w > background.shape[1]:
            return

        alpha_overlay = overlay[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay

        for c in range(3):
            background[y:y + h, x:x + w, c] = (alpha_overlay * overlay[:, :, c] +
                                               alpha_background * background[y:y + h, x:x + w, c])

    @profile
    def process_video(self, input_video_name: str):
        """
        Process video with optimized frame handling and detailed profiling

        Args:
            input_video_name: Nome del file video di input
        """
        # Genera il nome del file di output basato sui dati del template e del crossword
        output_path = self.file_manager.get_output_video_path(
            template_data=self.template_data,
            crossword_data=self.crossword_data,
            add_guid=True
        )

        input_video_path = self.file_manager.get_input_video_path(input_video_name)

        if not input_video_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        # Crea un file temporaneo con nome univoco
        temp_output = self.file_manager.get_temp_file_path("temp_output.mp4")

        cap = None
        out = None

        try:
            # Apertura video e inizializzazione
            with profile_section("Video Open"):
                cap = cv2.VideoCapture(str(input_video_path))
                if not cap.isOpened():
                    raise ValueError(f"Unable to open input video: {input_video_path}")

                # Ottieni i parametri del video
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Prepara il file di output temporaneo
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(temp_output), fourcc, self.fps,
                                      (self.width, self.height))

                # Dimensione del buffer per il batch processing
                buffer_size = self._frame_buffer_size

                print(f"Starting video processing...")
                print(f"Total frames: {total_frames}")
                print(f"FPS: {self.fps}")
                print(f"Resolution: {self.width}x{self.height}")
                print(f"Buffer size: {buffer_size} frames")
                print(f"Output will be saved as: {output_path.name}")

            try:
                # Pre-calcola i timing delle animazioni
                with profile_section("Timing Precalculation"):
                    animation_timings = self._precalculate_animation_timings()

                frame_number = 0
                frames_processed = 0
                last_progress_update = 0

                with profile_section("Frame Processing"):
                    while True:
                        # Leggi un batch di frame
                        with profile_section("Read Frames Batch"):
                            frames_batch = []
                            for _ in range(buffer_size):
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                frames_batch.append(frame)

                        if not frames_batch:
                            break

                        # Processa il batch di frame
                        processed_frames = []
                        with profile_section("Process Frames Batch"):
                            for frame in frames_batch:
                                result = frame.copy()
                                self._process_frame_with_cached_timings(
                                    result, frame_number, animation_timings)
                                processed_frames.append(result)
                                frame_number += 1

                        # Scrivi il batch di frame processati
                        with profile_section("Write Frames Batch"):
                            for frame in processed_frames:
                                out.write(frame)
                                frames_processed += 1

                        # Aggiorna il progresso ogni secondo
                        current_time = frames_processed / self.fps
                        if current_time - last_progress_update >= 1.0:
                            progress = (frames_processed / total_frames) * 100
                            print(f"Processed {frames_processed}/{total_frames} frames "
                                  f"({progress:.1f}%) - "
                                  f"{current_time:.1f} seconds")
                            last_progress_update = current_time

                print("\nFrame processing completed!")

                # Chiudi i writer
                if out is not None:
                    out.release()

                # Verifica che il file temporaneo sia stato creato
                if not temp_output.exists():
                    raise FileNotFoundError(f"Failed to create temporary video file: {temp_output}")

                # Gestione dell'audio e finalizzazione
                with profile_section("Audio Processing"):
                    print("\nCombining video with original audio...")
                    try:
                        # Carica i video
                        original_video = VideoFileClip(str(input_video_path))
                        processed_video = VideoFileClip(str(temp_output))

                        # Copia l'audio originale
                        final_video = processed_video.set_audio(original_video.audio)

                        # Assicurati che la directory di output esista
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        # Scrivi il video finale
                        print("Writing final video with audio...")
                        final_video.write_videofile(
                            str(output_path),
                            codec='libx264',
                            audio_codec='aac',
                            verbose=False,
                            logger=None
                        )

                        # Chiudi i video in ordine inverso rispetto all'apertura
                        final_video.close()
                        processed_video.close()
                        original_video.close()

                        print(f"Video successfully saved to: {output_path}")

                    except Exception as e:
                        print(f"Warning: Error during audio processing: {e}")
                        print("Saving video without audio...")
                        if temp_output.exists():
                            shutil.copy2(str(temp_output), str(output_path))
                        else:
                            raise FileNotFoundError("Temporary video file not found")

            except Exception as e:
                raise Exception(f"Error during frame processing: {str(e)}")

        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")

        finally:
            # Cleanup
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()

            # Pulisci i file temporanei
            self.file_manager.cleanup_temp_files()

            print("\nProcessing completed!")
            if profiler:
                profiler.print_stats()

    def _get_word_letters_sequence(self, word: Dict) -> List[Tuple[int, int, str]]:
        """
        Ottiene la sequenza di lettere per una parola con le loro coordinate

        Args:
            word: Dizionario contenente le informazioni della parola

        Returns:
            Lista di tuple (y, x, lettera) in ordine di apparizione
        """
        sequence = []
        x, y = word['x'], word['y']
        text = word['text']
        is_horizontal = word['is_horizontal']

        for i, letter in enumerate(text):
            if letter != '_':
                if is_horizontal:
                    sequence.append((y, x + i, letter))
                else:
                    sequence.append((y + i, x, letter))

        return sequence

    def _calculate_letter_visibility(self, frame_number: int, timing: TimingInfo,
                                     letter_index: int, total_letters: int,
                                     animation_config: dict = None) -> bool:
        """
        Determina se una lettera deve essere visibile in base alla configurazione dell'animazione
        """
        total_duration = timing.end_frame - timing.start_frame

        # Usa configurazione default se non specificata
        if not animation_config:
            animation_config = {
                "type": "sequential",
                "time_percentage": 100
            }

        animation_type = animation_config.get('type', 'sequential')

        if animation_type == 'sequential':
            time_percentage = animation_config.get('time_percentage', 100) / 100
            frames_per_letter = (total_duration * time_percentage) / total_letters
            letter_appears_at = timing.start_frame + (letter_index * frames_per_letter)

        elif animation_type == 'fixed_delay':
            delay_frames = animation_config.get('delay_frames', 3)
            letter_appears_at = timing.start_frame + (letter_index * delay_frames)

        elif animation_type == 'groups':
            group_size = animation_config.get('group_size', 2)
            time_percentage = animation_config.get('time_percentage', 30) / 100
            group_index = letter_index // group_size
            frames_per_group = total_duration * time_percentage / ((total_letters + group_size - 1) // group_size)
            letter_appears_at = timing.start_frame + (group_index * frames_per_group)

        elif animation_type == 'instant':
            letter_appears_at = timing.start_frame

        return frame_number >= letter_appears_at

    def _apply_initial_grid(self, frame: np.ndarray):
        """
        Applica l'animazione della griglia iniziale
        """
        grid_overlay = self._create_grid_overlay(is_initial=True)
        positions = self._calculate_positions(frame.shape[1], frame.shape[0])
        self._overlay_image(frame, grid_overlay, positions['grid'])

    @profile
    def _process_animation_sequence(self, frame: np.ndarray, sequence: Dict, frame_number: int):
        """Process animation sequence with profiling"""
        with profile_section("Animation Sequence"):
            if sequence['type'] == 'initial_grid':
                timing = TimingInfo(
                    self._seconds_to_frames(sequence['start'], self.fps),
                    self._seconds_to_frames(sequence['end'], self.fps)
                )
                if self._is_frame_in_timing(frame_number, timing):
                    self._apply_initial_grid(frame)

            elif sequence['type'] == 'word_reveal':
                for word_data in sequence['sequence']:
                    word_index = word_data['word_index']
                    for anim in word_data['animations']:
                        with profile_section("Word Animation"):
                            timing = TimingInfo(
                                self._seconds_to_frames(anim['start'], self.fps),
                                self._seconds_to_frames(anim['end'], self.fps)
                            )
                            if self._is_frame_in_timing(frame_number, timing):
                                self._apply_word_animation(frame, word_index,
                                                           anim, timing, frame_number)

    def _apply_word_animation(self, frame: np.ndarray,
                              word_index: int,
                              animation: Dict,
                              timing: TimingInfo,
                              frame_number: int):
        """
        Applica una singola animazione di una parola, supportando sia cruciverba standard che hidden word.

        Args:
            frame: Il frame video da modificare
            word_index: Indice della parola corrente
            animation: Dizionario con i dettagli dell'animazione
            timing: Informazioni sul timing dell'animazione
            frame_number: Numero del frame corrente
        """
        anim_type = animation['type']
        is_hidden_word = self.crossword_type == CrosswordType.HIDDEN_WORD

        if anim_type == 'show_grid_empty':
            # Per hidden word, possiamo mostrare anche l'intersezione durante l'highlight
            show_intersection = is_hidden_word and animation.get('show_intersection', False)
            grid_overlay = self._create_grid_overlay(
                highlight_word_index=word_index,
                show_letters=False,
                is_initial=False,
                show_intersections=show_intersection
            )
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        elif anim_type == 'show_grid_word':
            # Configurazione animazione lettere
            letter_animation_config = animation.get('letter_animation')
            show_intersection = is_hidden_word and animation.get('show_intersection', True)

            # Per hidden word, possiamo avere animazioni speciali per l'intersezione
            if is_hidden_word and animation.get('intersection_animation'):
                intersection_config = animation['intersection_animation']
                # Calcola se l'intersezione deve essere mostrata in questo frame
                current_progress = (frame_number - timing.start_frame) / (timing.end_frame - timing.start_frame)
                show_intersection = current_progress >= intersection_config.get('start_percentage', 0.0)

            grid_overlay = self._create_grid_overlay(
                highlight_word_index=word_index,
                show_letters=True,
                is_initial=False,
                frame_number=frame_number,
                timing=timing,
                letter_animation_config=letter_animation_config,
                show_intersections=show_intersection
            )
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        elif anim_type == 'show_clue':
            # Mostra l'indizio come nel cruciverba standard
            clue_text = self.words[word_index]['clue']
            clue_overlay, clue_size = self._create_clue_overlay(clue_text, animation)
            position = self._get_clue_position(animation, clue_size,
                                               (frame.shape[1], frame.shape[0]))
            self._overlay_image(frame, clue_overlay, position)

        elif anim_type == 'highlight_intersection' and is_hidden_word:
            # Animazione speciale solo per hidden word - evidenzia l'intersezione
            word = self.words[word_index]
            if 'intersection' in word:
                intersection = word['intersection']
                cell_size = self._get_cell_size()

                # Calcola la posizione dell'intersezione
                x = self.hidden_word_info.column * cell_size
                y = word['y'] * cell_size

                # Crea un overlay per l'evidenziazione
                highlight_color = animation.get('highlight_color',
                                                self.style.get('intersection_highlight_color',
                                                               (255, 255, 0)))  # Default giallo

                overlay = np.zeros((cell_size, cell_size, 4), dtype=np.uint8)
                overlay[:, :, :3] = highlight_color

                # Calcola l'opacità basata sul progresso dell'animazione
                progress = (frame_number - timing.start_frame) / (timing.end_frame - timing.start_frame)
                max_alpha = animation.get('max_alpha', 180)
                min_alpha = animation.get('min_alpha', 40)

                if animation.get('pulse', False):
                    # Effetto pulsante
                    alpha = min_alpha + (max_alpha - min_alpha) * (np.sin(progress * 2 * np.pi) + 1) / 2
                else:
                    # Fade in/out normale
                    alpha = min_alpha + (max_alpha - min_alpha) * progress

                overlay[:, :, 3] = int(alpha)

                # Applica l'overlay
                positions = self._calculate_positions(frame.shape[1], frame.shape[0])
                grid_x, grid_y = positions['grid']
                self._overlay_image(frame, overlay, (grid_x + x, grid_y + y))

        elif anim_type == 'reveal_hidden_letter' and is_hidden_word:
            # Animazione per rivelare una lettera della parola nascosta
            word = self.words[word_index]
            if 'intersection' in word:
                intersection = word['intersection']
                letter = intersection['letter']

                # Crea l'overlay per la lettera
                cell_size = self._get_cell_size()
                cell_img = Image.new('RGBA', (cell_size, cell_size), (0, 0, 0, 0))
                draw = ImageDraw.Draw(cell_img)

                # Calcola la posizione della lettera
                text_bbox = self.grid_font.getbbox(letter)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                x = (cell_size - text_width) // 2 + self.font_config.horizontal_adjustment
                y = (cell_size - text_height) // 2 + self.font_config.vertical_adjustment

                # Calcola l'opacità basata sul progresso
                progress = (frame_number - timing.start_frame) / (timing.end_frame - timing.start_frame)
                alpha = int(255 * progress)

                # Disegna la lettera con l'opacità calcolata
                draw.text((x, y), letter,
                          font=self.grid_font,
                          fill=(*self.font_config.grid_color, alpha))

                # Converti in array numpy e applica
                letter_overlay = np.array(cell_img)

                # Calcola la posizione nella griglia
                positions = self._calculate_positions(frame.shape[1], frame.shape[0])
                grid_x, grid_y = positions['grid']
                x = self.hidden_word_info.column * cell_size
                y = word['y'] * cell_size

                self._overlay_image(frame, letter_overlay, (grid_x + x, grid_y + y))

    def _initialize_cache(self):
        """Inizializza il sistema di cache per celle e lettere"""
        cell_size = self._get_cell_size()

        # Ottieni i colori dal template o usa i default
        bg_color = self._get_style_color('background_color', 'default_background_color')
        border_color = self._get_style_color('border_color', 'default_border_color')
        highlight_color = self._get_style_color('highlight_color', 'default_highlight_color')
        inactive_color = (128, 128, 128)  # Colore per celle non attive
        text_color = self.font_config.grid_color

        # Pre-calcola i colori più usati come array numpy
        self._cached_colors = {
            'bg': np.array([255, 255, 255, 255], dtype=np.uint8),
            'border': np.array([0, 0, 0, 255], dtype=np.uint8),
            'text': np.array(self.font_config.grid_color + (255,), dtype=np.uint8),
            'highlight': np.array(self._get_style_color('highlight_color', 'default_highlight_color') + (255,), dtype=np.uint8),
            'inactive': np.array([128, 128, 128, 255], dtype=np.uint8),
            'hidden_column_border': np.array([0, 0, 0, 255], dtype=np.uint8),  # Gold
            'hidden_column_bg': np.array([0, 191, 255, 255], dtype=np.uint8)  # Deep Sky Blue
        }

        # Crea cache speciale per le celle della colonna nascosta
        self._create_hidden_column_cells(cell_size)

        # Crea celle base con diversi bordi e spessori
        border_types = ['initial', 'inactive', 'highlight']
        thicknesses = [6]  # Puoi aggiungere altri spessori se necessario

        for border_type in border_types:
            for thickness in thicknesses:
                # Seleziona il colore del bordo in base al tipo
                current_border_color = {
                    'initial': border_color,
                    'inactive': inactive_color,
                    'highlight': highlight_color
                }[border_type]

                # Crea l'immagine base della cella
                cell_img = Image.new('RGBA', (cell_size, cell_size), (*bg_color, 255))
                draw = ImageDraw.Draw(cell_img)

                # Disegna il bordo
                half_thickness = thickness / 2
                borders = [
                    half_thickness,  # left
                    half_thickness,  # top
                    cell_size - half_thickness,  # right
                    cell_size - half_thickness  # bottom
                ]

                draw.rectangle(
                    borders,
                    outline=(*current_border_color, 255),
                    width=thickness
                )

                # Salva la cella base nella cache
                cell_array = np.array(cell_img)
                key = f'{border_type}_{thickness}'
                self._cell_cache[key] = cell_array

                # Se non è una cella iniziale, prepara anche le versioni con lettere
                if border_type != 'initial':
                    # Ottieni le configurazioni per il posizionamento del testo
                    vertical_adj = self.font_config.vertical_adjustment
                    horizontal_adj = self.font_config.horizontal_adjustment

                    effective_cell_size = cell_size - (thickness * 2)

                    # Crea le versioni con lettere per ogni lettera dell'alfabeto
                    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        # Crea una copia della cella base
                        composite_img = Image.fromarray(cell_array.copy())
                        letter_draw = ImageDraw.Draw(composite_img)

                        # Ottieni le dimensioni della lettera
                        bbox = self.grid_font.getbbox(letter)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]

                        # Calcola la posizione centrata con gli aggiustamenti
                        x_pos = (effective_cell_size - text_width) // 2 + thickness + horizontal_adj
                        y_pos = (effective_cell_size - text_height) // 2 + thickness + vertical_adj

                        # Disegna la lettera
                        letter_draw.text(
                            (x_pos, y_pos),
                            letter,
                            font=self.grid_font,
                            fill=(*text_color, 255)
                        )

                        # Salva nella cache composita
                        composite_key = f'{border_type}_{letter}'
                        self._composite_cache[composite_key] = np.array(composite_img)

        print(f"Cache initialized with {len(self._cell_cache)} cell types and "
              f"{len(self._composite_cache)} letter combinations")

    def _create_hidden_column_cells(self, cell_size: int):
        """Crea celle speciali per la colonna nascosta con bordo oro e sfondo azzurro"""
        thickness = 6
        border_types = ['initial', 'inactive', 'highlight']

        for border_type in border_types:
            # Crea l'immagine base della cella
            cell_img = Image.new('RGBA', (cell_size, cell_size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(cell_img)

            # Disegna prima il rettangolo di sfondo azzurro
            draw.rectangle(
                [0, 0, cell_size, cell_size],
                fill=tuple(self._cached_colors['hidden_column_bg']),  # Sfondo azzurro
                outline=None
            )

            # Disegna il bordo oro
            half_thickness = thickness / 2
            borders = [
                half_thickness,  # left
                half_thickness,  # top
                cell_size - half_thickness,  # right
                cell_size - half_thickness  # bottom
            ]

            draw.rectangle(
                borders,
                outline=tuple(self._cached_colors['hidden_column_border']),  # Bordo oro
                width=thickness
            )

            # Salva nella cache con prefisso 'hidden_'
            key = f'hidden_{border_type}_{thickness}'
            self._cell_cache[key] = np.array(cell_img)

            # Se non è una cella iniziale, prepara anche le versioni con lettere
            if border_type != 'initial':
                for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    composite_img = Image.fromarray(self._cell_cache[key].copy())
                    letter_draw = ImageDraw.Draw(composite_img)

                    # Ottieni le dimensioni della lettera
                    bbox = self.grid_font.getbbox(letter)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Calcola la posizione centrata della lettera
                    effective_cell_size = cell_size - (thickness * 2)
                    x_pos = (effective_cell_size - text_width) // 2 + thickness + self.font_config.horizontal_adjustment
                    y_pos = (effective_cell_size - text_height) // 2 + thickness + self.font_config.vertical_adjustment

                    # Disegna la lettera
                    letter_draw.text(
                        (x_pos, y_pos),
                        letter,
                        font=self.grid_font,
                        fill=(*self.font_config.grid_color, 255)
                    )

                    # Salva nella cache composita
                    composite_key = f'hidden_{border_type}_{letter}'
                    self._composite_cache[composite_key] = np.array(composite_img)

    def _precalculate_animation_timings(self) -> Dict:
        """Pre-calcola i timing delle animazioni per evitare calcoli ripetuti"""
        timings = {}
        # Usa template_data invece di template
        for sequence in self.template_data['animation_sequence']:
            if sequence['type'] == 'initial_grid':
                start_frame = self._seconds_to_frames(sequence['start'], self.fps)
                end_frame = self._seconds_to_frames(sequence['end'], self.fps)
                timings[start_frame] = {
                    'type': 'initial_grid',
                    'end_frame': end_frame
                }
            elif sequence['type'] == 'word_reveal':
                for word_data in sequence['sequence']:
                    word_index = word_data['word_index']
                    for anim in word_data['animations']:
                        start_frame = self._seconds_to_frames(anim['start'], self.fps)
                        end_frame = self._seconds_to_frames(anim['end'], self.fps)
                        if start_frame not in timings:
                            timings[start_frame] = []
                        timings[start_frame].append({
                            'type': anim['type'],
                            'word_index': word_index,
                            'end_frame': end_frame,
                            'data': anim
                        })
        return timings

    def _process_frame_with_cached_timings(self, frame: np.ndarray,
                                           frame_number: int,
                                           animation_timings: Dict):
        """Processa un frame usando i timing pre-calcolati"""
        # Controlla se ci sono animazioni che iniziano in questo frame
        if frame_number in animation_timings:
            animations = animation_timings[frame_number]
            if isinstance(animations, dict):  # initial_grid
                if animations['type'] == 'initial_grid' and \
                        frame_number <= animations['end_frame']:
                    self._apply_initial_grid(frame)
            else:  # word_reveal animations
                for anim in animations:
                    if frame_number <= anim['end_frame']:
                        timing = TimingInfo(frame_number, anim['end_frame'])
                        self._apply_word_animation(
                            frame, anim['word_index'],
                            anim['data'], timing, frame_number)

        # Controlla le animazioni in corso
        for start_frame, animations in animation_timings.items():
            if start_frame < frame_number:
                if isinstance(animations, dict):  # initial_grid
                    if animations['type'] == 'initial_grid' and \
                            frame_number <= animations['end_frame']:
                        self._apply_initial_grid(frame)
                else:  # word_reveal animations
                    for anim in animations:
                        if frame_number <= anim['end_frame']:
                            timing = TimingInfo(start_frame, anim['end_frame'])
                            self._apply_word_animation(
                                frame, anim['word_index'],
                                anim['data'], timing, frame_number)

    def _get_letter_adjustments(self) -> Tuple[int, int]:
        """Ottiene gli aggiustamenti di posizionamento delle lettere dalla configurazione"""
        grid_font_config = self.style.get('grid_font', {})
        vertical_adj = grid_font_config.get('vertical_adjustment', -2)
        horizontal_adj = grid_font_config.get('horizontal_adjustment', 0)
        return horizontal_adj, vertical_adj


def main():
    try:
        with profile_section("Total Execution"):
            # Inizializza il file manager
            file_manager = FileManager()

            # Verifica che i file necessari esistano
            required_files = {
                'input video': file_manager.get_input_video_path('input_video.mp4'),
                'template': file_manager.get_template_path('template.json'),
                'crossword data': file_manager.get_data_path('crossword-data-hidden-word.json'),
                'font': file_manager.get_font_path('PressStart2P-Regular.ttf')
            }

            for name, path in required_files.items():
                if not path.exists():
                    raise FileNotFoundError(f"Required {name} file not found: {path}")

            # Configurazione
            with profile_section("Configuration"):
                config = CrosswordConfig(
                    font_path=str(file_manager.get_font_path('PressStart2P-Regular.ttf')),
                    clue_font_size=24
                )
                config_manager = ConfigManager(config, file_manager)

            # Caricamento dati
            with profile_section("Data Loading"):
                template_data = config_manager.load_json_file('template.json')
                crossword_data = config_manager.load_json_file('crossword-data-hidden-word.json')
                config_manager.validate_template(template_data)
                config_manager.validate_crossword_data(crossword_data)

            # Generazione video
            with profile_section("Video Generation"):
                generator = CrosswordVideoGenerator(
                    template_data,
                    crossword_data,
                    config_manager,
                    file_manager
                )

                # Processa il video con il nuovo nome file
                generator.process_video('input_video.mp4')

        profiler.print_stats()

    except FileNotFoundError as e:
        print(f"File Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            file_manager.cleanup_temp_files()
        except:
            pass


if __name__ == "__main__":
    main()
