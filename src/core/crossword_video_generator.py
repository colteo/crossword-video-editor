import cv2
import numpy as np
from typing import Dict, List, Tuple, Set
from enum import Enum, auto
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
from typing import Optional, Dict, Any
import json
from typing import Dict, Optional
import os
from pathlib import Path
import shutil
from src.utils.profiler import profiler, profile, profile_section
from src.utils.file_manager import FileManager
from src.types.crossword_types import (
    CrosswordType,
    HiddenWordInfo,
    WordIntersection,
    CrosswordMetadata,
    CrosswordConfig,
    FontConfig,
    TimingInfo
)
from src.config.config_manager import ConfigManager
from src.font.font_manager import FontManager
from src.animation.animation_manager import AnimationManager, AnimationType, WordAnimation

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

                # Inizializza animation manager qui, dopo aver impostato fps
                self.animation_manager = AnimationManager(self.fps)

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
        Determina se una lettera deve essere visibile basandosi sull'AnimationManager
        """
        return self.animation_manager.calculate_letter_visibility(
            frame_number=frame_number,
            timing=timing,
            letter_index=letter_index,
            total_letters=total_letters,
            animation_config=animation_config
        )

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
        return self.animation_manager.calculate_animation_timings(self.template_data)

    def _process_frame_with_cached_timings(self, frame: np.ndarray,
                                           frame_number: int,
                                           animation_timings: Dict):
        """
        Processa un frame usando i timing pre-calcolati

        Args:
            frame: Frame video da processare
            frame_number: Numero del frame corrente
            animation_timings: Dictionary dei timing pre-calcolati
        """
        # Controlla se ci sono animazioni che iniziano in questo frame
        if frame_number in animation_timings:
            animations = animation_timings[frame_number]
            if isinstance(animations, dict):  # initial_grid
                if animations['type'] == AnimationType.INITIAL_GRID.value and \
                        frame_number <= animations['end_frame']:
                    self._apply_initial_grid(frame)
            else:  # word_reveal animations o random_phrases
                for anim in animations:
                    if frame_number <= anim['end_frame']:
                        if anim['type'] == AnimationType.RANDOM_PHRASES.value:
                            self._apply_random_phrase(frame, anim)
                        elif 'word_index' in anim:  # word_reveal animations
                            timing = TimingInfo(frame_number, anim['end_frame'])
                            self._apply_word_animation(
                                frame, anim['word_index'],
                                anim['data'], timing, frame_number)

        # Controlla le animazioni in corso
        for start_frame, animations in animation_timings.items():
            if start_frame < frame_number:
                if isinstance(animations, dict):  # initial_grid
                    if animations['type'] == AnimationType.INITIAL_GRID.value and \
                            frame_number <= animations['end_frame']:
                        self._apply_initial_grid(frame)
                else:  # word_reveal animations o random_phrases
                    for anim in animations:
                        if frame_number <= anim['end_frame']:
                            if anim['type'] == AnimationType.RANDOM_PHRASES.value:
                                self._apply_random_phrase(frame, anim)
                            elif 'word_index' in anim:  # word_reveal animations
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

    def _apply_random_phrase(self, frame: np.ndarray, animation: Dict):
        """
        Applica l'animazione della frase random al frame con parole evidenziate.
        """
        phrase_text = animation['phrase_text']
        style = animation['style']

        # Ottieni i colori di evidenziazione dal template
        highlight_colors = animation.get('highlight_colors', {})

        # Usa il nuovo metodo per creare l'overlay con parole evidenziate
        phrase_overlay, phrase_size = self._create_highlighted_text_overlay(
            phrase_text,
            {
                'max_text_width': style['max_text_width'],
                'text_color': style['text_color'],
                'text_align': style['text_align'],
                'padding': 20,
                'line_spacing': self.line_spacing
            },
            highlight_colors
        )

        # Calcola la posizione della frase
        position = self._get_clue_position(
            style,
            phrase_size,
            (frame.shape[1], frame.shape[0])
        )

        # Applica l'overlay al frame
        self._overlay_image(frame, phrase_overlay, position)

    def _create_highlighted_text_overlay(self, text: str, pattern: Dict,
                                         highlight_colors: Dict[str, Tuple[int, int, int]]) -> Tuple[
        np.ndarray, Tuple[int, int]]:
        """
        Crea l'overlay per il testo con parole chiave evidenziate usando colori configurabili

        Args:
            text: Testo da renderizzare
            pattern: Dictionary con le impostazioni di stile e posizionamento
            highlight_colors: Dictionary che mappa parole chiave ai loro colori RGB

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: Overlay del testo e le sue dimensioni
        """
        # Impostazioni di base
        padding = pattern.get('padding', 20)
        max_width = pattern.get('max_text_width', self.max_text_width)
        line_spacing = pattern.get('line_spacing', self.line_spacing)
        default_color = pattern.get('text_color', (0, 0, 0))
        text_align = pattern.get('text_align', 'left')

        # Crea un'immagine temporanea per misurare il testo
        temp_img = Image.new('RGBA', (max_width + padding * 2, 1000), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)

        # Divide il testo in parole
        words = text.split()
        lines = []
        current_line = []
        current_width = 0
        line_words_info = []  # Lista di tuple (parola, colore) per ogni linea

        # Elabora le parole e gestisce il wrapping
        for word in words:
            word_width = draw.textlength(word, font=self.clue_font)
            space_width = draw.textlength(" ", font=self.clue_font)

            # Determina il colore della parola
            word_color = highlight_colors.get(word.lower(), default_color)

            if current_width + word_width <= max_width:
                current_line.append(word)
                line_words_info.append((word, word_color))
                current_width += word_width + space_width
            else:
                if current_line:
                    lines.append(line_words_info)
                current_line = [word]
                line_words_info = [(word, word_color)]
                current_width = word_width + space_width

        if current_line:
            lines.append(line_words_info)

        # Calcola l'altezza totale necessaria
        line_height = self._get_font_height(self.clue_font)
        total_height = len(lines) * line_height * line_spacing

        # Crea l'immagine finale
        img = Image.new('RGBA',
                        (max_width + padding * 2, int(total_height) + padding * 2),
                        (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Disegna il testo linea per linea
        y = padding
        for line_info in lines:
            # Calcola la larghezza totale della linea per l'allineamento
            line_width = sum(draw.textlength(word, font=self.clue_font) +
                             draw.textlength(" ", font=self.clue_font)
                             for word, _ in line_info[:-1])
            line_width += draw.textlength(line_info[-1][0], font=self.clue_font)  # Ultima parola senza spazio

            # Calcola la posizione x iniziale in base all'allineamento
            if text_align == 'center':
                x = (max_width - line_width) / 2 + padding
            elif text_align == 'right':
                x = max_width - line_width + padding
            else:  # 'left' o qualsiasi altro valore
                x = padding

            # Disegna ogni parola con il suo colore
            for word, color in line_info:
                draw.text((x, y), word, font=self.clue_font, fill=(*color, 255))
                x += draw.textlength(word, font=self.clue_font) + draw.textlength(" ", font=self.clue_font)

            y += line_height * line_spacing

        return np.array(img), img.size
