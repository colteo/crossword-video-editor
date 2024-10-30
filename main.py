import cv2
import json
import textwrap
import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum, auto
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip

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


class CrosswordVideoGenerator:
    def __init__(self, template_data: Dict, crossword_data: Dict):
        """
        Inizializza il generatore del video con la corretta sequenza di inizializzazione
        """
        # Salva i dati di input
        self.template = template_data
        self.crossword = crossword_data

        # Estrai i dati principali
        self.grid = np.array(crossword_data['grid'])
        self.words = crossword_data['words']
        self.style = template_data['style_settings']['crossword']
        self.layout = template_data['layout']

        # Inizializza le dimensioni e i parametri della griglia
        self.valid_cells = self._get_valid_cells()
        self.bounds = self._calculate_bounds()
        min_x, min_y, max_x, max_y = self.bounds
        self.grid_width = (max_x - min_x + 1) * self.style['cell_size']
        self.grid_height = (max_y - min_y + 1) * self.style['cell_size']

        # Inizializza i parametri di testo
        self.max_text_width = self.style.get('max_text_width', 500)
        self.line_spacing = self.style.get('line_spacing', 1.5)

        # Ottieni le dimensioni dei font dalle impostazioni
        clue_font_size = self.style['clue_font']['size']
        grid_font_size = self.style['grid_font']['size']

        # Inizializza i font
        try:
            # Prova a caricare i font personalizzati
            self.clue_font = ImageFont.truetype("PressStart2P-Regular.ttf", clue_font_size)
            self.grid_font = ImageFont.truetype("PressStart2P-Regular.ttf", grid_font_size)
            print(f"Font caricati - Clue size: {clue_font_size}, Grid size: {grid_font_size}")
        except IOError as e:
            print(f"Errore nel caricamento dei font ({str(e)}), uso il font di default")
            # Se il caricamento fallisce, usa i font di default
            default_font = ImageFont.load_default()
            self.clue_font = default_font
            self.grid_font = default_font
        except Exception as e:
            print(f"Errore inaspettato nell'inizializzazione dei font: {str(e)}")
            # In caso di altri errori, usa comunque i font di default
            default_font = ImageFont.load_default()
            self.clue_font = default_font
            self.grid_font = default_font

        # Attributo per fps, inizializzato a None e settato più tardi in process_video
        self.fps = None

    def _get_font_height(self, font: ImageFont.FreeTypeFont) -> int:
        """
        Ottiene l'altezza del font usando get_bbox
        """
        bbox = font.getbbox("Aj")  # Usa lettere alte e basse per ottenere l'altezza completa
        return bbox[3] - bbox[1]

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """
        Divide il testo in righe basandosi sulla larghezza massima usando PIL
        """
        words = text.split()
        lines = []
        current_line = []
        current_width = 0

        # Crea un'immagine temporanea per misurare il testo
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)

        for word in words:
            word_width = draw.textlength(word, font=font)
            space_width = draw.textlength(" ", font=font)

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

        return lines

    def _create_clue_overlay(self, clue_text: str, pattern: Dict) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crea l'overlay per l'indizio usando PIL per il rendering del font
        """
        # Converti il testo in maiuscolo
        clue_text = clue_text.upper()

        padding = pattern.get('padding', 20)
        max_width = pattern.get('max_text_width', self.max_text_width)
        line_spacing = pattern.get('line_spacing', self.line_spacing)
        text_color = pattern.get('text_color', (0, 0, 0))

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

        # Disegna il testo
        y = padding
        for line in lines:
            draw.text((padding, y), line,
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

    def _get_word_index_from_animation_type(self, anim_type: AnimationType) -> int:
        """
        Estrae l'indice della parola dal tipo di animazione

        Args:
            anim_type: Tipo di animazione (enum AnimationType)

        Returns:
            Indice della parola (0-based)
        """
        name = anim_type.name

        # Cerca il numero alla fine del nome usando una regex
        import re
        if "_EMPTY" in name:
            # Per le animazioni tipo SHOW_GRID_WORD_01_EMPTY
            match = re.search(r'_(\d+)_EMPTY$', name)
        else:
            # Per le animazioni tipo SHOW_GRID_WORD_01 o SHOW_CLUE_01
            match = re.search(r'_(\d+)$', name)

        if match:
            number = int(match.group(1))
            return number - 1  # Convertiamo in 0-based index

        raise ValueError(f"Non è possibile estrarre l'indice da {name}")

    def _apply_animation(self, frame: np.ndarray, pattern: Dict, timing: TimingInfo, frame_number: int):
        """Applica una specifica animazione al frame"""
        anim_type = AnimationType(pattern['type'])

        # Gestione di tutte le animazioni show_grid_word
        if anim_type in [
            AnimationType.SHOW_GRID_WORD_01,
            AnimationType.SHOW_GRID_WORD_02,
            AnimationType.SHOW_GRID_WORD_03,
            AnimationType.SHOW_GRID_WORD_04,
            AnimationType.SHOW_GRID_WORD_05
        ]:
            word_index = self._get_word_index_from_animation_type(anim_type)

            grid_overlay = self._create_grid_overlay(
                highlight_word_index=word_index,
                show_letters=True,
                is_initial=False,
                frame_number=frame_number,
                timing=timing
            )
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        # Gestione delle animazioni empty
        elif anim_type in [
            AnimationType.SHOW_GRID_WORD_01_EMPTY,
            AnimationType.SHOW_GRID_WORD_02_EMPTY,
            AnimationType.SHOW_GRID_WORD_03_EMPTY,
            AnimationType.SHOW_GRID_WORD_04_EMPTY,
            AnimationType.SHOW_GRID_WORD_05_EMPTY
        ]:
            word_index = self._get_word_index_from_animation_type(anim_type)

            grid_overlay = self._create_grid_overlay(
                highlight_word_index=word_index,
                show_letters=False,
                is_initial=False
            )
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        # Gestione delle animazioni clue
        elif anim_type in [
            AnimationType.SHOW_CLUE_01,
            AnimationType.SHOW_CLUE_02,
            AnimationType.SHOW_CLUE_03,
            AnimationType.SHOW_CLUE_04,
            AnimationType.SHOW_CLUE_05
        ]:
            word_index = self._get_word_index_from_animation_type(anim_type)
            clue_text = self.words[word_index]['clue']
            clue_overlay, clue_size = self._create_clue_overlay(clue_text, pattern)
            position = self._get_clue_position(pattern, clue_size, (frame.shape[1], frame.shape[0]))
            self._overlay_image(frame, clue_overlay, position)

        # Gestione della griglia iniziale
        elif anim_type == AnimationType.INITIAL_GRID:
            grid_overlay = self._create_grid_overlay(is_initial=True)
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

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

    def _get_timing_info(self, pattern: Dict, fps: int) -> TimingInfo:
        """
        Converte il timing da secondi a frames

        Args:
            pattern: Pattern di animazione dal template
            fps: Frames per secondo del video

        Returns:
            TimingInfo con start_frame e end_frame
        """
        start_frame = self._seconds_to_frames(pattern['start'], fps)
        end_frame = self._seconds_to_frames(pattern['end'], fps)

        return TimingInfo(start_frame, end_frame)

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

    def _create_grid_overlay(self, highlight_word_index: int = None, show_letters: bool = False,
                             is_initial: bool = False, frame_number: int = None,
                             timing: TimingInfo = None) -> np.ndarray:
        """
        Crea l'overlay della griglia del cruciverba con lettere che appaiono sequenzialmente
        """
        cell_size = self.style['cell_size']
        grid_width = (self.bounds[2] - self.bounds[0] + 1) * cell_size
        grid_height = (self.bounds[3] - self.bounds[1] + 1) * cell_size

        overlay_img = Image.new('RGBA', (grid_width, grid_height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay_img)

        bg_color = tuple(self.style['background_color'])
        black_color = tuple(self.style['border_color'])
        grey_color = (128, 128, 128)
        highlight_color = tuple(self.style['highlight_color'])
        text_color = tuple(self.style['text_color'])

        highlight_thickness = 6
        normal_thickness = 6

        highlighted_cells = set()
        if highlight_word_index is not None:
            highlighted_cells = self._get_word_cells(self.words[highlight_word_index])

        revealed_cells = set()
        if not is_initial and highlight_word_index is not None:
            for i in range(highlight_word_index):
                revealed_cells.update(self._get_word_cells(self.words[i]))

        min_x, min_y, max_x, max_y = self.bounds

        # Ottieni la sequenza di lettere per la parola corrente
        current_word_letters = []
        visible_letters = set()  # Set di coordinate delle lettere da mostrare
        if highlight_word_index is not None and show_letters and frame_number is not None and timing is not None:
            current_word_letters = self._get_word_letters_sequence(self.words[highlight_word_index])
            for i, (ly, lx, _) in enumerate(current_word_letters):
                if self._calculate_letter_visibility(frame_number, timing, i, len(current_word_letters)):
                    visible_letters.add((ly, lx))

        for y, x in self.valid_cells:
            rel_y = y - min_y
            rel_x = x - min_x
            px = rel_x * cell_size
            py = rel_y * cell_size

            cell_img = Image.new('RGBA', (cell_size, cell_size), (*bg_color, 255))
            cell_draw = ImageDraw.Draw(cell_img)

            if is_initial:
                current_border_color = black_color
                current_thickness = normal_thickness
            else:
                is_highlighted = (y, x) in highlighted_cells
                current_border_color = highlight_color if is_highlighted else grey_color
                current_thickness = highlight_thickness if is_highlighted else normal_thickness

            half_thickness = current_thickness / 2
            left = half_thickness
            top = half_thickness
            right = cell_size - half_thickness
            bottom = cell_size - half_thickness

            cell_draw.rectangle(
                [left, top, right, bottom],
                outline=(*current_border_color, 255),
                width=current_thickness
            )

            # Gestione delle lettere
            should_show_letter = (
                    (y, x) in revealed_cells or  # Lettere delle parole già rivelate
                    (y, x) in visible_letters  # Lettere della parola corrente che devono essere visibili
            )

            if should_show_letter:
                letter = self.grid[y][x]
                if letter != '_':
                    test_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    max_bbox = self.grid_font.getbbox(test_letter)
                    max_height = max_bbox[3] - max_bbox[1]

                    bbox = self.grid_font.getbbox(letter.upper())
                    text_width = bbox[2] - bbox[0]

                    margin_x = cell_size - text_width
                    margin_y = cell_size - max_height

                    text_x = margin_x // 2
                    text_y = margin_y // 2
                    y_offset = -2

                    cell_draw.text(
                        (text_x, text_y + y_offset),
                        letter.upper(),
                        font=self.grid_font,
                        fill=text_color
                    )

            overlay_img.paste(cell_img, (px, py))

        return np.array(overlay_img)

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

    def process_video(self, input_video_path: str, output_video_path: str):
        """Processa il video applicando le animazioni"""
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError("Impossibile aprire il video di input")

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_output = "temp_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, self.fps,
                              (self.width, self.height))

        frame_number = 0
        print(f"Inizio elaborazione video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = frame.copy()

            # Processa ogni sequenza di animazione nel template
            for sequence in self.template['animation_sequence']:
                self._process_animation_sequence(result, sequence, frame_number)

            out.write(result)
            frame_number += 1

            if frame_number % self.fps == 0:
                print(f"Elaborati {frame_number / self.fps:.1f} secondi...")

        cap.release()
        out.release()

        # Gestione dell'audio e finalizzazione come prima
        print("Combinazione del video con l'audio originale...")
        try:
            original_video = VideoFileClip(input_video_path)
            processed_video = VideoFileClip(temp_output)
            final_video = processed_video.set_audio(original_video.audio)
            final_video.write_videofile(output_video_path,
                                        codec='libx264',
                                        audio_codec='aac')
            original_video.close()
            processed_video.close()
            import os
            os.remove(temp_output)
        except Exception as e:
            print(f"Errore durante la combinazione dell'audio: {e}")
            import shutil
            shutil.move(temp_output, output_video_path)

        print(f"Elaborazione video completata. Output salvato in: {output_video_path}")

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
                                     letter_index: int, total_letters: int) -> bool:
        """
        Determina se una lettera deve essere visibile o no

        Args:
            frame_number: Frame corrente
            timing: Informazioni sul timing dell'animazione
            letter_index: Indice della lettera nella sequenza
            total_letters: Numero totale di lettere

        Returns:
            True se la lettera deve essere visibile, False altrimenti
        """
        total_duration = timing.end_frame - timing.start_frame
        frames_per_letter = total_duration / total_letters
        letter_appears_at = timing.start_frame + (letter_index * frames_per_letter)

        return frame_number >= letter_appears_at

    def _draw_letter(self, draw: ImageDraw.Draw, letter: str, cell_size: int,
                     text_color: Tuple[int, int, int], opacity: int):
        """
        Disegna una lettera nella cella con la data opacità mantenendo il colore corretto
        """
        test_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        max_bbox = self.grid_font.getbbox(test_letter)
        max_height = max_bbox[3] - max_bbox[1]

        bbox = self.grid_font.getbbox(letter.upper())
        text_width = bbox[2] - bbox[0]

        margin_x = cell_size - text_width
        margin_y = cell_size - max_height

        text_x = margin_x // 2
        text_y = margin_y // 2
        y_offset = -2

        # Usiamo il colore del testo con l'opacità desiderata
        # Manteniamo il colore originale e modifichiamo solo il canale alpha
        text_color_with_opacity = (text_color[0], text_color[1], text_color[2], opacity)

        draw.text(
            (text_x, text_y + y_offset),
            letter.upper(),
            font=self.grid_font,
            fill=text_color_with_opacity
        )

    def _apply_initial_grid(self, frame: np.ndarray):
        """
        Applica l'animazione della griglia iniziale
        """
        grid_overlay = self._create_grid_overlay(is_initial=True)
        positions = self._calculate_positions(frame.shape[1], frame.shape[0])
        self._overlay_image(frame, grid_overlay, positions['grid'])

    def _process_animation_sequence(self, frame: np.ndarray,
                                    sequence: Dict,
                                    frame_number: int):
        """
        Processa una sequenza di animazioni per un frame specifico
        """
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
                    timing = TimingInfo(
                        self._seconds_to_frames(anim['start'], self.fps),
                        self._seconds_to_frames(anim['end'], self.fps)
                    )
                    if self._is_frame_in_timing(frame_number, timing):
                        self._apply_word_animation(frame, word_index, anim, timing, frame_number)

    def _apply_word_animation(self, frame: np.ndarray,
                              word_index: int,
                              animation: Dict,
                              timing: TimingInfo,
                              frame_number: int):
        """
        Applica una singola animazione di una parola
        """
        anim_type = animation['type']

        if anim_type == 'show_grid_empty':
            grid_overlay = self._create_grid_overlay(
                highlight_word_index=word_index,
                show_letters=False,
                is_initial=False
            )
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        elif anim_type == 'show_grid_word':
            grid_overlay = self._create_grid_overlay(
                highlight_word_index=word_index,
                show_letters=True,
                is_initial=False,
                frame_number=frame_number,
                timing=timing
            )
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        elif anim_type == 'show_clue':
            clue_text = self.words[word_index]['clue']
            clue_overlay, clue_size = self._create_clue_overlay(clue_text, animation)
            position = self._get_clue_position(animation, clue_size,
                                               (frame.shape[1], frame.shape[0]))
            self._overlay_image(frame, clue_overlay, position)

def main():
    """Funzione principale"""
    try:
        # Carica il template
        with open('template.json', 'r') as f:
            template_data = json.load(f)

        # Carica i dati del cruciverba
        with open('crossword-data-02.json', 'r') as f:
            crossword_data = json.load(f)

        # Crea il generatore
        generator = CrosswordVideoGenerator(template_data, crossword_data)

        # Genera il video
        generator.process_video('input_video.mp4', 'output_video.mp4')

    except FileNotFoundError as e:
        print(f"Errore: File non trovato - {e}")
    except json.JSONDecodeError as e:
        print(f"Errore: JSON non valido - {e}")
    except ValueError as e:
        print(f"Errore: {e}")
    except Exception as e:
        print(f"Errore inaspettato: {e}")


if __name__ == "__main__":
    main()