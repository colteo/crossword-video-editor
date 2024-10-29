import cv2
import json
import textwrap
import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum, auto

class AnimationType(Enum):
    INITIAL_GRID = "initial_grid"
    SHOW_CLUE_01 = "show_clue_01"
    SHOW_GRID_WORD_01_EMPTY = "show_grid_word_01_empty"
    SHOW_GRID_WORD_01 = "show_grid_word_01"
    SHOW_CLUE_02 = "show_clue_02"
    SHOW_GRID_WORD_02_EMPTY = "show_grid_word_02_empty"
    SHOW_GRID_WORD_02 = "show_grid_word_02"
    SHOW_CLUE_03 = "show_clue_03"
    SHOW_GRID_WORD_03_EMPTY = "show_grid_word_03_empty"
    SHOW_GRID_WORD_03 = "show_grid_word_03"
    SHOW_CLUE_04 = "show_clue_04"
    SHOW_GRID_WORD_04_EMPTY = "show_grid_word_04_empty"
    SHOW_GRID_WORD_04 = "show_grid_word_04"
    SHOW_CLUE_05 = "show_clue_05"
    SHOW_GRID_WORD_05_EMPTY = "show_grid_word_05_empty"
    SHOW_GRID_WORD_05 = "show_grid_word_05"

@dataclass
class TimingInfo:
    """Classe per gestire le informazioni di timing"""
    start_frame: int
    end_frame: int


class CrosswordVideoGenerator:
    def __init__(self, template_data: Dict, crossword_data: Dict):
        """
        Inizializza il generatore del video
        """
        self.template = template_data
        self.crossword = crossword_data
        self.grid = np.array(crossword_data['grid'])
        self.words = crossword_data['words']
        self.style = template_data['style_settings']['crossword']
        self.layout = template_data['layout']
        self.valid_cells = self._get_valid_cells()
        self.bounds = self._calculate_bounds()
        min_x, min_y, max_x, max_y = self.bounds
        self.grid_width = (max_x - min_x + 1) * self.style['cell_size']
        self.grid_height = (max_y - min_y + 1) * self.style['cell_size']
        # Configurazione globale per il wrapping del testo
        self.max_text_width = self.style.get('max_text_width', 500)
        self.line_spacing = self.style.get('line_spacing', 1.5)

    def _wrap_text(self, text: str, font, font_scale: float, thickness: int, max_width: int) -> List[str]:
        """
        Divide il testo in righe basandosi sulla larghezza massima.

        Args:
            text: Il testo da wrappare
            font: Il font da utilizzare
            font_scale: La scala del font
            thickness: Lo spessore del testo
            max_width: La larghezza massima in pixel

        Returns:
            Lista di stringhe, una per ogni riga
        """
        # Prima stima approssimativa dei caratteri per riga
        test_text = "A" * 100
        (test_width, _), _ = cv2.getTextSize(test_text, font, font_scale, thickness)
        chars_per_pixel = len(test_text) / test_width
        estimated_chars = int(max_width * chars_per_pixel * 0.85)  # 85% per sicurezza

        # Dividi il testo in righe
        wrapped_lines = textwrap.wrap(text, width=estimated_chars)

        # Verifica e aggiusta le righe se necessario
        final_lines = []
        current_line = ""
        words = text.split()

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            (test_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

            if test_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    final_lines.append(current_line)
                current_line = word

        if current_line:
            final_lines.append(current_line)

        return final_lines

    def _create_clue_overlay(self, clue_text: str, pattern: Dict) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crea l'overlay per l'indizio con supporto per il wrapping del testo

        Args:
            clue_text: Testo dell'indizio
            pattern: Pattern di animazione con le informazioni di stile e posizione
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = pattern.get('font_scale', self.style['font_scale'])
        thickness = 2
        padding = pattern.get('padding', 20)

        # Usa la larghezza specificata nel pattern, altrimenti usa quella globale
        max_width = pattern.get('max_text_width', self.max_text_width)
        line_spacing = pattern.get('line_spacing', self.line_spacing)

        # Ottieni il colore del testo dal pattern o usa il default (nero)
        text_color = pattern.get('text_color', (0, 0, 0))

        # Dividi il testo in righe
        text_lines = self._wrap_text(
            clue_text,
            font,
            font_scale,
            thickness,
            max_width
        )

        # Calcola le dimensioni del testo
        line_heights = []
        line_widths = []
        for line in text_lines:
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            line_heights.append(text_height)
            line_widths.append(text_width)

        # Calcola le dimensioni totali dell'overlay
        max_line_width = max(line_widths)
        total_text_height = sum(line_heights)
        line_spacing_px = int(max(line_heights) * (line_spacing - 1))
        total_height = total_text_height + (len(text_lines) - 1) * line_spacing_px

        # Crea l'immagine per l'indizio con canale alpha completamente trasparente
        width = max_line_width + 2 * padding
        height = total_height + 2 * padding
        clue_overlay = np.zeros((height, width, 4), dtype=np.uint8)

        # Non aggiungiamo più il rettangolo dello sfondo

        # Disegna ogni riga di testo
        y_position = padding + line_heights[0]  # Inizia dal padding superiore
        for i, line in enumerate(text_lines):
            cv2.putText(
                clue_overlay,
                line,
                (padding, y_position),
                font,
                font_scale,
                (*text_color, 255),  # Aggiungi alpha 255 al colore del testo
                thickness
            )
            # Aggiorna la posizione y per la prossima riga
            if i < len(text_lines) - 1:  # Se non è l'ultima riga
                y_position += line_heights[i] + line_spacing_px

        return clue_overlay, (width, height)

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

    def _apply_animation(self, frame: np.ndarray, pattern: Dict, timing: TimingInfo, frame_number: int):
        """
        Applica una specifica animazione al frame
        """
        anim_type = AnimationType(pattern['type'])

        if anim_type == AnimationType.INITIAL_GRID:
            grid_overlay = self._create_grid_overlay(is_initial=True)
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        elif anim_type == AnimationType.SHOW_CLUE_01:
            clue_text = self.words[0]['clue']
            clue_overlay, clue_size = self._create_clue_overlay(clue_text, pattern)
            position = self._get_clue_position(pattern, clue_size, (frame.shape[1], frame.shape[0]))
            self._overlay_image(frame, clue_overlay, position)

        elif anim_type == AnimationType.SHOW_GRID_WORD_01_EMPTY:
            grid_overlay = self._create_grid_overlay(
                highlight_word_index=0,
                show_letters=False,
                is_initial=False
            )
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        elif anim_type == AnimationType.SHOW_GRID_WORD_01:
            grid_overlay = self._create_grid_overlay(
                highlight_word_index=0,
                show_letters=True,
                is_initial=False
            )
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        elif anim_type == AnimationType.SHOW_CLUE_02:
            clue_text = self.words[1]['clue']
            clue_overlay, clue_size = self._create_clue_overlay(clue_text, pattern)
            position = self._get_clue_position(pattern, clue_size, (frame.shape[1], frame.shape[0]))
            self._overlay_image(frame, clue_overlay, position)

        elif anim_type == AnimationType.SHOW_GRID_WORD_02_EMPTY:
            grid_overlay = self._create_grid_overlay(
                highlight_word_index=1,
                show_letters=False,
                is_initial=False
            )
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        elif anim_type == AnimationType.SHOW_GRID_WORD_02:
            grid_overlay = self._create_grid_overlay(
                highlight_word_index=1,
                show_letters=True,
                is_initial=False
            )
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        elif anim_type == AnimationType.SHOW_CLUE_03:
            clue_text = self.words[2]['clue']
            clue_overlay, clue_size = self._create_clue_overlay(clue_text, pattern)
            position = self._get_clue_position(pattern, clue_size, (frame.shape[1], frame.shape[0]))
            self._overlay_image(frame, clue_overlay, position)

        elif anim_type == AnimationType.SHOW_GRID_WORD_03_EMPTY:
            grid_overlay = self._create_grid_overlay(
                highlight_word_index=2,
                show_letters=False,
                is_initial=False
            )
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        elif anim_type == AnimationType.SHOW_GRID_WORD_03:
            grid_overlay = self._create_grid_overlay(
                highlight_word_index=2,
                show_letters=True,
                is_initial=False
            )
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        elif anim_type == AnimationType.SHOW_CLUE_04:
            clue_text = self.words[3]['clue']
            clue_overlay, clue_size = self._create_clue_overlay(clue_text, pattern)
            position = self._get_clue_position(pattern, clue_size, (frame.shape[1], frame.shape[0]))
            self._overlay_image(frame, clue_overlay, position)

        elif anim_type == AnimationType.SHOW_GRID_WORD_04_EMPTY:
            grid_overlay = self._create_grid_overlay(
                highlight_word_index=3,
                show_letters=False,
                is_initial=False
            )
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        elif anim_type == AnimationType.SHOW_GRID_WORD_04:
            grid_overlay = self._create_grid_overlay(
                highlight_word_index=3,
                show_letters=True,
                is_initial=False
            )
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        elif anim_type == AnimationType.SHOW_CLUE_05:
            clue_text = self.words[4]['clue']
            clue_overlay, clue_size = self._create_clue_overlay(clue_text, pattern)
            position = self._get_clue_position(pattern, clue_size, (frame.shape[1], frame.shape[0]))
            self._overlay_image(frame, clue_overlay, position)

        elif anim_type == AnimationType.SHOW_GRID_WORD_05_EMPTY:
            grid_overlay = self._create_grid_overlay(
                highlight_word_index=4,
                show_letters=False,
                is_initial=False
            )
            positions = self._calculate_positions(frame.shape[1], frame.shape[0])
            self._overlay_image(frame, grid_overlay, positions['grid'])

        elif anim_type == AnimationType.SHOW_GRID_WORD_05:
            grid_overlay = self._create_grid_overlay(
                highlight_word_index=4,
                show_letters=True,
                is_initial=False
            )
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
                             is_initial: bool = False) -> np.ndarray:
        """
        Crea l'overlay della griglia del cruciverba

        Args:
            highlight_word_index: Indice della parola da evidenziare
            show_letters: Se mostrare le lettere nelle celle
            is_initial: Se è la griglia iniziale (determina il colore dei bordi)
        """
        overlay = np.zeros((self.grid_height, self.grid_width, 4), dtype=np.uint8)
        cell_size = self.style['cell_size']

        # Converti i colori da lista a tuple
        bg_color = tuple(self.style['background_color'])
        black_color = tuple(self.style['border_color'])
        grey_color = (128, 128, 128)
        highlight_color = tuple(self.style['highlight_color'])
        text_color = tuple(self.style['text_color'])

        # Definisci gli spessori dei bordi
        highlight_thickness = 4
        normal_thickness = 3

        # Se c'è una parola da evidenziare, ottieni le sue celle
        highlighted_cells = set()
        if highlight_word_index is not None:
            highlighted_cells = self._get_word_cells(self.words[highlight_word_index])

        # Ottieni tutte le celle delle parole già rivelate
        revealed_cells = set()
        if not is_initial and highlight_word_index is not None:
            # Include tutte le parole completamente rivelate (fino a highlight_word_index - 1)
            for i in range(highlight_word_index):
                revealed_cells.update(self._get_word_cells(self.words[i]))

        min_x, min_y, _, _ = self.bounds

        # Disegna tutte le celle valide
        for y, x in self.valid_cells:
            rel_y = y - min_y
            rel_x = x - min_x
            px = rel_x * cell_size
            py = rel_y * cell_size

            # Crea la cella
            cell = np.full((cell_size, cell_size, 4),
                           [*bg_color, 255], dtype=np.uint8)

            # Scegli il colore e lo spessore del bordo
            if is_initial:
                current_border_color = black_color
                current_thickness = normal_thickness
            else:
                is_highlighted = (y, x) in highlighted_cells
                current_border_color = highlight_color if is_highlighted else grey_color
                current_thickness = highlight_thickness if is_highlighted else normal_thickness

            # Disegna il bordo della cella
            cv2.rectangle(cell, (0, 0), (cell_size - 1, cell_size - 1),
                          (*current_border_color, 255), current_thickness)

            # Aggiungi la lettera se:
            # 1. NON è la griglia iniziale E
            # 2. (La cella appartiene a una parola già rivelata O
            #     è la parola corrente con show_letters True)
            should_show_letter = (
                    not is_initial and (
                    (y, x) in revealed_cells or
                    (show_letters and (y, x) in highlighted_cells)
            )
            )

            if should_show_letter:
                letter = self.grid[y][x]
                if letter != '_':
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = self.style['font_scale']
                    thickness = 2
                    text_size = cv2.getTextSize(letter, font, font_scale, thickness)[0]

                    text_x = int((cell_size - text_size[0]) / 2)
                    text_y = int((cell_size + text_size[1]) / 2)

                    cv2.putText(cell, letter, (text_x, text_y), font, font_scale,
                                (*text_color, 255), thickness)

            overlay[py:py + cell_size, px:px + cell_size] = cell

        return overlay

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

        # Ottieni le proprietà del video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prepara il writer del video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_number = 0
        print(f"Inizio elaborazione video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = frame.copy()

            # Applica tutte le animazioni attive per questo frame
            for pattern in self.template['animation_patterns']:
                timing = self._get_timing_info(pattern, fps)
                if self._is_frame_in_timing(frame_number, timing):
                    self._apply_animation(result, pattern, timing, frame_number)

            out.write(result)
            frame_number += 1

            if frame_number % fps == 0:  # Mostra il progresso ogni secondo
                print(f"Elaborati {frame_number / fps:.1f} secondi...")

        cap.release()
        out.release()
        print(f"Elaborazione video completata. Output salvato in: {output_video_path}")


def main():
    """Funzione principale"""
    try:
        # Carica il template
        with open('template-reale.json', 'r') as f:
            template_data = json.load(f)

        # Carica i dati del cruciverba
        with open('crossword-data.json', 'r') as f:
            crossword_data = json.load(f)

        # Crea il generatore
        generator = CrosswordVideoGenerator(template_data, crossword_data)

        # Genera il video
        generator.process_video('input_video_2.mp4', 'output_video.mp4')

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