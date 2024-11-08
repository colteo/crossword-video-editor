from pathlib import Path
import shutil
import tempfile
import uuid
from datetime import datetime
import os
import random

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

    def get_random_template_path_by_type(self, template_type: str):
        template_path_type = self.templates_dir / template_type

        templates = self._get_all_file_in_folder(template_path_type)

        template = random.choice(templates)

        return template_path_type / template

    def _get_all_file_in_folder(self, path):
        """
        Lista tutti i file in una cartella usando os.listdir()
        """
        try:
            files = os.listdir(path)
            return files
        except Exception as e:
            print(f"Errore durante la lettura della cartella: {e}")
            return []