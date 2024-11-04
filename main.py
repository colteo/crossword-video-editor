from src.utils.profiler import profiler, profile, profile_section
from src.utils.file_manager import FileManager
from src.types.crossword_types import (
    CrosswordConfig,
)
from src.config.config_manager import ConfigManager
from src.core.crossword_video_generator import CrosswordVideoGenerator

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
