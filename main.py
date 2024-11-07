from flask import Flask, request, jsonify
from src.utils.profiler import profiler, profile, profile_section
from src.utils.file_manager import FileManager
from src.types.crossword_types import CrosswordConfig
from src.config.config_manager import ConfigManager
from src.core.crossword_video_generator import CrosswordVideoGenerator
import json

app = Flask(__name__)

# Costante per il template path
TEMPLATE_FILE = 'template-a.json'

def get_template_config(template_path):
    try:
        with open(template_path, 'r') as f:
            template_data = json.load(f)
            return {
                'font': template_data['fonts']['main']['file'],
                'input_video': template_data['input_video']
            }
    except Exception as e:
        # Fallback ai valori precedenti in caso di errore
        return {
            'font': 'PressStart2P-Regular.ttf',
            'input_video': 'input_video_long.mp4'
        }

# File configurations centralized at the top of the file
template_config = get_template_config(TEMPLATE_FILE)
FILE_PATHS = {
    'input video': template_config['input_video'],
    'template': TEMPLATE_FILE,
    'font': template_config['font']
}

def generate_video(crossword_data):
    try:
        with profile_section("Total Execution"):
            # Inizializza il file manager
            file_manager = FileManager()

            # Verifica che i file necessari esistano
            required_files = {
                'input video': file_manager.get_input_video_path(FILE_PATHS['input video']),
                'template': file_manager.get_template_path(FILE_PATHS['template']),
                'font': file_manager.get_font_path(FILE_PATHS['font'])
            }

            for name, path in required_files.items():
                if not path.exists():
                    raise FileNotFoundError(f"Required {name} file not found: {path}")

            # Configurazione
            with profile_section("Configuration"):
                config = CrosswordConfig(
                    font_path=str(file_manager.get_font_path(FILE_PATHS['font'])),
                    clue_font_size=24
                )
                config_manager = ConfigManager(config, file_manager)

            # Caricamento e validazione del template
            with profile_section("Data Loading"):
                template_data = config_manager.load_json_file(FILE_PATHS['template'])
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

                # Processa il video con il nome file dalla configurazione
                generator.process_video(FILE_PATHS['input video'])

        profiler.print_stats()
        return {"status": "success", "message": "Video generated successfully"}

    except FileNotFoundError as e:
        return {"status": "error", "message": f"File Error: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}"}
    finally:
        try:
            file_manager.cleanup_temp_files()
        except:
            pass

@app.route('/generate-video', methods=['POST'])
def generate_video_api():
    data = request.get_json()
    if not data or 'crossword_data' not in data:
        return jsonify({"status": "error", "message": "Missing 'crossword_data' in request"}), 400

    crossword_data = data['crossword_data']
    result = generate_video(crossword_data)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)