from flask import Flask, request, jsonify
from src.utils.profiler import profiler, profile, profile_section
from src.utils.file_manager import FileManager
from src.types.crossword_types import CrosswordConfig
from src.config.config_manager import ConfigManager
from src.core.crossword_video_generator import CrosswordVideoGenerator
import json

app = Flask(__name__)

def generate_video(crossword_data):
    try:
        with profile_section("Total Execution"):
            # Inizializza il file manager
            file_manager = FileManager()

            # Inizializza il file manager
            config = CrosswordConfig()
            config_manager = ConfigManager(config, file_manager)

            template_path = file_manager.get_random_template_path_by_type(crossword_data['crossword_type'])

            # Caricamento e validazione del template
            template_data = config_manager.load_json_file_by_path(template_path)
            config_manager.validate_template(template_data)
            config_manager.validate_crossword_data(crossword_data)

            # Verifica che i file necessari esistano
            required_files = {
                'input video': file_manager.get_input_video_path(template_data['input_video']),
                'font': file_manager.get_font_path(template_data['fonts']['main']['file'])
            }

            for name, path in required_files.items():
                if not path.exists():
                    raise FileNotFoundError(f"Required {name} file not found: {path}")

            # Generazione video
            generator = CrosswordVideoGenerator(
                template_data,
                crossword_data,
                config_manager,
                file_manager
            )

            # Processa il video con il nome file dalla configurazione
            generator.process_video(template_data['input_video'])

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