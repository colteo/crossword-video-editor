import pytest
from pathlib import Path
import shutil
from datetime import datetime
from src.utils.file_manager import FileManager


@pytest.fixture
def temp_base_dir(tmp_path):
    """Provide a temporary directory for testing"""
    return tmp_path


@pytest.fixture
def file_manager(temp_base_dir):
    """Provide a FileManager instance with temporary base directory"""
    fm = FileManager(str(temp_base_dir))
    yield fm
    # Cleanup after tests
    fm.cleanup_temp_files()


def test_directory_structure_creation(file_manager):
    """Test that all required directories are created"""
    assert file_manager.input_dir.exists()
    assert file_manager.output_dir.exists()
    assert file_manager.video_input_dir.exists()
    assert file_manager.templates_dir.exists()
    assert file_manager.data_dir.exists()
    assert file_manager.fonts_dir.exists()


def test_generate_output_filename(file_manager):
    """Test output filename generation"""
    current_date = datetime.now().strftime("%Y%m%d")
    filename = file_manager.generate_output_filename(
        crossword_type="standard",
        template_type="basic",
        add_guid=False
    )

    assert filename.startswith(current_date)
    assert filename.endswith(".mp4")
    assert "standard" in filename
    assert "basic" in filename


def test_generate_output_filename_with_guid(file_manager):
    """Test output filename generation with GUID"""
    filename = file_manager.generate_output_filename(
        crossword_type="standard",
        template_type="basic",
        add_guid=True
    )

    parts = filename.split('_')
    assert len(parts) == 4  # date, type, template, guid
    assert len(parts[3].split('.')[0]) == 8  # GUID length before extension


def test_temp_directory_creation(file_manager):
    """Test temporary directory creation and cleanup"""
    temp_dir = file_manager.create_temp_dir()
    assert temp_dir.exists()
    assert temp_dir.parent == file_manager.output_dir

    file_manager.cleanup_temp_files()
    assert not temp_dir.exists()


def test_get_temp_file_path(file_manager):
    """Test temporary file path generation"""
    temp_path = file_manager.get_temp_file_path("test.txt")
    assert temp_path.parent == file_manager.temp_dir
    assert temp_path.name == "test.txt"


def test_path_getters(file_manager):
    """Test various path getter methods"""
    # Input video path
    video_path = file_manager.get_input_video_path("test.mp4")
    assert video_path.parent == file_manager.video_input_dir
    assert video_path.name == "test.mp4"

    # Template path
    template_path = file_manager.get_template_path("template.json")
    assert template_path.parent == file_manager.templates_dir
    assert template_path.name == "template.json"

    # Data path
    data_path = file_manager.get_data_path("data.json")
    assert data_path.parent == file_manager.data_dir
    assert data_path.name == "data.json"

    # Font path
    font_path = file_manager.get_font_path("font.ttf")
    assert font_path.parent == file_manager.fonts_dir
    assert font_path.name == "font.ttf"


def test_output_video_path(file_manager):
    """Test output video path generation"""
    template_data = {"template_type": "basic"}
    crossword_data = {"crossword_type": "standard"}

    path = file_manager.get_output_video_path(template_data, crossword_data)
    assert path.parent == file_manager.output_dir
    assert path.suffix == ".mp4"


def test_clean_type_name(file_manager):
    """Test type name cleaning"""
    # Test basic cleaning
    assert file_manager._clean_type_name("test type") == "test_type"

    # Test special characters
    assert file_manager._clean_type_name("test@type!") == "testtype"

    # Test multiple spaces
    assert file_manager._clean_type_name("test   type") == "test_type"

    # Test leading/trailing spaces
    assert file_manager._clean_type_name("  test type  ") == "test_type"


def test_temp_dir_cleanup(file_manager):
    """Test temporary directory cleanup"""
    # Create some temporary files
    temp_dir = file_manager.create_temp_dir()
    (temp_dir / "test.txt").touch()

    assert temp_dir.exists()
    assert (temp_dir / "test.txt").exists()

    file_manager.cleanup_temp_files()
    assert not temp_dir.exists()