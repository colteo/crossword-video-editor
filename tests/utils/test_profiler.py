import pytest
import time
from src.utils.profiler import SimpleProfiler, profile, profile_section


def test_profiler_basic_functionality():
    profiler = SimpleProfiler()

    # Test adding time
    profiler.add_time("test", 1.0)
    assert "test" in profiler.times
    assert profiler.times["test"] == [1.0]

    # Test multiple times
    profiler.add_time("test", 2.0)
    assert profiler.times["test"] == [1.0, 2.0]


def test_profiler_stats():
    profiler = SimpleProfiler()
    times = [1.0, 2.0, 3.0]

    for t in times:
        profiler.add_time("test", t)

    stats = profiler.get_stats()
    assert "test" in stats
    assert stats["test"]["avg"] == 2.0
    assert stats["test"]["min"] == 1.0
    assert stats["test"]["max"] == 3.0
    assert stats["test"]["total"] == 6.0
    assert stats["test"]["calls"] == 3


def test_profiler_decorator():
    profiler = SimpleProfiler()

    @profile
    def test_func():
        time.sleep(0.1)

    test_func()
    stats = profiler.get_stats()

    assert "test_func" in stats
    assert stats["test_func"]["calls"] == 1
    assert 0.1 <= stats["test_func"]["total"] <= 0.2


def test_profiler_context_manager():
    profiler = SimpleProfiler()

    with profile_section("test_section"):
        time.sleep(0.1)

    stats = profiler.get_stats()
    assert "test_section" in stats
    assert stats["test_section"]["calls"] == 1
    assert 0.1 <= stats["test_section"]["total"] <= 0.2


def test_profiler_enabled_disabled():
    profiler = SimpleProfiler()

    # Test when enabled
    profiler.enabled = True
    profiler.add_time("test", 1.0)
    assert "test" in profiler.times

    # Test when disabled
    profiler.enabled = False
    profiler.add_time("test2", 1.0)
    assert "test2" not in profiler.times


def test_profiler_reset():
    profiler = SimpleProfiler()

    profiler.add_time("test", 1.0)
    assert len(profiler.times) > 0

    profiler.reset()
    assert len(profiler.times) == 0


def test_custom_section_name():
    profiler = SimpleProfiler()

    @profile(section_name="custom_name")
    def test_func():
        time.sleep(0.1)

    test_func()
    stats = profiler.get_stats()

    assert "custom_name" in stats
    assert "test_func" not in stats