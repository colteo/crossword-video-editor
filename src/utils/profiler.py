import statistics
import time
import functools
from typing import Dict, Optional
from contextlib import contextmanager


class SimpleProfiler:
    """
    A simple profiler to track execution times across different sections of code.

    Examples:
        # Using the decorator
        @profile
        def my_function():
            pass

        # Using with specific section name
        @profile(section_name="Custom Section")
        def another_function():
            pass

        # Using as context manager
        with profile_section("My Section"):
            # code to profile
            pass
    """

    def __init__(self):
        self.times: Dict[str, list] = {}
        self._enabled: bool = True

    @property
    def enabled(self) -> bool:
        """Get the current enabled state of the profiler"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """Set the enabled state of the profiler"""
        self._enabled = value

    def add_time(self, name: str, elapsed: float):
        """
        Add execution time for a specific section.

        Args:
            name: Name of the section
            elapsed: Time elapsed in seconds
        """
        if not self._enabled:
            return

        if name not in self.times:
            self.times[name] = []
        self.times[name].append(elapsed)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all tracked sections.

        Returns:
            Dictionary containing statistics for each tracked section
        """
        stats = {}
        for name, times in self.times.items():
            if times:
                stats[name] = {
                    'avg': statistics.mean(times),
                    'min': min(times),
                    'max': max(times),
                    'total': sum(times),
                    'calls': len(times),
                    'std_dev': statistics.stdev(times) if len(times) > 1 else 0
                }
        return stats

    def print_stats(self):
        """Print formatted statistics to console"""
        stats = self.get_stats()
        if not stats:
            print("No profiling data available")
            return

        print("\n=== Profiling Results ===")
        # Find longest name for formatting
        max_name = max(len(name) for name in stats.keys())

        # Print header
        header = (f"{'Section':<{max_name}} | {'Avg (ms)':>10} | {'Min (ms)':>10} | "
                  f"{'Max (ms)':>10} | {'Total (s)':>10} | {'Calls':>8} | {'Std Dev':>10}")
        print(header)
        print("-" * len(header))

        # Print each section's stats
        for name, data in sorted(stats.items()):
            print(f"{name:<{max_name}} | {data['avg'] * 1000:10.2f} | {data['min'] * 1000:10.2f} | "
                  f"{data['max'] * 1000:10.2f} | {data['total']:10.2f} | {data['calls']:8d} | "
                  f"{data['std_dev'] * 1000:10.2f}")

    def reset(self):
        """Reset all profiling data"""
        self.times.clear()


# Create a global profiler instance
profiler = SimpleProfiler()


def profile(func=None, section_name: Optional[str] = None):
    """
    Decorator to profile function execution time.
    Can be used with or without parameters.

    Args:
        func: Function to profile
        section_name: Optional custom name for the profiling section

    Returns:
        Decorated function
    """
    if func is None:
        return lambda f: profile(f, section_name)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not profiler.enabled:
            return func(*args, **kwargs)

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
    """
    Context manager to profile a section of code.

    Args:
        name: Name of the section to profile

    Example:
        with profile_section("my operation"):
            # code to profile
    """
    if not profiler.enabled:
        yield
        return

    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        profiler.add_time(name, elapsed)