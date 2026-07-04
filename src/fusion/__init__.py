"""Fusion module for combining vision, audio, and NLP outputs into summaries."""

from src.fusion.summarize import generate_summary, fuse

__all__ = ["generate_summary", "fuse"]