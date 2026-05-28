# config/settings.py

from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent

CATALOG_DIR = ROOT_DIR / 'catalog_images'

PICS_DIR = ROOT_DIR / 'figures'
PNG_PICS_DIR = PICS_DIR / 'png'
PDF_PICS_DIR = PICS_DIR / 'pdf'

DATA_DIR = ROOT_DIR / 'data'

