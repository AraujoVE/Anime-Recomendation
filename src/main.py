from tracemalloc import start
from classification_strategy import ClassificationStrategy
from gui.gui import start_gui
from gui.main_menu import create_main_menu
from src.anime_dataset import AnimeDataset


def initialize_dataset():
    animeDataset = AnimeDataset()

def main():
    start_gui(create_main_menu())
    pass

if __name__ == "__main__":
    main()