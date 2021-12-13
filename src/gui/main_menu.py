from turtle import onclick
from typing import Union
import dearpygui.dearpygui as dpg

import pandas as pd

from anime_dataset import ANIME_DATASET
import cols

current_anime_id = None

def update_anime_info(anime: Union[pd.Series, None]):
    if anime is None:
        dpg.set_value("search_name", f"No anime found")
        dpg.set_value("search_genres", "")
        dpg.set_value("search_score", "")
        dpg.set_value("search_episodes", "")
        dpg.set_value("search_type", "")
        dpg.set_value("search_producers", "")
        dpg.set_value("search_studios", "")
        dpg.set_value("search_source", "")
        dpg.set_value("search_synopsis_keywords", "")
        dpg.set_value("search_synopsis", "")
    else:
        dpg.set_value("search_name", f'Name: {anime[cols.NAME]}')
        dpg.set_value("search_genres", f'Genres: {anime[cols.GENRES]}')
        dpg.set_value("search_score", f'Score: {anime[cols.SCORE]}')
        dpg.set_value("search_episodes", f'Episodes: {anime[cols.EPISODES]}')
        dpg.set_value("search_type", f'Type: {anime[cols.TYPE]}')
        dpg.set_value("search_producers", f'Producers: {anime[cols.PRODUCERS]}')
        dpg.set_value("search_studios", f'Studios: {anime[cols.STUDIOS]}')
        dpg.set_value("search_source", f'Source: {anime[cols.SOURCE]}')
        dpg.set_value("search_synopsis_keywords", f'Synopsis_Keywords: {anime[cols.SYNOPSIS_KEYWORDS]}')
        dpg.set_value("search_synopsis", f'Synopsis: {anime[cols.SYNOPSIS]}', multiline=True)    

def search_anime(anime_name: str):
    animes = ANIME_DATASET.get_by_name(anime_name)
    if animes.empty:
        print("No anime found")
        update_anime_info(None)
        return
    anime = animes.iloc[0]  
    print("Anime found!" + anime[cols.NAME])
    update_anime_info(anime)

    current_anime_id = anime[cols.MAL_ID]
    pass

def recomend_anime():


    pass

def create_main_menu():
    viewport = dpg.create_viewport(title="Anime Recomendation", width=800, height=600)

    with dpg.window(label="Anime Recomendation") as main_window:
        # dpg.add_text("Type Anime name:")
        # dpg.add_input_text(label="Anime name:", id="anime_name")
        # dpg.add_button(label="Search", callback=lambda: search_anime(dpg.get_value("anime_name")), id="recomend_button")

        # Vertical space
        dpg.add_spacing()

        dpg.add_text("Search Results:")
        dpg.add_text("Name:", id="search_name")
        dpg.add_text("Genres:", id="search_genres")
        dpg.add_text("Score:", id="search_score")
        dpg.add_text("Episodes:", id="search_episodes")
        dpg.add_text("Type:", id="search_type")
        dpg.add_text("Producers:", id="search_producers")
        dpg.add_text("Studios:", id="search_studios")
        dpg.add_text("Source:", id="search_source")
        dpg.add_text("Synopsis_Keywords:", id="search_synopsis_keywords")
        dpg.add_text("Synopsis:", id="search_synopsis")


        dpg.add_button(label="Recomend", callback=lambda: recomend_anime(), id="recomend_button")


        dpg.add_text("Name:", id="recomend_name")
        dpg.add_text("Genres:", id="recomend_genres")
        dpg.add_text("Score:", id="recomend_score")
        dpg.add_text("Episodes:", id="recomend_episodes")
        dpg.add_text("Type:", id="recomend_type")
        dpg.add_text("Producers:", id="recomend_producers")
        dpg.add_text("Studios:", id="recomend_studios")
        dpg.add_text("Source:", id="recomend_source")
        dpg.add_text("Synopsis_Keywords:", id="recomend_synopsis_keywords")
        dpg.add_text("Synopsis:", id="recomend_synopsis")

        dpg.set_primary_window(main_window, True)

    return viewport
    