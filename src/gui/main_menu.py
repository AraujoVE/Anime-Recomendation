from turtle import onclick
import dearpygui.dearpygui as dpg

from anime_dataset import ANIME_DATASET
import cols

def recommend_anime(anime_name: str):
    animes = ANIME_DATASET.search_by_name(anime_name)
    if animes.empty:
        print("No anime found")
        return
    print("Anime found:")
    anime = animes.iloc[0]

    print("Name:", anime[cols.NAME])
    print("Genres:", anime[cols.GENRES])
    print("Score:", anime[cols.SCORE])
    print("Episodes:", anime[cols.EPISODES])
    print("Type:", anime[cols.TYPE])
    print("Producers:", anime[cols.PRODUCERS])
    print("Studios:", anime[cols.STUDIOS])
    print("Source:", anime[cols.SOURCE])
    print("Synopsis_Keywords:", anime[cols.SYNOPSIS_KEYWORDS])


    pass

def create_main_menu():
    viewport = dpg.create_viewport(title="Anime Recomendation", width=800, height=600)

    with dpg.window(label="Anime Recomendation") as main_window:
        dpg.add_text("Type Anime name:")
        dpg.add_input_text(label="Anime name:", id="anime_name")
        dpg.add_button(label="Recomend!", callback=lambda: recommend_anime(dpg.get_value("anime_name")), id="recomend_button")
        dpg.set_primary_window(main_window, True)

    return viewport
    