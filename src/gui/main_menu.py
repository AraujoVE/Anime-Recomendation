from turtle import onclick
import dearpygui.dearpygui as dpg

def recommend_anime(anime_name: str):
    print("Recomend for " + anime_name)
    pass

def create_main_menu():
    viewport = dpg.create_viewport(title="Anime Recomendation", width=800, height=600)

    with dpg.window(label="Anime Recomendation") as main_window:
        dpg.add_text("Type Anime name:")
        dpg.add_input_text(label="Anime name:", id="anime_name")
        dpg.add_button(label="Recomend!", callback=lambda: recommend_anime(dpg.get_value("anime_name")), id="recomend_button")
        dpg.set_primary_window(main_window, True)

    return viewport
    