from typing import List, Union
import dearpygui.dearpygui as dpg

def start_gui(viewports: Union[List[str], str]):
    if isinstance(viewports, str):
        viewports = [viewports]

    if len(viewports) != 1:
        raise ValueError("Only one viewport is supported at the moment")
    
    dpg.setup_dearpygui()
    for viewport in viewports:
        dpg.show_viewport(viewport)
    dpg.start_dearpygui()
