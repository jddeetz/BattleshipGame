#!/usr/bin/env python3
# Authors: jddeetz@gmail.com
"""
Creates a bokeh application to play battleship.
"""

import numpy as np
import pandas as pd
import pickle
import time

from bokeh.events import Tap
from bokeh.layouts import layout, row
from bokeh.models import Tabs, TabPanel, Paragraph
from bokeh.models.widgets import Div
from bokeh.plotting import curdoc, figure
from bokeh.transform import factor_cmap

import torch
# This is a hacky workaround to make relative imports work
# bokeh serve does not make main.py __main__
import sys
import os
sys.path.insert(0, os.path.abspath('.'))             
from model import BattleshipModel

CMAP = {"-1.0":"#0b5394", "0.0":"#323232", "1.0":"#cccccc"}
TEXT_CMAP = {"Fog":"#000000", "Sea":"#000000", "Ship":"#000000", "Hit!":"#ff0000", "Miss":"#ff0000"}

########################## OTHER STUFF
def get_board():
    """This is a helper function to load a random game board.

    Returns:
        np.array: Game board
    """
    # Open data file containing all of the game boards
    with open("./data/no_fog_boards.pkl", "rb") as input_file:
        data = pickle.load(input_file)
    
    # Get a random game board from the boards
    ind = np.random.randint(0, data.shape[0])
    return data[ind].reshape(10, 10)

def get_labels(shown_map_str:str, hidden_map_str:str):
    if shown_map_str == "0.0":
        return "Fog"
    elif shown_map_str != hidden_map_str:
        if shown_map_str == "1.0":
            return "Ship"
        else:
            return "Sea"
    else:
        if shown_map_str == "1.0":
            return "Hit!"
        else:
            return "Miss"
        
def make_ai_prediction(user_fog_map: np.array):
    """Given a np array, use neural network to predict locations of ships. 
    Return the x, y position that still has fog, that has the maximum value.

    Args:
        user_fog_map (np.array): what the ai can see about the current position of user ships.
            A value of 0 indicates fog, or unknown what is at this location. 
            Values of -1 and 1 correspond to empty and ship locations.

    Returns:
        x, y: the predicted location of a ship on the users map
    """
    # Go from np array to torch tensor in right shape and single precision
    map_tensor = torch.from_numpy((user_fog_map.reshape(1, 1, 10, 10)).astype(np.single))
    predictions = BM(map_tensor)

    # Get predictions back into a numpy array
    predictions = predictions.detach().numpy()
    predictions = predictions.reshape(10,10)

    # For the spaces which are fog (== 0), which has the maximum value
    predictions[user_fog_map != 0] = -999
    ind = np.unravel_index(np.argmax(predictions, axis=None), predictions.shape)
    
    return ind
    
########################## WIDGETS AND PLOTS
def get_link_div(url:str, link_text:str):
    """Creates HTML Div for some links on bio page.

    Returns:
        HTML Div
    """
    html_link = """<p><a href="{}" target="_blank" rel="noopener noreferrer">{}</a></p>"""
    return Div(text=html_link.format(url, link_text), margin=(5, 5, 20, 5))

def make_grid_map(primary_map:np.array, secondary_map:np.array, title:str):
    """What the user sees of their own map has no fog, whereas when they look at the AI board, it has fog.
    This function collapses code for generating both maps into a single function.

    The goal is to have a map that displays the following square colors:
        Blue -> ocean (empty space)
        Grey -> ship location (occupied space)
        Black -> unknown what is here (fog)

    Additionally there will be text in each space, corresponding to HIT or MISS.

    Args:
        primary_map: the map to be displayed, with only modifications for the circle colors above
        secondary_map: used to determine the color of the circles
        title: str to be displayed at top of figure

    Returns:
        bokeh figure
    """
    # These define the grid locations
    x_locs = [str(x) for x in range(0, 10)]
    y_locs = list(reversed(x_locs))

    # Make the figure object on the primary_map
    # Disable tap if this is the users board
    tools = "tap"
    if title == "Your Board":
        tools = []
    p = figure(title=title, width=400, height=400, x_range=x_locs, y_range=y_locs,
               toolbar_location=None, tools=tools)
    
    # Format dataframe, with "x", "y", and "occupied" as string-like columns
    df = pd.DataFrame({"shown_map":primary_map.flatten(), "hidden_map":secondary_map.flatten()})
    df["x"] = [i%10 for i in range(100)]
    df["y"] = [i//10 for i in range(100)]
    df = df.astype(str)

    # Get text to be written on each square
    df["label"] = df.apply(lambda x: get_labels(x["shown_map"], x["hidden_map"]), axis=1)

    # Make plot of grid squares with different colors
    p.rect(x="x", y="y", height=0.95, width=0.95, source=df, fill_alpha=0.6,
           color=factor_cmap('shown_map', palette=list(CMAP.values()), factors=list(CMAP.keys())))
    # Make text for each gridspace.
    p.text(x="x", y="y", text="label", text_font_style="bold", source=df, text_align="center", text_baseline="middle",
           color=factor_cmap('label', palette=list(TEXT_CMAP.values()), factors=list(TEXT_CMAP.keys())))

    # Turn off ticks, axes, and gridlines
    p.outline_line_color = None
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0

    return p

########################## FORMAT TABS
def get_about_me_tab():
    """Creates layout describing the awesome author.

    Returns:
        layout for biography
    """
    title = Div(text="<b>About Me</b>", sizing_mode="scale_both",
                margin=(5, 5, 20, 5))

    # Write a short biography
    bio = ("Josh Deetz is a Data Scientist with 5 years of experience. "
           "He has a PhD in Chemical Engineering from UC Davis. "
           "He loves hiking, jazz music, and slow cooked BBQ. "
           "If you know of any job opportunties, lets chat!")       
    biography = Paragraph(text=bio, width=400)

    # Create some links
    mailto_div = get_link_div("mailto:jddeetz@gmail.com", 
                              "&#9993 jddeetz@gmail.com")
    resume_div = get_link_div("https://drive.google.com/file/d/1ncRUiNztjxHMr0oaZfhMpGne99yr1Ykq/view?usp=sharing", 
                              "&#9883 My Resume")
    github_div = get_link_div("https://github.com/jddeetz/BattleshipGame", 
                              "&#9758 Github for this Project")
    li_div = get_link_div("https://www.linkedin.com/in/josh-deetz/", 
                          "&#9901 My LinkedIn")

    return layout([title, biography, mailto_div, resume_div, li_div, github_div])

def get_game_tab():
    """Creates layout for interacting with the game board.

    Returns:
        layout for game
    """
    # Get initial game boards
    ai_nofog_map = get_board()
    user_nofog_map = get_board()
    ai_fog_map = np.zeros((10, 10))
    user_fog_map = np.zeros((10, 10))

    # Make grid plots that user sees
    user_map = make_grid_map(user_nofog_map, user_fog_map, title="Your Map")
    ai_map = make_grid_map(ai_fog_map, ai_nofog_map, title="Enemy Map (Click on a square to fire a shot!)")
    map_plots = row(ai_map, user_map)

    def callback(event):
        ### Step 1: Reveal what is at this location
        # Convert float coords to map indices where the shot would fall
        user_shot_x = int(event.x)
        user_shot_y = 9 - int(event.y)
        # Remove fog at this locations
        ai_fog_map[user_shot_y, user_shot_x] = ai_nofog_map[user_shot_y, user_shot_x]

        ### Step 2: Update the AI map based on the user selected coordinates
        ai_map = make_grid_map(ai_fog_map, ai_nofog_map, title="Enemy Map (Click on a square to fire a shot!)")
        ai_map.on_event(Tap, callback)
        map_plots.children[0] = ai_map

        ### Step 3: Use CNN to predict grid locations of target
        ai_shot_x, ai_shot_y = make_ai_prediction(user_fog_map)
        
        ### Step 4: Change the user_fog_map to remove fog at this locations
        user_fog_map[ai_shot_x, ai_shot_y] = user_nofog_map[ai_shot_x, ai_shot_y]

        ### Step 5: Update the user map based on the AI selected coordinates
        user_map = make_grid_map(user_nofog_map, user_fog_map, title="Your Map")
        map_plots.children[1] = user_map

        ### Step 6: End game conditions
        ai_ships_found = np.nonzero(user_fog_map == 1)[0].shape[0]
        user_ships_found = np.nonzero(ai_fog_map == 1)[0].shape[0]

        if ai_ships_found == 17 and user_ships_found == 17:
            map_plots.children[0] = Div(text='''It's a tie! Congratulations Human!''', width=400,
                                        styles={'font-size': '200%'})
        elif ai_ships_found == 17:
            map_plots.children[0] = Div(text='''The AI found all of your ships. Better luck next time!''', width=400,
                                        styles={'font-size': '200%'})
        elif user_ships_found == 17:
            map_plots.children[0] = Div(text='''You beat the AI!!! Congratulations!!!''', width=400,
                                        styles={'font-size': '200%'})            


    # Add callbacks to ai map
    ai_map.on_event(Tap, callback)

    # Make things pretty by adding a title and greeting
    # Display title
    title = Div(text="<b>Battleship! Powered by Convolutional Neural Networks!</b>",
                sizing_mode="scale_both", margin=(5, 5, 20, 5))

    # Display greeting message
    instructions1 = Div(text='''OBJECT OF THE GAME: Be the first to sink all 5 of your opponents ships. There are five 
                             ships with lengths of 5, 4, 3, 3, and 2 square spaces. The Enemy Map is covered with fog 
                             and the ships are initially hidden. Similarly, Your Map is initially covered in fog 
                             and the AI doesn't know where your ships are. You must fire shots into the fog on the 
                             Enemy Map to learn if there is a ship at that location.''',
                        width=800, margin=(5, 5, 20, 5))
    instructions2 = Div(text='''CALL YOUR SHOT!: On your turn, click on a target square on the Enemy Map. If there is 
                             a ship in that location, it was hit and will be indicated on the map in red. If not, it 
                             was a miss, and your shot sank into the bottomless depths of the sea.''',
                        width=800, margin=(5, 5, 20, 5))
    instructions3 = Div(text='''If you don't like the configuration of your ships, you can reload the page to generate 
                             new ones. Good luck!''',
                        width=800, margin=(5, 5, 20, 5))

    # Format front page layout
    return layout([[title], [instructions1], [instructions2], [instructions3], map_plots])

def run_battleship_app():
    """Create dashboard with tabs and widgets.
    """
    # Create game layout
    game_layout = get_game_tab()

    # Get biography layout
    bio_layout = get_about_me_tab()

    # Make initial panels for tabs
    tabs = Tabs(tabs=[TabPanel(child=game_layout, title='Battleship Game'),
                      TabPanel(child=bio_layout, title='About the Author')])

    return tabs

# Loads the model once to use everywhere
BM = BattleshipModel()
BM.load_state_dict(torch.load("model/Battleship_Model.pt"))
BM.eval()

# Creates all tabs
tabs = run_battleship_app()

# Add tabs to the document/app
curdoc().add_root(tabs)
curdoc().title = "Battleship App"