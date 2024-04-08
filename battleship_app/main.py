from bokeh.layouts import layout
from bokeh.models import Tabs, TabPanel, Paragraph
from bokeh.models.widgets import Div
from bokeh.plotting import curdoc

def get_link_div(url:str, link_text:str):
    """Creates HTML Div for some links on bio page.

    Returns:
        HTML Div
    """
    html_link = """<p><a href="{}" target="_blank" rel="noopener noreferrer">{}</a></p>"""
    return Div(text=html_link.format(url, link_text), margin=(5, 5, 20, 5))

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
    github_div = get_link_div("https://github.com/jddeetz", 
                              "&#9758 My Github")
    li_div = get_link_div("https://www.linkedin.com/in/josh-deetz/", 
                          "&#9901 My LinkedIn")

    return layout([title, biography, mailto_div, resume_div, li_div, github_div])

def get_game_tab():
    # Make things pretty by adding a title and greeting
    # Display title
    title = Div(text="<b>Battleship!</b>", sizing_mode="scale_both", margin=(5, 5, 20, 5))

    # Display greeting message
    instructions = Div(text='''Here is what to do.''', width=400)

    # https://docs.bokeh.org/en/3.3.0/docs/examples/topics/categorical/periodic.html

    # Format front page layout
    return layout([title, instructions])

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

tabs = run_battleship_app()

curdoc().add_root(tabs)
curdoc().title = "Battleship App"