# Styles file


# Color variables for styling
RED_A = "rgb(221, 221, 221)"  # Light gray text color
GRAY_60 = "rgb(60, 60, 60)"    # Dark gray background
GRAY_40 = "rgb(40, 44, 52)"   # Medium dark gray
GRAY_33 = "rgb(33, 37, 43)"   # Background color
GRAY_44 = "rgb(44, 49, 58)"   # Border color
BLUE_60 = "rgb(60, 80, 180)"   # Blue highlight
BLUE_HIGHLIGHT = "rgb(60, 80, 180)"   # Blue highlight

# Tab-specific colors
TAB_BACKGROUND = "rgb(33, 37, 43)"
TAB_TEXT = "rgb(221, 221, 221)"
TAB_SELECTED = "rgb(60, 80, 180)"
TAB_BORDER = "rgb(44, 49, 58)"

# Tab widget stylesheet
TAB_WIDGET_STYLE = """
    QTabWidget::pane {
        border: 1px solid {TAB_BORDER};
        background-color: {TAB_BACKGROUND};
    }
    QTabBar::tab {
        background-color: {TAB_BACKGROUND};
        color: {TAB_TEXT};
        padding: 5px;
        font-size: 16px;
        border: none;
    }
    QTabBar::tab:selected {
        background-color: {TAB_SELECTED};
    }
    QTabBar::tab:hover {
        background-color: rgb(50, 70, 170);
    }
"""