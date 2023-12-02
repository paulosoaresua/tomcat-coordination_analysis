from __future__ import annotations
import uuid
import streamlit as st


class DropDown:
    """
    Creates a drop down that receives a list of drop down objects with a name and optional prefix.
    It also handles the logic to include a default option to avoid preselection of one of the
    values in the list of options.
    """

    DEFAULT_SELECTION_TEXT = "-- Select a value --"

    def __init__(self,
                 label: str,
                 key: str,
                 options: List[DropDownOption],
                 default_selection_text: str = DEFAULT_SELECTION_TEXT):
        """
        Creates a drop down widget.

        @param label: label of the widget.
        @param key: unique identifies of the component in the page.
        @param options: list of options to choose from.
        @param default_selection_text: text to show with the default option (first in the list).
        """
        self.label = label
        self.key = key
        self.options = options
        self.default_selection_text = default_selection_text

    def create(self) -> DropDownOption:
        """
        Creates the widget.

        @return: selected option.
        """
        def format_func(option: DropDownOption):
            """
            Define which text to display for an option in a dropdown.

            @param option: option of the list.
            @return: text for the option to be displayed in the dropdown.
            """
            if option:
                return str(option)
            else:
                return self.default_selection_text

        return st.selectbox(
            self.label,
            key=self.key,
            options=[None] + self.options,
            format_func=format_func
        )


class DropDownOption:
    """
    This class represents a dropdown option with an optional text prefix.
    """

    def __init__(self, name: str, prefix: Optional[str] = None):
        """
        Creates a dropdown option.

        @param name: option name.
        @param prefix: prefix to be prepended to the option.
        """
        self.prefix = prefix
        self.name = name

    def __repr__(self) -> str:
        """
        Gets a textual representation of the option with a prepended prefix if not undefined.

        @return: textual representation of the option.
        """
        if self.prefix:
            return f"{self.prefix} {self.name}"
        else:
            return self.name
