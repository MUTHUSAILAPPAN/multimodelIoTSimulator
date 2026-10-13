"""
Streamlit Custom Component wrapper for Interactive Multi-View House.
Provides bidirectional communication between room viewer iframe and Streamlit Python.
"""
import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "room_component")
_room_viewer_component = components.declare_component("room_viewer_component", path=_COMPONENT_DIR)

def render_room_viewer(current_view=1, lights_on=True, visual_overlay=None, key="room_viewer_widget"):
    """
    Renders the interactive multi-view room viewer.

    Parameters:
    - current_view: Integer (1 to 4) indicating current camera view.
    - lights_on: Boolean indicating ceiling lights state.
    - visual_overlay: Dict with 'type' and 'object' for active incident effects.
    - key: Unique Streamlit component key.

    Returns:
    - Dict with event payload from JavaScript or None if no new event.
    """
    return _room_viewer_component(
        current_view=current_view,
        lights_on=lights_on,
        visual_overlay=visual_overlay,
        default=None,
        key=key,
    )
