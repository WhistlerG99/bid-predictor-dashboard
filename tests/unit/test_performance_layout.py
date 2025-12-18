from dash.development.base_component import Component

from bid_predictor_ui.performance_tracker import layout


def _find_component(component: Component, target_id: str):
    """Recursively search for a Dash component by id."""

    def _walk(node):
        if isinstance(node, (list, tuple)):
            for child in node:
                yield from _walk(child)
            return

        yield node

        children = getattr(node, "children", None)
        if children is not None:
            yield from _walk(children)

    for candidate in _walk(component):
        if getattr(candidate, "id", None) == target_id:
            return candidate

    return None


def test_accept_prob_bin_count_input_updates_on_change():
    section = layout._build_distribution_section()

    input_component = _find_component(section, "accept-prob-bin-count")

    assert input_component is not None, "Bin count input should be present"
    assert input_component.debounce is False, "Bin input should emit changes immediately"
