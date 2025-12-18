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


def test_performance_carrier_dropdown_present():
    controls = layout._build_control_panel()

    carrier_dropdown = _find_component(controls, "performance-carrier")

    assert carrier_dropdown is not None, "Carrier dropdown should be present in performance controls"


def test_roc_pr_controls_present():
    section = layout._build_roc_pr_section()

    carrier_dropdown = _find_component(section, "roc-pr-carrier")
    hours_slider = _find_component(section, "roc-pr-hours-range")
    roc_graph = _find_component(section, "roc-curve")
    pr_graph = _find_component(section, "precision-recall-curve")
    neg_roc_graph = _find_component(section, "negative-roc-curve")
    neg_pr_graph = _find_component(section, "negative-precision-recall-curve")

    assert carrier_dropdown is not None, "Carrier dropdown should be present"
    assert hours_slider is not None, "Hours slider should be present"
    assert roc_graph is not None, "ROC graph should be present"
    assert pr_graph is not None, "Precision-recall graph should be present"
    assert neg_roc_graph is not None, "Negative ROC graph should be present"
    assert neg_pr_graph is not None, "Negative precision-recall graph should be present"
