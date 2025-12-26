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
    threshold_input = _find_component(section, "roc-pr-threshold-points")
    roc_graph = _find_component(section, "roc-curve")
    pr_graph = _find_component(section, "precision-recall-curve")
    neg_roc_graph = _find_component(section, "negative-roc-curve")
    neg_pr_graph = _find_component(section, "negative-precision-recall-curve")

    assert carrier_dropdown is not None, "Carrier dropdown should be present"
    assert hours_slider is not None, "Hours slider should be present"
    assert threshold_input is not None, "Threshold points input should be present"
    assert roc_graph is not None, "ROC graph should be present"
    assert pr_graph is not None, "Precision-recall graph should be present"
    assert neg_roc_graph is not None, "Negative ROC graph should be present"
    assert neg_pr_graph is not None, "Negative precision-recall graph should be present"


def test_roc_pr_threshold_points_accepts_arbitrary_values():
    section = layout._build_roc_pr_section()

    threshold_input = _find_component(section, "roc-pr-threshold-points")

    assert threshold_input is not None, "Threshold input should exist"
    assert threshold_input.min == 11, "Threshold input should allow values greater than 10"
    assert threshold_input.step == 1, "Threshold input should increment in single steps"


def test_roc_pr_graph_order_matches_grid_request():
    section = layout._build_roc_pr_section()

    def _collect_graph_ids(component: Component):
        ids = []

        def _walk(node):
            if isinstance(node, (list, tuple)):
                for child in node:
                    yield from _walk(child)
                return

            if getattr(node, "__class__", None) and node.__class__.__name__ == "Graph":
                ids.append(getattr(node, "id", None))

            children = getattr(node, "children", None)
            if children is not None:
                yield from _walk(children)

        list(_walk(component))
        return ids

    graph_ids = _collect_graph_ids(section)
    expected_order = [
        "roc-curve",
        "precision-recall-curve",
        "negative-precision-recall-curve",
        "negative-roc-curve",
    ]

    assert all(graph_id in graph_ids for graph_id in expected_order), "All graphs should be present"
    assert graph_ids[:4] == expected_order, "Graphs should render in the requested clockwise order"


def test_performance_overview_controls_and_table_present():
    section = layout._build_performance_overview_section()

    threshold_slider = _find_component(section, "performance-overview-threshold")
    carrier_dropdown = _find_component(section, "performance-overview-carrier")
    hours_range = _find_component(section, "performance-overview-hours-range")
    grid = _find_component(section, "performance-overview-grid")

    assert threshold_slider is not None, "Performance overview threshold slider should be present"
    assert carrier_dropdown is not None, "Performance overview carrier dropdown should be present"
    assert hours_range is not None, "Performance overview hours range slider should be present"
    assert grid is not None, "Performance overview grid should be present"


def test_threshold_metrics_controls_and_chart_present():
    section = layout._build_threshold_metrics_section()

    carrier_dropdown = _find_component(section, "threshold-metrics-carrier")
    hours_range = _find_component(section, "threshold-metrics-hours-range")
    threshold_input = _find_component(section, "threshold-metrics-threshold-points")
    metrics_selection = _find_component(section, "threshold-metrics-selection")
    graph = _find_component(section, "threshold-metrics-graph")

    assert carrier_dropdown is not None, "Threshold metrics carrier dropdown should be present"
    assert hours_range is not None, "Threshold metrics hours range slider should be present"
    assert threshold_input is not None, "Threshold points input should be present"
    assert metrics_selection is not None, "Metrics checklist should be present"
    assert graph is not None, "Threshold metrics graph should be present"
    assert metrics_selection.value == [
        "Accuracy",
        "F-Score",
        "Negative F-Score",
    ], "Default metric selection should be accuracy and F-scores"
