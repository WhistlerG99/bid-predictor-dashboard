from dash.development.base_component import Component

from bid_predictor_ui.acceptance_explorer import layout as acceptance_layout
from bid_predictor_ui.feature_sensitivity import layout as sensitivity_layout
from bid_predictor_ui.performance_tracker import layout as performance_layout
from bid_predictor_ui.snapshot import layout as snapshot_layout


def _find_component_with_class(component: Component, class_name: str):
    """Recursively search for a Dash component by class name."""

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
        class_value = getattr(candidate, "className", None)
        if class_value is None:
            continue
        if isinstance(class_value, str) and class_name in class_value.split():
            return candidate

    return None


def test_snapshot_layout_includes_responsive_classes():
    tab = snapshot_layout.build_snapshot_tab()

    assert _find_component_with_class(tab, "tab-flex") is not None
    assert _find_component_with_class(tab, "filter-card") is not None
    assert _find_component_with_class(tab, "graph-tall") is not None


def test_acceptance_layout_includes_responsive_classes():
    tab = acceptance_layout.build_acceptance_tab()

    assert _find_component_with_class(tab, "tab-flex") is not None
    assert _find_component_with_class(tab, "filter-card") is not None
    assert _find_component_with_class(tab, "graph-tall") is not None


def test_performance_layout_includes_responsive_classes():
    chart_grid = performance_layout._build_chart_grid()
    distribution_section = performance_layout._build_distribution_section()
    overview_section = performance_layout._build_performance_overview_section()
    roc_section = performance_layout._build_roc_pr_section()

    assert _find_component_with_class(chart_grid, "chart-grid") is not None
    assert _find_component_with_class(distribution_section, "split-panel") is not None
    assert _find_component_with_class(distribution_section, "side-panel") is not None
    assert _find_component_with_class(overview_section, "split-panel") is not None
    assert _find_component_with_class(overview_section, "side-panel") is not None
    assert _find_component_with_class(roc_section, "split-panel") is not None
    assert _find_component_with_class(roc_section, "side-panel") is not None
    assert _find_component_with_class(roc_section, "roc-pr-grid") is not None


def test_feature_sensitivity_layout_includes_responsive_classes():
    tab = sensitivity_layout.build_feature_sensitivity_tab()

    assert _find_component_with_class(tab, "tab-flex") is not None
    assert _find_component_with_class(tab, "filter-card") is not None
    assert _find_component_with_class(tab, "graph-tall") is not None
