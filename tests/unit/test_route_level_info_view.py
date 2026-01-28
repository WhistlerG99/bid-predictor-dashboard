import importlib.util
from pathlib import Path


def load_route_level_view():
    repo_root = Path(__file__).resolve().parents[2]
    view_path = repo_root / "bid_predictor_ui" / "route_level_info" / "view.py"
    spec = importlib.util.spec_from_file_location("route_level_info_view", view_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_component_by_id(component, target_id):
    if getattr(component, "id", None) == target_id:
        return component

    children = getattr(component, "children", None)
    if children is None:
        return None

    if isinstance(children, (list, tuple)):
        for child in children:
            found = find_component_by_id(child, target_id)
            if found is not None:
                return found
        return None

    return find_component_by_id(children, target_id)


def test_route_level_info_download_button_present():
    module = load_route_level_view()
    tab = module.build_route_level_info_tab()

    download_component = find_component_by_id(tab, "routes-table-download")
    assert download_component is not None

    download_button = find_component_by_id(tab, "routes-table-download-button")
    assert download_button is not None
    assert download_button.title == "download data"
    assert "Download" in "".join(str(child) for child in download_button.children)
