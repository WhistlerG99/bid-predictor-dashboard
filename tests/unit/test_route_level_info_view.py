from bid_predictor_ui.route_level_info.view import build_route_level_info_tab


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
    tab = build_route_level_info_tab()

    download_component = find_component_by_id(tab, "routes-table-download")
    assert download_component is not None

    download_button = find_component_by_id(tab, "routes-table-download-button")
    assert download_button is not None
    assert download_button.children == "⬇️"
    assert download_button.title == "download data"
