import pytest
from uitests.automation import Element
from uitests.automation import ElementNotFoundError
from uitests.automation import UIDriver
from uitests.automation import get_asset_path
from pynput.mouse import Button
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path
import cv2


@pytest.fixture
def found_element() -> Element:
    return Element(confidence=0.95, x=100, y=200)


@pytest.fixture
def not_found_element() -> Element:
    return Element(confidence=0.32)


@pytest.fixture
def images_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(get_asset_path("", assets_dir="images"))


def load_screenshot(screenshot: Path) -> np.ndarray:
    """Load a test screenshot."""
    return cv2.imread(str(screenshot))


def test_element_exists_when_coordinates_present(found_element: Element) -> None:
    assert found_element.exists is True


def test_element_does_not_exist_when_coordinates_missing(
    not_found_element: Element,
) -> None:
    assert not_found_element.exists is False


def test_element_properties(found_element: Element) -> None:
    assert found_element.x == 100
    assert found_element.y == 200
    assert found_element.confidence == 0.95


def test_click_on_found_element(found_element: Element) -> None:
    mock_mouse = Mock()
    found_element._mouse = mock_mouse

    # Perform the click
    found_element.click()

    # Verify mouse position was set and click was called
    mock_mouse.position = (found_element.x, found_element.y)
    mock_mouse.click.assert_called_once_with(Button.left, count=1)


def test_click_on_not_found_element_raises_error(not_found_element: Element) -> None:
    with pytest.raises(ElementNotFoundError):
        not_found_element.click()


def test_double_click(found_element: Element) -> None:
    mock_mouse = Mock()
    found_element._mouse = mock_mouse

    found_element.double_click()
    mock_mouse.click.assert_called_once_with(Button.left, count=2)


def test_right_click(found_element: Element) -> None:
    mock_mouse = Mock()
    found_element._mouse = mock_mouse

    found_element.right_click()
    mock_mouse.click.assert_called_once_with(Button.right, count=1)


def test_capture_screen() -> None:
    with patch("mss.mss") as mock_mss:
        mock_screenshot = MagicMock()
        mock_screenshot.width = 1000
        mock_screenshot.height = 800
        mock_mss.return_value.__enter__.return_value.grab.return_value = mock_screenshot

        driver = UIDriver()
        screen = driver.capture_screen()

        assert isinstance(screen, np.ndarray)
        assert driver.current_screen is not None


def test_find_by_icon_without_capture_raises_error() -> None:
    driver = UIDriver()
    with pytest.raises(ValueError, match="No screenshot available"):
        driver.find_by_icon("dummy.png")


def test_find_by_icon_invalid_path() -> None:
    driver = UIDriver()
    driver.current_screen = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Could not load image"):
        driver.find_by_icon("nonexistent.png")


def test_find_by_text_without_capture_raises_error() -> None:
    driver = UIDriver()
    with pytest.raises(ValueError, match="No screenshot available"):
        driver.find_by_text("test text")


def test_scale_factor_calculation() -> None:
    with patch("mss.mss") as mock_mss:
        mock_monitor = {"width": 2000}
        mock_screenshot = MagicMock()
        mock_screenshot.width = 1000

        mock_mss.return_value.__enter__.return_value.monitors = [None, mock_monitor]
        mock_mss.return_value.__enter__.return_value.grab.return_value = mock_screenshot

        driver = UIDriver()
        assert driver.scale_factor == 2.0


@pytest.mark.parametrize(
    "screenshot_path, target_height, should_find",
    [
        ("mac/app-icon-passing.png", 40, True),
        ("mac/app-icon-passing-green.png", 40, True),
        ("mac/app-main-menu.png", 40, True),
        ("mac/app-icon-failing.png", 40, True),
        ("mac/no-app.png", 40, False),
    ],
)
def test_find_pareto_icon(
    images_dir: Path, screenshot_path: str, target_height: int, should_find: bool
) -> None:
    """Test finding an icon that exists in the screenshot."""
    driver = UIDriver()
    driver.current_screen = load_screenshot(images_dir / screenshot_path)
    icon_path = str(images_dir / "icon" / "pareto-white-large.png")
    element = driver.find_by_icon(icon_path, target_height=target_height)

    if should_find:
        assert element.exists
        assert isinstance(element.x, int)
        assert isinstance(element.y, int)
        assert element.confidence is not None
        assert element.confidence > 0.8  # High confidence for exact match
    else:
        assert not element.exists
        assert element.x is None
        assert element.y is None
        assert element.confidence is not None
        assert element.confidence < 0.6


@pytest.mark.parametrize(
    "icon_path, screenshot_path, target_height, should_find",
    [
        ("icon/cross-icon-highlighted.png", "mac/app-main-menu-sub-menu.png", 52, True),
        ("icon/cross-icon-highlighted.png", "mac/app-main-menu.png", 52, False),
    ],
)
def test_find_by_icon(
    images_dir: Path,
    icon_path: str,
    screenshot_path: str,
    target_height: int,
    should_find: bool,
) -> None:
    """Test finding an icon that exists in the screenshot."""
    driver = UIDriver()
    driver.current_screen = load_screenshot(images_dir / screenshot_path)
    icon_path = str(images_dir / icon_path)
    element = driver.find_by_icon(icon_path, target_height=target_height)

    if should_find:
        assert element.exists
        assert isinstance(element.x, int)
        assert isinstance(element.y, int)
        assert element.confidence is not None
        assert element.confidence > 0.8  # High confidence for exact match
    else:
        assert not element.exists
        assert element.x is None
        assert element.y is None
        assert element.confidence is not None
        assert element.confidence < 0.6


@pytest.mark.parametrize(
    "screenshot_path, search_text, should_find",
    [
        ("mac/app-main-menu.png", "Access Security", True),
        ("mac/app-main-menu.png", "Brave", True),
        ("mac/app-main-menu.png", "Quit Pareto", True),
        ("mac/app-main-menu.png", "Last check 2 minutes ago", True),
        ("mac/app-preferences-general.png", "Teams", True),
        ("mac/app-main-menu-sub-menu.png", "Application Updates", True),
        ("mac/app-main-menu.png", "NonexistentText", False),
    ],
)
def test_find_by_text(
    images_dir: Path, screenshot_path: str, search_text: str, should_find: bool
) -> None:
    """Test finding text that exists in the screenshot using OCR."""
    driver = UIDriver()
    driver.current_screen = load_screenshot(images_dir / screenshot_path)
    element = driver.find_by_text(search_text)

    if should_find:
        assert element.exists
        assert isinstance(element.x, int)
        assert isinstance(element.y, int)
        # OCR matches are binary - either found or not
        assert element.confidence == 1.0
    else:
        assert not element.exists
        assert element.x is None
        assert element.y is None
        assert element.confidence == 0.0


def test_get_asset_path_not_found() -> None:
    """Test FileNotFoundError when assets dir not found."""
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(
            FileNotFoundError,
            match="Could not find 'nonexistent' directory in path tree",
        ):
            get_asset_path("some_file.pngV", assets_dir="nonexistent")
