from pynput.mouse import Listener
from pynput.mouse import Button
import structlog

log = structlog.get_logger()


def on_click(x: int, y: int, button: Button, pressed: bool) -> None:
    """Log coordinates when the mouse is clicked."""
    if pressed:
        # Only print when the button is pressed, not released
        log.info("Mouse clicked", x=x, y=y, button=button)


# Create and start the listener
with Listener(on_click=on_click) as listener:
    try:
        log.info("Monitoring mouse clicks... Press Ctrl+C to exit")
        listener.join()  # Keep the script running
    except KeyboardInterrupt:
        log.info("Monitoring stopped")
