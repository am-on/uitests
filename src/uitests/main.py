from uitests.automation import UIDriver
from uitests.automation import get_asset_path

import structlog

log = structlog.get_logger()


if __name__ == "__main__":
    ui = UIDriver()
    ui.capture_screen()
    ui.find_by_text("Submit").click()
    icon = ui.find_by_icon(
        get_asset_path("icon/pareto-white-large.png"), target_height=41
    )
    if icon.exists:
        log.info(
            "Icon found",
            x=icon.x,
            y=icon.y,
            confidence=icon.confidence,
        )
    else:
        log.info("Icon not found")
    breakpoint()
