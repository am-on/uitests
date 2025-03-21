import cv2
import numpy as np
import mss
import time
import pytesseract
import typing as t
from pynput.mouse import Controller as MouseController
from pynput.mouse import Button
import structlog
from pathlib import Path

log = structlog.get_logger()


class ElementNotFoundError(Exception):
    """Raised when trying to interact with an element that was not found"""

    pass


class Element:
    def __init__(
        self,
        confidence: float,
        x: t.Optional[int] = None,
        y: t.Optional[int] = None,
        bounds: t.Optional[
            tuple[int, int, int, int]
        ] = None,  # (left, top, width, height)
    ) -> None:
        self.x = x
        self.y = y
        self.confidence = confidence
        self.bounds = bounds

        self._mouse = MouseController()

    @property
    def exists(self) -> bool:
        """Check if the element was found."""
        return self.x is not None and self.y is not None

    def _ensure_element_exists(self) -> None:
        """Helper method to check element existence before actions."""
        if not self.exists:
            raise ElementNotFoundError(
                "Cannot interact with element that was not found"
            )

    def _move_mouse_to_element(self, sleep: float = 0.5) -> None:
        """Move the mouse to the element's coordinates."""
        self._ensure_element_exists()
        self._mouse.position = (self.x, self.y)  # type: ignore
        time.sleep(sleep)

    def click(self, click_type: Button = Button.left, count: int = 1) -> "Element":
        """Click the element."""
        self._move_mouse_to_element()
        self._mouse.click(click_type, count=count)
        log.info("Clicked at coordinates", x=self.x, y=self.y)
        return self

    def double_click(self) -> "Element":
        """Double click the element."""
        return self.click(count=2)

    def right_click(self) -> "Element":
        """Right click the element."""
        return self.click(Button.right)


class UIDriver:
    def __init__(self, debug: bool = True) -> None:
        self.current_screen: t.Optional[np.ndarray] = None
        self.debug = debug
        self._debug_counter = 0

        # Get the scaling factor for Retina displays
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            self.scale_factor = monitor["width"] / sct.grab(monitor).width

    def capture_screen(self) -> np.ndarray:
        """Capture and store current screenshot."""
        with mss.mss() as sct:
            screenshot = sct.grab(sct.monitors[1])
            self.current_screen = np.array(screenshot)
        return self.current_screen

    def _save_debug_image(
        self,
        image: np.ndarray,
        description: str = "",
        *,
        found: bool = False,
        annotations: t.Optional[t.Dict] = None,
    ) -> None:
        """Save debug image with visualizations.

        Args:
            image: Base image to annotate
            description: Description for filename
            found: Whether the element was found
            annotations: Dict with visualization data:
                - rect: (x, y, w, h) for rectangle
                - text: Text to display
                - confidence: Confidence score to display
                - threshold: Confidence threshold
                - ocr_boxes: List of (x, y, w, h) for OCR results
        """
        debug_dir = Path(__file__).parent / "debug_screenshots"
        debug_dir.mkdir(exist_ok=True)

        debug_image = image.copy()

        if annotations:
            # Draw rectangle if coordinates provided
            if "rect" in annotations:
                x, y, w, h = annotations["rect"]
                color = (
                    (0, 255, 0) if found else (0, 0, 255)
                )  # Green if found, Red if not
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, 8)

                # Calculate unscaled center coordinates
                center_x = x + w // 2
                center_y = y + h // 2

                # Draw crosshair at center if element was found
                cv2.drawMarker(
                    debug_image,
                    (center_x, center_y),  # Use unscaled coordinates
                    (0, 0, 255),  # Red color
                    cv2.MARKER_CROSS,
                    20,
                    4,
                )

            # Draw OCR boxes if provided
            if "ocr_boxes" in annotations:
                for box in annotations["ocr_boxes"]:
                    x, y, w, h = box
                    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw text if provided
            debug_texts = []
            if "confidence" in annotations:
                threshold = annotations.get("threshold", 0)
                debug_texts.append(
                    f"Confidence: {annotations['confidence']:.3f} {'<' if not found else '>='} {threshold}"
                )
            if "text" in annotations:
                debug_texts.append(f"Searching for: {annotations['text']}")

            # If we have any text to display
            if debug_texts:
                # Text properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                font_thickness = 4
                padding = 10

                # Calculate text sizes and total height
                text_sizes = [
                    cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    for text in debug_texts
                ]
                text_heights = [size[1] for size in text_sizes]
                max_width = max(size[0] for size in text_sizes)
                total_height = sum(text_heights) + padding * (len(debug_texts) + 1)

                # Create background rectangle
                img_height, img_width = debug_image.shape[:2]
                overlay = debug_image.copy()
                bg_start_y = img_height - total_height
                cv2.rectangle(
                    overlay,
                    (0, bg_start_y),
                    (max_width + padding * 2, img_height),
                    (255, 255, 255),
                    -1,
                )

                # Add semi-transparency
                alpha = 0.95
                debug_image = cv2.addWeighted(overlay, alpha, debug_image, 1 - alpha, 0)

                # Draw texts
                current_y = bg_start_y + padding + text_heights[0]
                for i, (text, text_height) in enumerate(zip(debug_texts, text_heights)):
                    color = (0, 150, 0) if found else (0, 0, 150)
                    cv2.putText(
                        debug_image,
                        text,
                        (padding, current_y),
                        font,
                        font_scale,
                        color,
                        font_thickness,
                    )
                    current_y += text_height + padding

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        status = "found" if found else "not_found"
        filename = f"debug_{timestamp}_{self._debug_counter}_{status}_{description}.png"
        self._debug_counter += 1

        cv2.imwrite(str(debug_dir / filename), debug_image)
        log.debug("Saved debug screenshot", filename=filename)

    def find_by_icon(
        self,
        target_img_path: str,
        threshold: float = 0.8,
        target_height: t.Optional[int] = None,
        debug_description: str = "",
    ) -> Element:
        """Find an icon on the screen."""
        if self.current_screen is None:
            raise ValueError("No screenshot available. Call capture_screen first.")

        # Load template with alpha channel
        template = cv2.imread(target_img_path, cv2.IMREAD_UNCHANGED)
        if template is None:
            raise ValueError(f"Could not load image from path '{target_img_path}'")

        if target_height is not None:
            # Resize target image to given height
            current_height = template.shape[0]
            scale_factor = target_height / current_height
            new_width = int(template.shape[1] * scale_factor)
            template = cv2.resize(
                template, (new_width, target_height), interpolation=cv2.INTER_AREA
            )

        # Check if the image has an alpha channel
        has_alpha = template.shape[-1] == 4
        if has_alpha:
            # Split the image into BGR and alpha channels
            bgr = template[:, :, :3]
            alpha = template[:, :, 3].astype(np.uint8)

            # Create a mask from alpha channel
            mask = np.where(alpha > np.uint8(0), np.uint8(255), np.uint8(0))

            # Convert BGR to grayscale for template matching
            template_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # Apply the alpha mask to the grayscale template
            template_gray = cv2.bitwise_and(template_gray, template_gray, mask=mask)
        else:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        screen_gray = cv2.cvtColor(self.current_screen, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        x, y = max_loc
        w, h = template.shape[1], template.shape[0]
        center_x = int((x + w // 2) * self.scale_factor)
        center_y = int((y + h // 2) * self.scale_factor)

        if self.debug:
            self._save_debug_image(
                self.current_screen,
                debug_description,
                found=max_val >= threshold,
                annotations={
                    "rect": (x, y, w, h),
                    "confidence": max_val,
                    "threshold": threshold,
                    "text": f"Icon: {target_img_path}",
                },
            )

        if max_val >= threshold:
            log.info(
                "Found icon",
                path=target_img_path,
                x=center_x,
                y=center_y,
                confidence=max_val,
            )
            return Element(max_val, center_x, center_y)
        else:
            log.info("Icon not found", path=target_img_path, confidence=max_val)
            return Element(confidence=max_val)

    def _find_matching_text_element(
        self,
        data: dict,
        target_text: str,
    ) -> Element:
        """Search for matching text in OCR results."""
        target_words = [word for word in target_text.split()]
        target_words_count = len(target_words)

        for i in range(len(data["text"]) - target_words_count + 1):
            match = True
            for j in range(target_words_count):
                ocr_word = data["text"][i + j].strip()
                if ocr_word != target_words[j]:
                    match = False
                    break

            if match:
                # Calculate bounding box
                xs = [data["left"][i + j] for j in range(target_words_count)]
                ys = [data["top"][i + j] for j in range(target_words_count)]
                widths = [data["width"][i + j] for j in range(target_words_count)]
                heights = [data["height"][i + j] for j in range(target_words_count)]

                left_bound = min(xs)
                top_bound = min(ys)
                right_bound = max(x + w for x, w in zip(xs, widths))
                bottom_bound = max(y + h for y, h in zip(ys, heights))

                w = right_bound - left_bound
                h = bottom_bound - top_bound
                center_x = int((left_bound + right_bound) / 2 * self.scale_factor)
                center_y = int((top_bound + bottom_bound) / 2 * self.scale_factor)

                bounds = (left_bound, top_bound, w, h)
                return Element(1.0, center_x, center_y, bounds)

        return Element(confidence=0.0)

    def _preprocess_contrast_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """OCR the given image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        adjusted_image = clahe.apply(gray)
        return adjusted_image

    def _preprocess_binary_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """OCR the given image using binary thresholding.
        Creates high contrast black and white image for better text detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to get binary image
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2,  # Constant subtracted from mean
        )

        return binary

    def find_by_text(self, target_text: str, debug_description: str = "") -> Element:
        """Find text on screen using OCR."""
        if self.current_screen is None:
            raise ValueError("No screenshot available. Call capture_screen first.")

        element = Element(confidence=0.0)

        # Images with different preprocessing methods in hopes for better OCR
        # results
        adjusted_images = [
            self._preprocess_contrast_for_ocr(self.current_screen),
            self._preprocess_binary_for_ocr(self.current_screen),
        ]
        for count, image in enumerate(adjusted_images):
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
            )

            element = self._find_matching_text_element(data, target_text)

            if self.debug:
                debug_annotations: t.Dict[str, t.Any] = {}
                debug_annotations["ocr_boxes"] = [
                    (
                        data["left"][i],
                        data["top"][i],
                        data["width"][i],
                        data["height"][i],
                    )
                    for i in range(len(data["text"]))
                    if data["text"][i].strip()
                ]
                debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                debug_annotations["text"] = f"{target_text}; attempt: {count};"
                if element.bounds:
                    debug_annotations.update({"rect": element.bounds})

                self._save_debug_image(
                    debug_image,
                    debug_description,
                    found=element.exists,
                    annotations=debug_annotations,
                )

            if element.exists:
                # Element found, stop searching
                log.info("Found text", text=target_text, x=element.x, y=element.y)
                return element

        log.info("Text not found", text=target_text)
        return element


def get_asset_path(relative_path: str, assets_dir: str = "images") -> str:
    """Get the full path to requested asset.

    Function assumes that the assets folder is located in this script's directory or any parent directory.
    """
    current_path = Path(__file__).resolve().parent

    # Walk up the directory tree until we find the assets directory
    while True:
        if (current_path / assets_dir).exists():
            return str(current_path / assets_dir / relative_path)

        if current_path == current_path.parent:  # Reached root
            raise FileNotFoundError(
                f"Could not find '{assets_dir}' directory in path tree"
            )

        current_path = current_path.parent
