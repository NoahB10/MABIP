"""macOS adaptation for the MABIP GUI.

Every layout in this app was sized against the Pi rig: X11, a ~10 pt system
font, 24 px toolbar icons, DPR 1. Cocoa differs on all four, so the same widget
tree comes out too big for its window on a Mac — the tab strip crowds the
SHUTDOWN button, the matplotlib toolbar renders 32 px icons over a grey
gradient, and the window opens far smaller than its own sizeHint, squeezing the
plot.

Everything here is a no-op off Darwin, so the Pi keeps its current appearance
byte-for-byte. Import is safe from any platform.
"""
import sys

from PyQt5.QtCore import QRectF, QSize, Qt
from PyQt5.QtGui import QFont, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication

IS_MAC = sys.platform == "darwin"

# Cocoa's default UI font is 13 pt against the Pi's ~10 pt. Splitting the
# difference keeps text legible on a Retina panel while letting the fixed-width
# columns (268 px side panels, 12-column well grid) still fit their labels.
BASE_FONT_PT = 11

# matplotlib asks Qt for the style's PM_ToolBarIconSize, which is 32 on Cocoa
# and 24 on the Pi's style. 18 keeps the nav strip subordinate to the plot,
# which is the only thing on that tab anyone looks at.
NAV_ICON_PX = 18

# Leave room for the menu bar and Dock rather than filling the screen.
_SCREEN_MARGIN_W = 80
_SCREEN_MARGIN_H = 120
_MAX_W = 1500
_MAX_H = 950


def apply(app: QApplication) -> None:
    """Apply the Mac-only application-wide tweaks. Call once, right after the
    QApplication exists and before any window is constructed — the base font
    has to be in place before widgets compute their size hints."""
    if not IS_MAC:
        return

    font = app.font()
    font.setPointSizeF(BASE_FONT_PT)
    app.setFont(font)

    # flow_control_tab asks for the X11 family name "Monospace", which Cocoa
    # does not have; Qt then walks the whole font list to fail, logging
    # "Populating font family aliases took N ms" on every launch.
    QFont.insertSubstitutions("Monospace", ["Menlo", "Monaco", "Courier New"])


def tune_nav_toolbar(toolbar) -> None:
    """Shrink and flatten a matplotlib NavigationToolbar2QT.

    Cocoa gives QToolBar a 32 px icon size and a grey gradient background that
    spans the full tab width, so the strip reads as the loudest element on the
    Plotting tab. Neither is what the Pi shows."""
    if not IS_MAC:
        return

    toolbar.setIconSize(QSize(NAV_ICON_PX, NAV_ICON_PX))
    toolbar.setMovable(False)
    toolbar.setFloatable(False)
    toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
    # Name the background explicitly rather than using `transparent`:
    # matplotlib's icon engine decides black-vs-white glyphs from the toolbar's
    # background colour, and a transparent one reads as dark, so the icons come
    # out white on white and vanish.
    toolbar.setStyleSheet(
        "QToolBar { background: #f1f4f7; border: none; padding: 0px;"
        " spacing: 2px; }"
        "QToolBar::separator { background: #d0d7de; width: 1px;"
        " margin: 4px 6px; }"
        "QToolButton { border: none; border-radius: 4px; padding: 3px; }"
        "QToolButton:hover { background: #e7eef5; }"
        "QToolButton:pressed { background: #dfe7ef; }"
        "QLabel { color: #666; }"
    )
    _resharpen_nav_icons(toolbar)


def _resharpen_nav_icons(toolbar) -> None:
    """Redraw the nav icons from matplotlib's SVGs at the screen's real pixel
    density.

    matplotlib's own icon engine rasterises at logical size and lets Qt stretch
    the result, which on a 2x panel gives hard-aliased, slightly clipped glyphs
    — the blocky arrows and half-cut save icon. Rendering the same SVGs
    ourselves at size x devicePixelRatio, with the pixmap tagged at that ratio,
    puts a real 36 px bitmap behind an 18 px slot."""
    try:
        from pathlib import Path

        from PyQt5.QtSvg import QSvgRenderer
        from matplotlib import get_data_path
    except ImportError:
        return                       # leave matplotlib's icons alone

    images = Path(get_data_path()) / "images"
    dpr = toolbar.devicePixelRatioF() or 1.0
    by_text = {a.text(): a for a in toolbar.actions()}

    for text, _tip, image_file, _cb in type(toolbar).toolitems:
        action = by_text.get(text)
        if action is None or not image_file:
            continue
        svg = images / f"{image_file}.svg"
        if not svg.exists():
            continue

        renderer = QSvgRenderer(str(svg))
        if not renderer.isValid():
            continue

        side = int(round(NAV_ICON_PX * dpr))
        pixmap = QPixmap(side, side)
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        # Name the target rect in logical units. A pixmap tagged with a device
        # pixel ratio hands QPainter a logical coordinate system, but
        # QSvgRenderer.render(painter) alone targets the raw device viewport —
        # it would draw the icon at 2x and clip to the top-left quarter.
        renderer.render(painter, QRectF(0, 0, NAV_ICON_PX, NAV_ICON_PX))
        painter.end()

        action.setIcon(QIcon(pixmap))


def preferred_window_size(default_w: int, default_h: int) -> tuple:
    """Opening size for the main window.

    The stored defaults (1050x566) come from the Pi's screen. On a Mac the same
    tree wants ~1500x900, and Qt clamps the window to a minimum that leaves the
    plot cramped, so open closer to what the content actually asks for while
    staying inside the available screen."""
    if not IS_MAC:
        return default_w, default_h

    screen = QApplication.primaryScreen()
    if screen is None:
        return default_w, default_h

    avail = screen.availableGeometry()
    width = max(default_w, min(_MAX_W, avail.width() - _SCREEN_MARGIN_W))
    height = max(default_h, min(_MAX_H, avail.height() - _SCREEN_MARGIN_H))
    return width, height
