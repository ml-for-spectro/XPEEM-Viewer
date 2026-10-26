import sys
import os
import numpy as np
import h5py
import time
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QSlider,
    QGridLayout,
    QCheckBox,
    QRubberBand,
    QSpinBox,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QTextBrowser,
)

from PySide6.QtCore import Qt, QPoint, QRect, QSize, QSettings
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon
from scipy.ndimage import shift as ndi_shift, sobel as ndi_sobel, gaussian_filter, zoom as ndi_zoom
from skimage.registration import phase_cross_correlation
from PIL import Image

import numpy as np

__version__ = "0.1.0"
class HDF5Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"XPEEM Viewer v{__version__}")
        self.setGeometry(80, 60, 1500, 900)
        self.setMinimumSize(1100, 700)

        self.settings = QSettings("Synchrotron SOLEIL", "XPEEM Viewer")
        self.last_directory = self.settings.value("last_directory", ".")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # --- Top metadata label ---
        self.metadata_label = QLabel("No file loaded")
        self.image_label = QLabel(self)
        self.layout.addWidget(self.metadata_label)

        # --- Grouped controls ---
        # Buttons are grouped by task so the workflow is easier to understand.
        # Only the layout has changed; the existing callbacks and button objects
        # are kept unchanged.
        self.controls_layout = QGridLayout()

        # 1. File / export
        self.file_group = QGroupBox("File && Export")
        file_layout = QGridLayout(self.file_group)

        self.load_button = QPushButton("Load HDF5 / NXS")
        self.load_button.clicked.connect(self.load_file)
        file_layout.addWidget(self.load_button, 0, 0)

        self.load_tiff_button = QPushButton("Load TIFF Image / Stack")
        self.load_tiff_button.clicked.connect(self.load_tiff)
        file_layout.addWidget(self.load_tiff_button, 0, 1)

        self.save_tiff_button = QPushButton("Save Current TIFF")
        self.save_tiff_button.clicked.connect(self.save_current_tiff)
        self.save_tiff_button.setEnabled(False)
        file_layout.addWidget(self.save_tiff_button, 1, 0)


        self.sum_stack_button = QPushButton("Sum Stack → TIFF")
        self.sum_stack_button.clicked.connect(self.sum_stack_and_save_tiff)
        self.sum_stack_button.setEnabled(False)
        file_layout.addWidget(self.sum_stack_button, 1, 1)

        self.save_axis_button = QPushButton("Save as Axis HDF5")
        self.save_axis_button.clicked.connect(self.save_for_axis)
        self.save_axis_button.setEnabled(False)
        file_layout.addWidget(self.save_axis_button, 2, 0)

        self.save_processed_button = QPushButton("Save Processed Stack (HDF5)")
        self.save_processed_button.clicked.connect(self.save_processed_stack_hdf5)
        self.save_processed_button.setEnabled(False)
        file_layout.addWidget(self.save_processed_button, 2, 1)

        # 2. Image preparation / registration
        self.image_group = QGroupBox("Image Processing && Registration")
        image_layout = QGridLayout(self.image_group)

        self.crop_button = QPushButton("Crop ROI")
        self.crop_button.clicked.connect(self.enable_crop_mode)
        image_layout.addWidget(self.crop_button, 0, 0)

        self.drift_button = QPushButton("Drift Correction")
        self.drift_button.clicked.connect(self.open_drift_dialog)
        self.drift_button.setEnabled(False)
        image_layout.addWidget(self.drift_button, 0, 1)

        self.remove_norm_button = QPushButton("Normalize by TIFF")
        self.remove_norm_button.clicked.connect(self.remove_norm)
        image_layout.addWidget(self.remove_norm_button, 1, 0, 1, 2)

        # 3. ROI / spectroscopy
        self.spectroscopy_group = QGroupBox("ROI && Spectroscopy")
        spectroscopy_layout = QGridLayout(self.spectroscopy_group)

        self.roi_button = QPushButton("Select Polygon ROI Spectrum")
        self.roi_button.setCheckable(True)
        self.roi_button.clicked.connect(self.toggle_roi_selector)
        spectroscopy_layout.addWidget(self.roi_button, 0, 0)

        self.mean_button = QPushButton("Circular-Area Mean Spectrum")
        self.mean_button.clicked.connect(self.show_mean_spectrum)
        spectroscopy_layout.addWidget(self.mean_button, 0, 1)

        self.i0_button = QPushButton("I₀ Normalize Stack")
        self.i0_button.clicked.connect(self.start_i0_normalization)
        spectroscopy_layout.addWidget(self.i0_button, 1, 0, 1, 2)

        self.save_spectrum_button = QPushButton("Save Current Spectrum")
        self.save_spectrum_button.clicked.connect(self.save_roi_spectrum)
        spectroscopy_layout.addWidget(self.save_spectrum_button, 2, 0, 1, 2)

        # 4. XMCD workflow
        self.xmcd_group = QGroupBox("XMCD")
        xmcd_layout = QVBoxLayout(self.xmcd_group)

        self.align_xmcd_button = QPushButton("Align CL/CR → Save Stack")
        self.align_xmcd_button.clicked.connect(self.align_cl_cr_and_save_stack)
        xmcd_layout.addWidget(self.align_xmcd_button)

        self.xmcd_button = QPushButton("XMCD Analysis")
        self.xmcd_button.clicked.connect(self.run_xmcd_analysis)
        xmcd_layout.addWidget(self.xmcd_button)

        # 5. Application
        self.application_group = QGroupBox("Application")
        application_layout = QVBoxLayout(self.application_group)

        self.help_button = QPushButton("Processing Help")
        self.help_button.clicked.connect(self.show_processing_help)
        application_layout.addWidget(self.help_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        application_layout.addWidget(self.exit_button)
        application_layout.addStretch()

        # Two rows of clearly separated functional groups.
        self.controls_layout.addWidget(self.file_group, 0, 0)
        self.controls_layout.addWidget(self.image_group, 0, 1)
        self.controls_layout.addWidget(self.spectroscopy_group, 0, 2)
        self.controls_layout.addWidget(self.xmcd_group, 1, 0)
        self.controls_layout.addWidget(self.application_group, 1, 2)
        self.controls_layout.setColumnStretch(0, 1)
        self.controls_layout.setColumnStretch(1, 1)
        self.controls_layout.setColumnStretch(2, 1)

        self.layout.addLayout(self.controls_layout)

        # File label (show current loaded file)
        self.file_label = QLabel("File: None")
        self.layout.addWidget(self.file_label)

        # Keep existing spectra when plotting another ROI/mean/I0 spectrum.
        self.add_plot_checkbox = QCheckBox("Add Plot (keep existing spectra)")
        self.add_plot_checkbox.setChecked(False)
        self.add_plot_checkbox.setToolTip(
            "Checked: add the next spectrum to the existing plot. "
            "Unchecked: clear the spectrum panel before plotting."
        )
        self.layout.addWidget(self.add_plot_checkbox)

        # --- Main plotting canvas ---
        # Main plotting area: give the image more room than the spectrum/plot.
        # constrained_layout prevents labels/titles from squeezing either panel.
        self.canvas = FigureCanvas(
            Figure(figsize=(12, 6), constrained_layout=True)
        )
        grid = self.canvas.figure.add_gridspec(
            1, 2, width_ratios=[3, 2], wspace=0.12
        )
        self.ax_image = self.canvas.figure.add_subplot(grid[0, 0])
        self.ax_spectrum = self.canvas.figure.add_subplot(grid[0, 1])

        # Standard Matplotlib toolbar: Home, Back/Forward, Pan, Zoom and Save.
        # The Zoom tool works directly on the XPEEM image panel.
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.nav_toolbar)
        self.layout.addWidget(self.canvas, stretch=5)

        self.spectrum_canvas = self.canvas  # reuse the same canvas

        # --- Sliders for energy, x, y slices ---
        self.slider_layout = QGridLayout()

        self.energy_slider = QSlider(Qt.Horizontal)
        self.energy_slider.setMinimum(0)
        self.energy_slider.valueChanged.connect(self.update_image)
        self.slider_layout.addWidget(QLabel("Slice:"), 0, 0)
        self.slider_layout.addWidget(self.energy_slider, 0, 1)

        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setMinimum(0)
        self.x_slider.valueChanged.connect(self.update_x_slice)
        self.slider_layout.addWidget(QLabel("X (vertical) cross-section:"), 1, 0)
        self.slider_layout.addWidget(self.x_slider, 1, 1)

        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setMinimum(0)
        self.y_slider.valueChanged.connect(self.update_y_slice)
        self.slider_layout.addWidget(QLabel("Y (horizontal) cross-section:"), 2, 0)
        self.slider_layout.addWidget(self.y_slider, 2, 1)

        self.layout.addLayout(self.slider_layout)

        # --- Other initializations ---
        self.crop_mode = False
        self.crop_rect = None
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.image_label)
        self.origin = QPoint()

        self.data = None
        self.original_data = None
        self.pre_drift_data = None
        self.drift_shifts = None
        self.drift_reference_idx = None
        self.drift_algorithm = None
        self.drift_details = {}
        self.last_normalized_image = None
        self.xmcd_image = None
        self.aligned_cl_cr_stack = None
        self.cl_cr_shift = None
        self.energy = None
        self.axis_is_energy = False
        self.slice_type = "energy"
        self.roi_selector = None
        self.polygon_vertices = []
        self.polygon_line = None
        self.polygon_patch = None
        self.polygon_cid = None
        self.polygon_mode = None
        self.i0_spectrum = None
        self.i0_vertices = None
        self.i0_normalization_mode = None
        self.current_color_index = 0
        self.color_cycle = [
            "red",
            "blue",
            "green",
            "magenta",
            "orange",
            "cyan",
            "black",
        ]

    def _discover_hdf5_stack(self, h5file):
        """Discover the most plausible 3-D image stack and a matching 1-D axis."""
        datasets = []
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append((name, obj))
        h5file.visititems(visitor)

        candidates = []
        for path, ds in datasets:
            if ds.ndim != 3 or not np.issubdtype(ds.dtype, np.number):
                continue
            shape = tuple(ds.shape)
            # XPEEM images should have two substantial spatial dimensions.
            dims_sorted = sorted(shape, reverse=True)
            if dims_sorted[1] < 32:
                continue
            score = 0
            low = path.lower()
            if low.endswith('/data') or '/data_' in low: score += 6
            if 'image' in low: score += 5
            if 'xpeem' in low: score += 5
            if 'scan_data' in low: score += 3
            if 'detector' in low: score += 2
            # Prefer stacks with one dimension much smaller than image dimensions.
            if min(shape) <= 500: score += 2
            candidates.append([score, path, ds, None, None])

        if not candidates:
            found = '\n'.join(f"  {p}: shape={d.shape}" for p, d in datasets if d.ndim >= 2)
            raise ValueError("No plausible 3-D image stack was found.\nDatasets inspected:\n" + found[:5000])

        # Search globally for 1-D numeric axes whose length matches a stack dimension.
        axis_words = ('energy', 'actuator_1_1', 'photon', 'energy_axis')
        for c in candidates:
            _, data_path, ds, _, _ = c
            best_axis = None
            best_axis_score = -1
            best_axis_dim = None
            for axis_path, axis_ds in datasets:
                if axis_ds.ndim != 1 or not np.issubdtype(axis_ds.dtype, np.number):
                    continue
                for dim, n in enumerate(ds.shape):
                    if axis_ds.shape[0] != n:
                        continue
                    alow = axis_path.lower()
                    ascore = 0
                    # Only accept a 1-D channel as an energy axis when its
                    # name explicitly identifies it as energy. This prevents
                    # machine-current channels (~500 mA), timestamps, etc. from
                    # being displayed as photon energy merely because their
                    # length matches the number of frames.
                    is_energy_axis = any(w in alow for w in axis_words)
                    if not is_energy_axis:
                        continue

                    ascore += 10
                    # Prefer an energy axis in the same parent/group as image data.
                    if axis_path.rsplit('/', 1)[0] == data_path.rsplit('/', 1)[0]:
                        ascore += 5
                    if 'sample_x' in alow or 'sample_y' in alow:
                        ascore -= 8
                    if ascore > best_axis_score:
                        best_axis_score, best_axis, best_axis_dim = ascore, axis_path, dim
            if best_axis is not None:
                c[0] += max(best_axis_score, 0)
                c[3], c[4] = best_axis, best_axis_dim

        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0]

        # If several candidates are equally plausible, let the user choose explicitly.
        tied = [c for c in candidates if c[0] == best[0]]
        if len(tied) > 1:
            chooser = QDialog(self)
            chooser.setWindowTitle("Choose XPEEM image dataset")
            form = QFormLayout(chooser)
            combo = QComboBox()
            for c in tied:
                combo.addItem(f"{c[1]}   shape={tuple(c[2].shape)}", c[1])
            form.addRow("Several image stacks were found:", combo)
            bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            form.addRow(bb); bb.accepted.connect(chooser.accept); bb.rejected.connect(chooser.reject)
            if chooser.exec() != QDialog.Accepted:
                raise RuntimeError("Dataset selection cancelled.")
            chosen = combo.currentData()
            best = next(c for c in tied if c[1] == chosen)

        _, data_path, ds, axis_path, axis_dim = best
        data = np.asarray(ds[()])
        if axis_path is not None:
            energy = np.asarray(h5file[axis_path][()]).squeeze()
            if axis_dim != 0:
                data = np.moveaxis(data, axis_dim, 0)
            axis_is_energy = True
        else:
            # Without an identified energy axis, assume the smallest dimension is stack index.
            axis_dim = int(np.argmin(data.shape))
            if axis_dim != 0:
                data = np.moveaxis(data, axis_dim, 0)
            energy = np.arange(data.shape[0], dtype=float)
            axis_path = "Frame number (no explicit energy axis identified)"
            axis_is_energy = False

        return data, energy, data_path, axis_path, axis_is_energy

    def show_processing_help(self):
        """Show practical processing guidance for the main XPEEM workflows."""
        dialog = QDialog(self)
        dialog.setWindowTitle("XPEEM Viewer — Processing Help")
        dialog.resize(900, 720)

        layout = QVBoxLayout(dialog)
        browser = QTextBrowser(dialog)
        browser.setOpenExternalLinks(False)
        browser.setHtml("""
        <h1>XPEEM Viewer — Processing Guide</h1>
        <p>This guide describes the intended order of operations in the GUI.
        Keep the original files unchanged and save processed stacks under new names.</p>

        <h2>1. XMCD workflow</h2>
        <ol>
          <li><b>Drift-correct the CL stack.</b> Load the CL HDF5/NXS stack and use
          <i>Drift Correction</i>. Inspect the calculated X/Y shift curves before accepting.</li>
          <li><b>Drift-correct the CR stack</b> in the same way.</li>
          <li><b>Drift-correct the normalization stack</b> using the same care.</li>
          <li>With the corrected normalization stack loaded, choose
          <b>Sum Stack → TIFF</b>. Save the resulting 32-bit TIFF.</li>
          <li>Reload the corrected CL stack and choose <b>Normalize by TIFF</b>.
          Select the summed normalization TIFF. Repeat for the corrected CR stack.</li>
          <li>Save or retain the normalized CL and CR results as required.</li>
          <li>For CL/CR registration, use <b>Align CL/CR → Save Stack</b>.
          This aligns CR to CL and saves a two-frame floating-point TIFF
          (frame 0 = CL; frame 1 = aligned CR).</li>
          <li>Use <b>XMCD Analysis</b> either with that aligned two-frame TIFF or
          with separate CL and CR TIFF images. XMCD is calculated as
          <b>(CL − CR)/(CL + CR)</b>.</li>
        </ol>
        <p><b>Important:</b> CL and CR should be normalized consistently before calculating
        XMCD. Registration should be checked visually because residual spatial misalignment
        can generate strong edge artefacts in an XMCD image.</p>

        <h2>2. Drift correction</h2>
        <p>The viewer provides two independent translation-only registration algorithms so
        their results can be compared without changing the rest of the processing workflow.</p>

        <h3>Algorithm A — Phase cross-correlation (current method)</h3>
        <p>This uses <b>scikit-image phase_cross_correlation</b> with subpixel Fourier
        upsampling. Three reference strategies are available:</p>
        <p><b>Fixed reference:</b> every frame is registered directly to the selected
        reference frame. This avoids accumulated errors and is useful when image contrast
        remains sufficiently similar throughout the stack.</p>
        <p><b>Sequential:</b> neighbouring frames are registered and the shifts accumulated.
        This is useful when contrast evolves gradually, but small registration errors can accumulate.</p>
        <p><b>Rolling reference:</b> nearby already-aligned frames form a median reference.
        A window of 5 is a useful starting point for noisy stacks with gradual contrast changes.</p>

        <h3>Algorithm B — Igor/SPELEEM-style FFT correlation</h3>
        <p>This reproduces the workflow used by the legacy Igor Pro XPEEM analysis procedure:
        the selected ROI (or full image) is mean-subtracted, optionally edge/sharpen filtered,
        multiplied by a sine window, optionally interpolated to 2×/4×/6×/8× sampling, and
        registered by the peak of an FFT cross-correlation.</p>
        <p>Alignment starts from the selected reference slice and proceeds outward in both
        directions. After each newly aligned frame, the reference is updated as the
        <b>progressive mean of already aligned images</b>. Each frame is iteratively refined
        (up to 8 registrations), and the complete procedure is run for a second refinement pass.
        Fractional shifts are finally applied to the original full-resolution images by
        interpolation.</p>
        <p><b>Main difference:</b> Algorithm A estimates subpixel translations directly using
        phase correlation and offers fixed/sequential/rolling references. Algorithm B uses
        ordinary FFT cross-correlation on an interpolated, windowed image and a progressively
        averaged reference, closely following the older Igor/SPELEEM workflow.</p>

        <h3>Registration signal</h3>
        <ul>
          <li><b>Intensity:</b> use when contrast is stable between frames.</li>
          <li><b>Sobel edges:</b> a useful starting choice when XPEEM intensity or spectral
          contrast changes while structural edges remain stable.</li>
          <li><b>High-pass:</b> emphasizes fine spatial structure and suppresses slowly varying background.</li>
          <li><b>Gradient X / Y:</b> useful when the most stable features have a preferred direction.</li>
        </ul>
        <p><b>ROI registration:</b> choose <i>Select ROI on current image</i> when a stable
        feature is available. Zoom first if necessary; starting drift ROI mode automatically
        turns off the toolbar Zoom/Pan tool. The ROI is used only to calculate the translation;
        the resulting shift is applied to the complete image.</p>
        <p>A subpixel upsample value of <b>20</b> is a reasonable default. Always inspect the
        shift curves and the corrected stack before accepting the result.</p>

        <h2>3. Normalization</h2>
        <p><b>Normalize by TIFF</b> divides every image in the loaded stack by the selected
        TIFF normalization image. Use a normalization TIFF derived from the appropriate
        normalization acquisition and processed consistently with the data.</p>
        <p>Pixels where the normalization image is zero are excluded (NaN) rather than divided by zero.</p>

        <h2>4. ROI spectroscopy</h2>
        <p><b>Add Plot (keep existing spectra)</b>: leave this unchecked to replace the
        spectrum currently shown in the right panel. Check it when you want to overlay
        successive ROI, mean or I₀ spectra for direct comparison.</p>
        <p>Click <b>Select Polygon ROI Spectrum</b>. Left-click successive points around the material/region of interest and <b>double-click the final point</b> to close the polygon.
        The GUI averages all pixels inside the polygon through the complete stack and plots the resulting spectrum. Click <b>Select Polygon ROI Spectrum</b> again to leave ROI mode and remove the polygon; the plotted spectrum remains.</p>
        <p><b>Circular-Area Mean Spectrum</b> averages the central circular image area
        (972 pixel diameter). <b>Save Current Spectrum</b> saves the currently calculated
        ROI or circular-area spectrum.</p>

        <h2>5. I₀ normalization for NEXAFS</h2>
        <p>Use <b>I₀ Normalize Stack</b> when the image contains a clean reference/background
        region without the material of interest. Left-click points around that background region
        and <b>double-click the final point</b> to close the arbitrary polygon.</p>
        <p>The program calculates one background value per frame,
        <b>I₀(E)</b>, by averaging all pixels inside the polygon. The I₀ spectrum is displayed
        before the stack is modified. You can then choose:</p>
        <ul>
          <li><b>Transmission I/I₀</b></li>
          <li><b>Optical Density −ln(I/I₀)</b></li>
          <li><b>Cancel</b> to keep the extracted I₀ spectrum without modifying the stack.</li>
        </ul>
        <p>When the processed stack is saved as HDF5, the I₀ spectrum, polygon vertices and
        selected normalization mode are stored in the processing metadata.</p>

        <h2>6. Zoom and navigation</h2>
        <p>Use the Matplotlib toolbar above the images. The magnifying-glass tool zooms into
        a rectangle, Pan moves around a zoomed image, and Home restores the complete view.
        ROI and drift-selection modes automatically disable Pan/Zoom so mouse dragging can
        be used for rectangle selection.</p>

        <h2>7. Loading TIFF images</h2>
        <p><b>Load TIFF Image / Stack</b> opens either a single grayscale TIFF or a
        multi-page grayscale TIFF stack. A single image is treated as a one-frame stack.
        Multi-page TIFFs can be browsed and processed like other image stacks. Because a
        generic TIFF normally contains no photon-energy metadata, its first axis is shown
        as <b>Frame number</b>, not energy.</p>

        <h2>8. Saving processed data</h2>
        <p><b>Save Processed Stack (HDF5)</b> is the preferred HDF5 export after operations
        such as drift correction or normalization because it preserves floating-point data
        and NaN values. <b>Save Current TIFF</b> saves the currently displayed image, while
        <b>Sum Stack → TIFF</b> sums all frames of the current processed stack.</p>

        <h2>Suggested drift starting points for energy-dependent XPEEM</h2>
        <p><b>Phase correlation:</b> Rolling reference + ROI + Sobel edges + rolling window 5
        + subpixel 20 is a useful starting configuration when spectral contrast changes significantly.</p>
        <p><b>Igor/SPELEEM FFT:</b> ROI + Intensity or Find edges + 4× interpolation is a useful
        starting point. Use the same reference slice when comparing the two algorithms, and inspect
        both the corrected images and X/Y shift curves before accepting either result.</p>
        """)

        layout.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.clicked.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def load_tiff(self):
        """Load a single TIFF image or a multi-page TIFF stack."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load TIFF Image / Stack", self.last_directory,
            "TIFF Images (*.tif *.tiff);;All Files (*)"
        )
        if not file_path:
            return

        try:
            img = Image.open(file_path)
            frames = []
            frame_index = 0
            while True:
                try:
                    img.seek(frame_index)
                    frames.append(np.asarray(img, dtype=np.float32).copy())
                    frame_index += 1
                except EOFError:
                    break

            if not frames:
                raise ValueError("No readable image frames were found in the TIFF.")

            shape0 = frames[0].shape
            if len(shape0) != 2:
                raise ValueError(
                    "Only grayscale 2-D TIFF images/stacks are supported. "
                    f"Found frame shape {shape0}."
                )
            if any(frame.shape != shape0 for frame in frames):
                raise ValueError("All TIFF frames must have the same dimensions.")

            # Keep the internal representation consistently (frames, y, x).
            self.data = np.stack(frames, axis=0).astype(np.float32, copy=False)
            self.energy = np.arange(self.data.shape[0], dtype=float)
            self.axis_is_energy = False
            self.slice_type = "energy"
            self.current_file = file_path
            self.last_directory = os.path.dirname(file_path)
            self.settings.setValue("last_directory", self.last_directory)

            # Reset state inherited from a previously loaded dataset.
            self.drift_shifts = None
            self.drift_reference_idx = None
            self.pre_drift_data = None
            self.current_spectrum = None
            if self.roi_selector:
                try:
                    self.roi_selector.set_active(False)
                    self.roi_selector.disconnect_events()
                except Exception:
                    pass
                self.roi_selector = None
            self.roi_button.setChecked(False)

            n, ny, nx = self.data.shape
            sliders = (self.energy_slider, self.x_slider, self.y_slider)
            for slider in sliders:
                slider.blockSignals(True)
            try:
                self.energy_slider.setRange(0, n - 1)
                self.energy_slider.setValue(0)
                self.x_slider.setRange(0, ny - 1)
                self.x_slider.setValue(ny // 2)
                self.y_slider.setRange(0, nx - 1)
                self.y_slider.setValue(nx // 2)
            finally:
                for slider in sliders:
                    slider.blockSignals(False)
            self.save_tiff_button.setEnabled(True)
            self.sum_stack_button.setEnabled(True)
            self.save_axis_button.setEnabled(True)
            self.save_processed_button.setEnabled(True)
            self.drift_button.setEnabled(n > 1)

            self.metadata_label.setText(
                f"Loaded TIFF: {os.path.basename(file_path)}\n"
                f"Shape: {self.data.shape}\n"
                f"{'Single image' if n == 1 else f'{n} frames'}"
            )
            self.update_image()

        except Exception as exc:
            QMessageBox.critical(
                self, "TIFF load error",
                f"Could not load the TIFF image/stack:\n{exc}"
            )

    def load_file(self):
        self.current_spectrum = None

        start_dir = self.last_directory if hasattr(self, "last_directory") else "."

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open HDF5 File", start_dir, "HDF5 Files (*.h5 *.hdf5 *.nxs)"
        )

        if not file_path:
            return

        self.filename = file_path
        self.file_label.setText(f"File: {os.path.basename(file_path)}")
        self.last_directory = os.path.dirname(self.filename)
        self.settings.setValue("last_directory", self.last_directory)

        self.save_axis_button.setEnabled(True)
        self.save_tiff_button.setEnabled(True)
        self.sum_stack_button.setEnabled(True)
        self.save_processed_button.setEnabled(True)

        try:
            with h5py.File(self.filename, "r") as f:
                self.data, self.energy, dataset_path, energy_path, self.axis_is_energy = self._discover_hdf5_stack(f)
                file_format = f"Discovered HDF5/NX stack: {dataset_path}"

                self.data = np.asarray(self.data)

                if self.data.ndim != 3:
                    raise ValueError(
                        f"Expected a 3D stack (energy, y, x), found {self.data.shape}."
                    )

                self.energy = np.asarray(self.energy).reshape(-1)
                if self.energy.size != self.data.shape[0]:
                    raise ValueError(
                        f"Energy axis has {self.energy.size} points but the stack "
                        f"contains {self.data.shape[0]} images."
                    )

            # Keep the complete XPEEM detector image.
            # No automatic circular field-of-view mask is applied on loading.

            self.original_data = self.data.copy()
            self.pre_drift_data = None
            self.drift_shifts = None
            self.drift_reference_idx = None
            self.drift_button.setEnabled(True)

            self.metadata_label.setText(
                f"File loaded: {self.filename}\n"
                f"Format: {file_format}\n"
                f"Axis: {energy_path}\n"
                f"Shape: {self.data.shape}\n"
                + (
                    f"Energy range: {self.energy.min():.2f}-{self.energy.max():.2f} eV"
                    if self.axis_is_energy
                    else f"Frames: 0-{self.data.shape[0]-1}"
                )
            )

        except Exception as e:
            self.metadata_label.setText(f"Failed to load: {e}")
            QMessageBox.critical(self, "HDF5 loading error", str(e))
            return

        # Initialise all sliders silently, then explicitly display a normal frame.
        # Without this, X/Y valueChanged signals can leave a long cross-section image.
        sliders = (self.energy_slider, self.x_slider, self.y_slider)
        for slider in sliders:
            slider.blockSignals(True)
        try:
            self.energy_slider.setRange(0, self.data.shape[0] - 1)
            self.energy_slider.setValue(self.data.shape[0] // 2)
            self.x_slider.setRange(0, self.data.shape[1] - 1)
            self.x_slider.setValue(self.data.shape[1] // 2)
            self.y_slider.setRange(0, self.data.shape[2] - 1)
            self.y_slider.setValue(self.data.shape[2] // 2)
        finally:
            for slider in sliders:
                slider.blockSignals(False)

        if not self.add_plot_checkbox.isChecked():
            self.ax_spectrum.clear()
            self.spectrum_canvas.draw()

        self.update_image()

    def _registration_image(self, image, preprocessing="Intensity"):
        """Prepare intensity/edge/high-pass representations for phase correlation."""
        arr = np.asarray(image, dtype=np.float32)
        finite = np.isfinite(arr)
        if not np.any(finite):
            raise ValueError("Registration image contains no finite pixels.")
        fill = float(np.nanmedian(arr))
        arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)

        if preprocessing == "Sobel edges":
            gx = ndi_sobel(arr, axis=1, mode="reflect")
            gy = ndi_sobel(arr, axis=0, mode="reflect")
            arr = np.hypot(gx, gy)
        elif preprocessing == "High-pass":
            arr = arr - gaussian_filter(arr, sigma=3.0, mode="reflect")
        elif preprocessing == "Gradient X":
            arr = ndi_sobel(arr, axis=1, mode="reflect")
        elif preprocessing == "Gradient Y":
            arr = ndi_sobel(arr, axis=0, mode="reflect")

        arr -= np.mean(arr)
        std = float(np.std(arr))
        if std > 0:
            arr /= std
        return arr

    def _igor_registration_image(self, image, preprocessing="Intensity"):
        """Prepare an image for the legacy Igor/SPELEEM FFT-correlation workflow."""
        arr = np.asarray(image, dtype=np.float32)
        finite = np.isfinite(arr)
        if not np.any(finite):
            raise ValueError("Registration image contains no finite pixels.")
        fill = float(np.nanmedian(arr))
        arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)

        # The original Igor routine offered no filter, edge finding and sharpening.
        if preprocessing in ("Find edges", "Sobel edges"):
            gx = ndi_sobel(arr, axis=1, mode="reflect")
            gy = ndi_sobel(arr, axis=0, mode="reflect")
            arr = np.hypot(gx, gy)
        elif preprocessing == "Sharpen":
            blurred = gaussian_filter(arr, sigma=1.0, mode="reflect")
            arr = arr + (arr - blurred)

        arr -= float(np.mean(arr))

        # Legacy 2-D sine window: suppress ROI boundary discontinuities before FFT.
        ny, nx = arr.shape
        wy = np.sin(np.pi * (np.arange(ny, dtype=np.float32) + 0.5) / ny)
        wx = np.sin(np.pi * (np.arange(nx, dtype=np.float32) + 0.5) / nx)
        return arr * wy[:, None] * wx[None, :]

    def _igor_fft_shift(self, reference, moving, interpolation=4, preprocessing="Intensity"):
        """Return (dy, dx) using windowed FFT cross-correlation at interpolated sampling."""
        factor = int(interpolation)
        ref = self._igor_registration_image(reference, preprocessing)
        mov = self._igor_registration_image(moving, preprocessing)

        if factor > 1:
            ref = ndi_zoom(ref, factor, order=1, mode="nearest", prefilter=False)
            mov = ndi_zoom(mov, factor, order=1, mode="nearest", prefilter=False)

        # IFFT(conj(FFT(reference))*FFT(moving)); the peak gives moving relative
        # to reference, so the applied correction has the opposite sign.
        corr = np.fft.ifft2(np.conj(np.fft.fft2(ref)) * np.fft.fft2(mov)).real
        peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
        dy, dx = float(peak[0]), float(peak[1])
        ny, nx = corr.shape
        if dy > ny / 2:
            dy -= ny
        if dx > nx / 2:
            dx -= nx
        return np.array([-dy / factor, -dx / factor], dtype=np.float64)

    def calculate_and_apply_igor_drift(self, reference_idx=None, roi=None,
                                       interpolation=4, preprocessing="Intensity",
                                       max_iterations=8, second_pass=True):
        """Legacy Igor/SPELEEM-style progressive-reference FFT drift correction."""
        if self.data is None:
            return

        stack = np.asarray(self.data, dtype=np.float32)
        n_images = stack.shape[0]
        if reference_idx is None:
            reference_idx = n_images // 2
        reference_idx = int(np.clip(reference_idx, 0, n_images - 1))
        interpolation = int(interpolation)
        max_iterations = int(max(1, max_iterations))

        def crop(frame):
            if roi is None:
                return frame
            x1, x2, y1, y2 = roi
            return frame[y1:y2, x1:x2]

        try:
            # Work on a temporary aligned stack. Total shifts always refer to the
            # original input stack, not to an already shifted temporary frame.
            aligned = stack.copy()
            total_shifts = np.zeros((n_images, 2), dtype=np.float64)

            passes = 2 if second_pass else 1
            for _pass in range(passes):
                # Re-anchor the selected reference for each pass.
                aligned[reference_idx] = ndi_shift(
                    stack[reference_idx], shift=total_shifts[reference_idx],
                    order=1, mode="constant", cval=np.nan, prefilter=False
                )

                # Forward: progressive mean reference.
                progressive = aligned[reference_idx].copy()
                count = 1
                for i in range(reference_idx + 1, n_images):
                    moving = aligned[i].copy()
                    incremental = np.zeros(2, dtype=np.float64)
                    for _ in range(max_iterations):
                        delta = self._igor_fft_shift(
                            crop(progressive), crop(moving),
                            interpolation=interpolation,
                            preprocessing=preprocessing
                        )
                        incremental += delta
                        if np.linalg.norm(delta) < (0.5 / interpolation):
                            break
                        moving = ndi_shift(
                            moving, shift=delta, order=1,
                            mode="constant", cval=np.nan, prefilter=False
                        )
                    total_shifts[i] += incremental
                    aligned[i] = ndi_shift(
                        stack[i], shift=total_shifts[i], order=1,
                        mode="constant", cval=np.nan, prefilter=False
                    )
                    valid_old = np.isfinite(progressive)
                    valid_new = np.isfinite(aligned[i])
                    numerator = np.nan_to_num(progressive) * count + np.nan_to_num(aligned[i])
                    denominator = valid_old.astype(np.float32) * count + valid_new.astype(np.float32)
                    progressive = np.divide(
                        numerator, denominator,
                        out=np.full_like(progressive, np.nan),
                        where=denominator > 0
                    )
                    count += 1

                # Backward: independent progressive mean starting at same reference.
                progressive = aligned[reference_idx].copy()
                count = 1
                for i in range(reference_idx - 1, -1, -1):
                    moving = aligned[i].copy()
                    incremental = np.zeros(2, dtype=np.float64)
                    for _ in range(max_iterations):
                        delta = self._igor_fft_shift(
                            crop(progressive), crop(moving),
                            interpolation=interpolation,
                            preprocessing=preprocessing
                        )
                        incremental += delta
                        if np.linalg.norm(delta) < (0.5 / interpolation):
                            break
                        moving = ndi_shift(
                            moving, shift=delta, order=1,
                            mode="constant", cval=np.nan, prefilter=False
                        )
                    total_shifts[i] += incremental
                    aligned[i] = ndi_shift(
                        stack[i], shift=total_shifts[i], order=1,
                        mode="constant", cval=np.nan, prefilter=False
                    )
                    valid_old = np.isfinite(progressive)
                    valid_new = np.isfinite(aligned[i])
                    numerator = np.nan_to_num(progressive) * count + np.nan_to_num(aligned[i])
                    denominator = valid_old.astype(np.float32) * count + valid_new.astype(np.float32)
                    progressive = np.divide(
                        numerator, denominator,
                        out=np.full_like(progressive, np.nan),
                        where=denominator > 0
                    )
                    count += 1

        except Exception as exc:
            QMessageBox.critical(
                self, "Igor/SPELEEM drift correction failed",
                f"Could not calculate image shifts:\n{exc}"
            )
            return

        self.pre_drift_data = self.data.copy()
        corrected = np.empty_like(stack)
        for i, shift_yx in enumerate(total_shifts):
            corrected[i] = ndi_shift(
                stack[i], shift=shift_yx, order=1,
                mode="constant", cval=np.nan, prefilter=False
            )

        self.data = corrected
        self.drift_shifts = total_shifts
        self.drift_reference_idx = reference_idx
        self.drift_algorithm = "Igor/SPELEEM FFT correlation"
        self.drift_details = {
            "interpolation": interpolation,
            "preprocessing": preprocessing,
            "max_iterations": max_iterations,
            "second_pass": bool(second_pass),
        }
        self.update_image()
        self.show_drift_plot()

        magnitudes = np.sqrt(np.sum(total_shifts ** 2, axis=1))
        max_shift = float(np.nanmax(magnitudes))
        roi_text = "full image" if roi is None else f"ROI x={roi[0]}:{roi[1]}, y={roi[2]}:{roi[3]}"
        self.metadata_label.setText(
            "Drift correction applied.\n"
            f"Algorithm: Igor/SPELEEM FFT correlation; signal: {preprocessing}; area: {roi_text}\n"
            f"Reference slice: {reference_idx}; interpolation: {interpolation}×; "
            f"maximum shift: {max_shift:.3f} pixels"
        )

        reply = QMessageBox.question(
            self, "Drift correction",
            "Igor/SPELEEM-style drift correction has been applied and the shift curves are shown.\n\n"
            "Keep this correction?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if reply == QMessageBox.No:
            self.reset_drift_correction()

    def open_drift_dialog(self):
        if self.data is None:
            QMessageBox.warning(self, "No data loaded", "Please load an HDF5 stack first.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("XPEEM Drift Correction")
        form = QFormLayout(dialog)

        algorithm_box = QComboBox()
        algorithm_box.addItems([
            "Phase cross-correlation (current)",
            "Igor/SPELEEM FFT correlation (legacy)"
        ])
        form.addRow("Alignment algorithm:", algorithm_box)

        method_box = QComboBox()
        method_box.addItems(["Sequential", "Fixed reference", "Rolling reference"])
        form.addRow("Registration method:", method_box)

        area_box = QComboBox()
        area_box.addItems(["Full image", "Select ROI on current image"])
        form.addRow("Registration area:", area_box)

        # Keep Sobel and all existing preprocessing choices.
        prep_box = QComboBox()
        prep_box.addItems(["Intensity", "Sobel edges", "High-pass", "Gradient X", "Gradient Y"])
        form.addRow("Registration signal:", prep_box)

        ref_spin = QSpinBox()
        ref_spin.setRange(0, self.data.shape[0] - 1)
        ref_spin.setValue(self.energy_slider.value())
        form.addRow("Reference slice:", ref_spin)

        rolling_spin = QSpinBox()
        rolling_spin.setRange(2, min(25, max(2, self.data.shape[0])))
        rolling_spin.setValue(min(5, self.data.shape[0]))
        rolling_spin.setEnabled(False)
        form.addRow("Rolling window:", rolling_spin)

        accuracy_spin = QSpinBox()
        accuracy_spin.setRange(1, 100)
        accuracy_spin.setValue(20)
        form.addRow("Subpixel upsample:", accuracy_spin)

        igor_interp_box = QComboBox()
        igor_interp_box.addItems(["1×", "2×", "4×", "6×", "8×"])
        igor_interp_box.setCurrentText("4×")
        form.addRow("Igor interpolation:", igor_interp_box)

        igor_prep_box = QComboBox()
        igor_prep_box.addItems(["Intensity", "Find edges", "Sharpen"])
        form.addRow("Igor signal:", igor_prep_box)

        def update_rolling_control():
            rolling_spin.setEnabled(method_box.currentText() == "Rolling reference")

        def update_algorithm_controls():
            is_igor = algorithm_box.currentText().startswith("Igor/")
            method_box.setEnabled(not is_igor)
            rolling_spin.setEnabled((not is_igor) and method_box.currentText() == "Rolling reference")
            accuracy_spin.setEnabled(not is_igor)
            prep_box.setEnabled(not is_igor)
            igor_interp_box.setEnabled(is_igor)
            igor_prep_box.setEnabled(is_igor)

        method_box.currentTextChanged.connect(update_rolling_control)
        method_box.currentTextChanged.connect(update_algorithm_controls)
        algorithm_box.currentTextChanged.connect(update_algorithm_controls)
        update_algorithm_controls()

        info = QLabel(
            "Sobel edges is useful when XPEEM intensity/contrast changes with energy.\n"
            "ROI registration restricts alignment to a stable feature.\n"
            "Sequential compares neighbours; Fixed reference uses one frame;\n"
            "Rolling reference uses the median of nearby already-aligned frames.\n"
            "Igor/SPELEEM FFT uses a progressive averaged reference, sine window,\n"
            "interpolation-based subpixel correlation, iterative refinement and a second pass."
        )
        form.addRow(info)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        form.addRow(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        if dialog.exec() != QDialog.Accepted:
            return

        method_map = {
            "Sequential": "sequential",
            "Fixed reference": "reference",
            "Rolling reference": "rolling",
        }
        use_igor = algorithm_box.currentText().startswith("Igor/")
        if use_igor:
            kwargs = dict(
                reference_idx=ref_spin.value(),
                interpolation=int(igor_interp_box.currentText().replace("×", "")),
                preprocessing=igor_prep_box.currentText(),
                max_iterations=8,
                second_pass=True,
            )
        else:
            kwargs = dict(
                method=method_map[method_box.currentText()],
                reference_idx=ref_spin.value(),
                upsample_factor=accuracy_spin.value(),
                preprocessing=prep_box.currentText(),
                rolling_window=rolling_spin.value(),
            )

        def run_alignment(roi):
            if use_igor:
                self.calculate_and_apply_igor_drift(roi=roi, **kwargs)
            else:
                self.calculate_and_apply_drift(roi=roi, **kwargs)

        if area_box.currentText() == "Full image":
            run_alignment(roi=None)
        else:
            # RectangleSelector and the Matplotlib toolbar both consume mouse
            # drags. Turn off Pan/Zoom before activating the drift ROI selector.
            if hasattr(self, "nav_toolbar"):
                mode = str(getattr(self.nav_toolbar, "mode", "")).lower()
                if "zoom" in mode:
                    self.nav_toolbar.zoom()
                elif "pan" in mode:
                    self.nav_toolbar.pan()

            self.metadata_label.setText(
                "DRIFT ROI MODE: drag a rectangle with the LEFT mouse button "
                "on the image. Pan/Zoom has been switched off temporarily."
            )

            if self.roi_selector:
                try:
                    self.roi_selector.set_active(False)
                    self.roi_selector.disconnect_events()
                except Exception:
                    pass
                self.roi_selector = None

            def onselect(eclick, erelease):
                if (eclick.xdata is None or eclick.ydata is None or
                        erelease.xdata is None or erelease.ydata is None):
                    return

                x1, x2 = sorted([int(round(eclick.xdata)), int(round(erelease.xdata))])
                y1, y2 = sorted([int(round(eclick.ydata)), int(round(erelease.ydata))])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(self.data.shape[2], x2)
                y2 = min(self.data.shape[1], y2)

                if x2 - x1 < 8 or y2 - y1 < 8:
                    QMessageBox.warning(
                        self, "ROI too small",
                        "Choose a larger ROI for reliable registration."
                    )
                    return

                roi = (x1, x2, y1, y2)

                if self.roi_selector:
                    try:
                        self.roi_selector.set_active(False)
                        self.roi_selector.disconnect_events()
                    except Exception:
                        pass
                    self.roi_selector = None

                self.metadata_label.setText(
                    f"Drift ROI selected: x={x1}:{x2}, y={y1}:{y2}. "
                    "Calculating alignment..."
                )
                run_alignment(roi=roi)

            self.roi_selector = RectangleSelector(
                self.ax_image,
                onselect,
                useblit=False,
                button=[1],
                minspanx=8,
                minspany=8,
                spancoords="pixels",
                interactive=False,
                props=dict(facecolor="none", edgecolor="red", linewidth=2.0),
            )
            self.roi_selector.set_active(True)
            self.canvas.draw_idle()

    def calculate_and_apply_drift(self, method="sequential", reference_idx=None,
                                  upsample_factor=20, roi=None,
                                  preprocessing="Intensity", rolling_window=5):
        """Translation-only subpixel drift correction.

        Methods:
          sequential : register each frame to its neighbour and accumulate shifts.
          reference  : register every frame directly to one fixed reference.
          rolling    : register each new frame to the median of nearby frames that
                       have already been aligned into the reference coordinate system.

        ROI selection and Intensity/Sobel/High-pass/Gradient preprocessing are
        available for all three methods.  Preprocessing is used only to estimate
        shifts; the calculated translations are applied to the original images.
        """
        if self.data is None:
            return

        stack = np.asarray(self.data, dtype=np.float32)
        n_images = stack.shape[0]
        if reference_idx is None:
            reference_idx = n_images // 2
        reference_idx = int(np.clip(reference_idx, 0, n_images - 1))
        rolling_window = int(max(2, rolling_window))
        shifts = np.zeros((n_images, 2), dtype=np.float64)

        def crop(frame):
            if roi is None:
                return frame
            x1, x2, y1, y2 = roi
            return frame[y1:y2, x1:x2]

        def reg(frame):
            return self._registration_image(
                crop(frame), preprocessing=preprocessing
            )

        try:
            if method == "reference":
                ref = reg(stack[reference_idx])
                for i in range(n_images):
                    if i == reference_idx:
                        continue
                    shifts[i] = phase_cross_correlation(
                        ref, reg(stack[i]), upsample_factor=upsample_factor
                    )[0]

            elif method == "rolling":
                # Keep a temporary stack aligned to the coordinate system of
                # the selected reference.  The rolling reference is a median,
                # which reduces the influence of noise and one anomalous frame.
                aligned = np.full_like(stack, np.nan, dtype=np.float32)
                aligned[reference_idx] = stack[reference_idx]

                # Forward from the selected reference.
                for i in range(reference_idx + 1, n_images):
                    j0 = max(reference_idx, i - rolling_window)
                    previous = aligned[j0:i]
                    rolling_ref = np.nanmedian(previous, axis=0)

                    shift = phase_cross_correlation(
                        reg(rolling_ref),
                        reg(stack[i]),
                        upsample_factor=upsample_factor,
                    )[0]
                    shifts[i] = shift
                    aligned[i] = ndi_shift(
                        stack[i], shift=shift, order=1,
                        mode="constant", cval=np.nan, prefilter=False
                    )

                # Backward from the selected reference.
                for i in range(reference_idx - 1, -1, -1):
                    j1 = min(reference_idx + 1, i + rolling_window + 1)
                    following = aligned[i + 1:j1]
                    rolling_ref = np.nanmedian(following, axis=0)

                    shift = phase_cross_correlation(
                        reg(rolling_ref),
                        reg(stack[i]),
                        upsample_factor=upsample_factor,
                    )[0]
                    shifts[i] = shift
                    aligned[i] = ndi_shift(
                        stack[i], shift=shift, order=1,
                        mode="constant", cval=np.nan, prefilter=False
                    )

            else:  # sequential
                for i in range(reference_idx + 1, n_images):
                    inc = phase_cross_correlation(
                        reg(stack[i - 1]), reg(stack[i]),
                        upsample_factor=upsample_factor
                    )[0]
                    shifts[i] = shifts[i - 1] + inc

                for i in range(reference_idx - 1, -1, -1):
                    inc = phase_cross_correlation(
                        reg(stack[i + 1]), reg(stack[i]),
                        upsample_factor=upsample_factor
                    )[0]
                    shifts[i] = shifts[i + 1] + inc

        except Exception as exc:
            QMessageBox.critical(
                self, "Drift correction failed",
                f"Could not calculate image shifts:\n{exc}"
            )
            return

        self.pre_drift_data = self.data.copy()
        corrected = np.empty_like(stack)
        for i, shift_yx in enumerate(shifts):
            corrected[i] = ndi_shift(
                stack[i], shift=shift_yx, order=1,
                mode="constant", cval=np.nan, prefilter=False
            )

        self.data = corrected
        self.drift_shifts = shifts
        self.drift_reference_idx = reference_idx
        self.drift_algorithm = "Phase cross-correlation"
        self.drift_details = {
            "method": method,
            "upsample_factor": int(upsample_factor),
            "preprocessing": preprocessing,
            "rolling_window": int(rolling_window),
        }
        self.update_image()
        self.show_drift_plot()

        magnitudes = np.sqrt(np.sum(shifts ** 2, axis=1))
        max_shift = float(np.nanmax(magnitudes))
        roi_text = (
            "full image" if roi is None
            else f"ROI x={roi[0]}:{roi[1]}, y={roi[2]}:{roi[3]}"
        )
        method_text = method
        if method == "rolling":
            method_text += f" (window={rolling_window})"

        self.metadata_label.setText(
            f"Drift correction applied.\n"
            f"Method: {method_text}; signal: {preprocessing}; area: {roi_text}\n"
            f"Reference slice: {reference_idx}; maximum shift: {max_shift:.3f} pixels"
        )

        reply = QMessageBox.question(
            self, "Drift correction",
            "Drift correction has been applied and the shift curves are shown.\n\n"
            "Keep this correction?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply == QMessageBox.No:
            self.reset_drift_correction()

    def show_drift_plot(self):
        """Display calculated X and Y translations on the spectrum panel."""
        if self.drift_shifts is None:
            return

        self.ax_spectrum.clear()

        # Drift is a frame-to-frame measurement. Always plot against the
        # acquired frame index, irrespective of whether an energy axis exists.
        x_axis = np.arange(len(self.drift_shifts), dtype=int)

        self.ax_spectrum.plot(x_axis, self.drift_shifts[:, 1], label="X shift")
        self.ax_spectrum.plot(x_axis, self.drift_shifts[:, 0], label="Y shift")
        self.ax_spectrum.axhline(0, linewidth=0.8)
        self.ax_spectrum.set_title("Calculated XPEEM Drift")
        self.ax_spectrum.set_xlabel("Energy (eV)" if self.axis_is_energy else "Frame number")
        self.ax_spectrum.set_ylabel("Applied shift (pixels)")
        self.ax_spectrum.legend()
        self.spectrum_canvas.draw_idle()

    def reset_drift_correction(self):
        """Restore the stack to its state immediately before the last correction."""
        if self.pre_drift_data is None:
            QMessageBox.information(
                self, "Drift correction", "There is no drift correction to reset."
            )
            return
        self.data = self.pre_drift_data.copy()
        self.pre_drift_data = None
        self.drift_shifts = None
        self.drift_reference_idx = None
        self.drift_algorithm = None
        self.drift_details = {}
        self.update_image()
        self.metadata_label.setText("Last drift correction was reset.")

    def save_roi_spectrum(self):
        if self.data is None or self.energy is None:
            self.metadata_label.setText("No data to save.")
            return
        if not hasattr(self, "current_spectrum") or self.current_spectrum is None:
            QMessageBox.information(
                self, "No spectrum",
                "Draw an ROI first, or display the mean spectrum, before saving."
            )
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Current Spectrum", self.last_directory, "Text Files (*.txt)"
        )
        if file_path:
            self.last_directory = os.path.dirname(file_path)
            axis_name = "Energy (eV)" if self.axis_is_energy else "Frame number"
            with open(file_path, "w") as f:
                f.write(f"# {axis_name}, Intensity\n")
                for x, intensity in zip(self.energy, self.current_spectrum):
                    f.write(f"{x:.6g}, {intensity:.6f}\n")
            self.metadata_label.setText(f"Spectrum saved to {file_path}")

    def update_image(self):
        idx = self.energy_slider.value()
        if self.data is not None:
            self.slice_type = "energy"
            self.ax_image.clear()
            self.ax_image.imshow(self.data[idx, :, :], cmap="gray", origin="lower")
            if self.axis_is_energy:
                self.ax_image.set_title(f"Frame #{idx} — {self.energy[idx]:.2f} eV")
            else:
                self.ax_image.set_title(f"Frame #{idx}")
            self.canvas.draw()

    def update_x_slice(self):
        if self.roi_button.isChecked():
            self.roi_button.setChecked(False)
            self._remove_roi_selector()
        idx = self.x_slider.value()
        if self.data is not None:
            self.slice_type = "x"
            self.ax_image.clear()
            self.ax_image.imshow(self.data[:, idx, :], cmap="gray", origin="lower")
            self.ax_image.set_title(f"X cross-section @ x={idx}")
            self.canvas.draw()

    def update_y_slice(self):
        if self.roi_button.isChecked():
            self.roi_button.setChecked(False)
            self._remove_roi_selector()
        idx = self.y_slider.value()
        if self.data is not None:
            self.slice_type = "y"
            self.ax_image.clear()
            self.ax_image.imshow(self.data[:, :, idx], cmap="gray", origin="lower")
            self.ax_image.set_title(f"Y cross-section @ y={idx}")
            self.canvas.draw()

    def _clear_polygon_roi(self):
        """Disconnect polygon picking and remove its visible artists."""
        if self.polygon_cid is not None:
            try:
                self.canvas.mpl_disconnect(self.polygon_cid)
            except Exception:
                pass
            self.polygon_cid = None
        for artist_name in ("polygon_line", "polygon_patch"):
            artist = getattr(self, artist_name, None)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                setattr(self, artist_name, None)
        self.polygon_vertices = []
        self.polygon_mode = None
        self.canvas.draw_idle()

    def _remove_roi_selector(self):
        """Remove any rectangular or polygon ROI selector and visible ROI."""
        if self.roi_selector is not None:
            try:
                self.roi_selector.set_active(False)
                self.roi_selector.disconnect_events()
                for artist in list(getattr(self.roi_selector, "artists", [])):
                    try:
                        artist.remove()
                    except Exception:
                        pass
            except Exception:
                pass
            self.roi_selector = None
        self._clear_polygon_roi()
        self.canvas.draw_idle()

    def _polygon_mask(self, vertices):
        """Return a boolean (y,x) mask for an arbitrary polygon."""
        ny, nx = self.data.shape[1:]
        yy, xx = np.mgrid[:ny, :nx]
        points = np.column_stack((xx.ravel(), yy.ravel()))
        path = MplPath(np.asarray(vertices, dtype=float))
        return path.contains_points(points, radius=0.01).reshape(ny, nx)

    def _start_polygon_selection(self, mode):
        if self.data is None:
            QMessageBox.warning(self, "No data loaded", "Please load an image stack first.")
            return False
        if self.slice_type != "energy":
            QMessageBox.information(
                self, "Polygon ROI",
                "Polygon ROIs must be selected on a normal XPEEM image frame."
            )
            return False

        if hasattr(self, "nav_toolbar"):
            toolbar_mode = str(getattr(self.nav_toolbar, "mode", "")).lower()
            if "zoom" in toolbar_mode:
                self.nav_toolbar.zoom()
            elif "pan" in toolbar_mode:
                self.nav_toolbar.pan()

        self._remove_roi_selector()
        self.polygon_mode = mode
        self.polygon_vertices = []
        self.metadata_label.setText(
            "POLYGON ROI: left-click successive boundary points; "
            "DOUBLE-CLICK the final point to close and apply the ROI."
        )

        def onclick(event):
            if event.inaxes is not self.ax_image or event.button != 1:
                return
            if event.xdata is None or event.ydata is None:
                return

            self.polygon_vertices.append((float(event.xdata), float(event.ydata)))

            # Draw/update the open polygon while points are being added.
            if self.polygon_line is not None:
                try:
                    self.polygon_line.remove()
                except Exception:
                    pass
            xs = [p[0] for p in self.polygon_vertices]
            ys = [p[1] for p in self.polygon_vertices]
            (self.polygon_line,) = self.ax_image.plot(
                xs, ys, "-o", linewidth=1.5, markersize=4
            )
            self.canvas.draw_idle()

            if event.dblclick:
                if len(self.polygon_vertices) < 3:
                    QMessageBox.warning(
                        self, "ROI too small",
                        "Select at least three polygon vertices before closing the ROI."
                    )
                    return

                vertices = list(self.polygon_vertices)
                if self.polygon_cid is not None:
                    self.canvas.mpl_disconnect(self.polygon_cid)
                    self.polygon_cid = None

                if self.polygon_line is not None:
                    try:
                        self.polygon_line.remove()
                    except Exception:
                        pass
                    self.polygon_line = None

                self.polygon_patch = MplPolygon(
                    vertices, closed=True, fill=False,
                    edgecolor="red", linewidth=2.0
                )
                self.ax_image.add_patch(self.polygon_patch)
                self.canvas.draw_idle()

                mask = self._polygon_mask(vertices)
                if not np.any(mask):
                    QMessageBox.warning(self, "Empty ROI", "The selected polygon contains no pixels.")
                    self._clear_polygon_roi()
                    return

                if mode == "spectrum":
                    self.plot_mask_spectrum(mask, label="Polygon ROI")
                    self.metadata_label.setText(
                        f"Polygon ROI spectrum calculated from {int(mask.sum())} pixels. "
                        "Click 'Select Polygon ROI Spectrum' again to remove the ROI."
                    )
                elif mode == "i0":
                    self._finish_i0_normalization(mask, vertices)

        self.polygon_cid = self.canvas.mpl_connect("button_press_event", onclick)
        return True

    def toggle_roi_selector(self):
        """Toggle arbitrary polygon ROI spectroscopy."""
        if not self.roi_button.isChecked():
            self._remove_roi_selector()
            self.metadata_label.setText(
                "ROI spectroscopy disabled. The plotted spectrum is retained."
            )
            return

        if not self._start_polygon_selection("spectrum"):
            self.roi_button.setChecked(False)

    def plot_mask_spectrum(self, mask, label="ROI"):
        if self.data is None:
            return
        pixels = self.data[:, mask]
        spectrum = np.nanmean(pixels, axis=1)
        self.current_spectrum = spectrum

        if not self.add_plot_checkbox.isChecked():
            self.ax_spectrum.clear()
        color = self.color_cycle[self.current_color_index % len(self.color_cycle)]
        self.current_color_index += 1
        self.ax_spectrum.plot(
            self.energy, spectrum,
            label=f"{label} {self.current_color_index}", color=color
        )
        self.ax_spectrum.set_title("Spectrum from ROI")
        self.ax_spectrum.set_xlabel(
            "Energy (eV)" if self.axis_is_energy else "Frame number"
        )
        self.ax_spectrum.set_ylabel("Intensity")
        self.ax_spectrum.legend()
        self.spectrum_canvas.draw()

    def start_i0_normalization(self):
        """Select an arbitrary background polygon and derive I0(frame)."""
        if self.data is None:
            QMessageBox.warning(self, "No data loaded", "Please load an image stack first.")
            return
        self.roi_button.setChecked(False)
        self._start_polygon_selection("i0")

    def _finish_i0_normalization(self, mask, vertices):
        i0 = np.nanmean(self.data[:, mask], axis=1)
        self.i0_spectrum = np.asarray(i0, dtype=np.float32)
        self.i0_vertices = np.asarray(vertices, dtype=np.float32)
        self.current_spectrum = self.i0_spectrum

        if not self.add_plot_checkbox.isChecked():
            self.ax_spectrum.clear()
        color = self.color_cycle[self.current_color_index % len(self.color_cycle)]
        self.current_color_index += 1
        self.ax_spectrum.plot(
            self.energy, self.i0_spectrum,
            label=f"I₀ ROI {self.current_color_index}", color=color
        )
        self.ax_spectrum.set_title("I₀ Spectrum from Background ROI")
        self.ax_spectrum.set_xlabel(
            "Energy (eV)" if self.axis_is_energy else "Frame number"
        )
        self.ax_spectrum.set_ylabel("I₀ intensity")
        self.ax_spectrum.legend()
        self.spectrum_canvas.draw()

        if np.any(~np.isfinite(self.i0_spectrum)) or np.any(self.i0_spectrum <= 0):
            QMessageBox.warning(
                self, "Invalid I₀",
                "The selected I₀ spectrum contains non-finite or non-positive values. "
                "Choose a different background region."
            )
            return

        box = QMessageBox(self)
        box.setWindowTitle("Apply I₀ normalization")
        box.setText(
            f"I₀ spectrum calculated from {int(mask.sum())} pixels.\n\n"
            "Choose the output representation:"
        )
        transmission_btn = box.addButton("Transmission I/I₀", QMessageBox.AcceptRole)
        od_btn = box.addButton("Optical Density −ln(I/I₀)", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()

        clicked = box.clickedButton()
        if clicked not in (transmission_btn, od_btn):
            self.metadata_label.setText(
                "I₀ spectrum calculated but stack was not modified."
            )
            return

        denom = self.i0_spectrum[:, None, None].astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            transmission = np.asarray(self.data, dtype=np.float32) / denom

        if clicked is transmission_btn:
            self.data = transmission.astype(np.float32, copy=False)
            self.i0_normalization_mode = "I/I0"
            mode_text = "Transmission I/I₀"
        else:
            valid = np.isfinite(transmission) & (transmission > 0)
            od = np.full_like(transmission, np.nan, dtype=np.float32)
            od[valid] = -np.log(transmission[valid])
            self.data = od
            self.i0_normalization_mode = "OD=-ln(I/I0)"
            mode_text = "Optical Density −ln(I/I₀)"

        self.update_image()
        self.metadata_label.setText(
            f"I₀ normalization applied: {mode_text}. "
            f"Background polygon contained {int(mask.sum())} pixels."
        )

    def enable_crop_mode(self):
        self.crop_mode = True
        self.metadata_label.setText("Draw a rectangle to crop the datacube.")

        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            self.crop_rect = (xmin, xmax, ymin, ymax)
            self.crop_mode = False
            self.apply_crop()

        if self.roi_selector:
            self._remove_roi_selector()
        self.roi_button.setChecked(False)

        self.roi_selector = RectangleSelector(
            self.ax_image,
            onselect,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        self.canvas.draw()

    def plot_roi_spectrum(self, xmin, xmax, ymin, ymax):
        if self.data is None:
            return

        # ROI spectroscopy is defined spatially on an XPEEM image (y, x).
        # It is deliberately not evaluated on x/energy or y/energy cross-sections,
        # where the rectangle axes no longer correspond to two spatial dimensions.
        if self.slice_type != "energy":
            QMessageBox.information(
                self, "ROI spectroscopy",
                "ROI spectra can be selected on a normal image frame only. "
                "Return to the frame/energy image view and draw the ROI there."
            )
            return

        roi_data = self.data[:, ymin:ymax, xmin:xmax]

        if not self.add_plot_checkbox.isChecked():
            self.ax_spectrum.clear()

        color = self.color_cycle[self.current_color_index % len(self.color_cycle)]
        self.current_color_index += 1

        spectrum = np.nanmean(roi_data, axis=(1, 2))

        self.current_spectrum = spectrum
        self.ax_spectrum.plot(
            self.energy, spectrum, label=f"ROI {self.current_color_index}", color=color
        )
        self.ax_spectrum.set_title("Spectrum from ROI")
        self.ax_spectrum.set_xlabel("Energy (eV)" if self.axis_is_energy else "Frame number")
        self.ax_spectrum.set_ylabel("Intensity")
        self.ax_spectrum.legend()
        self.spectrum_canvas.draw()

    def show_mean_spectrum(self):
        if self.data is None or self.energy is None:
            self.metadata_label.setText("No data loaded.")
            return

        # mean_spectrum = self.data.mean(axis=(1, 2))
        mean_spectrum = self.compute_mean_spectrum(self.data)
        self.current_spectrum = mean_spectrum

        if not self.add_plot_checkbox.isChecked():
            self.ax_spectrum.clear()

        color = self.color_cycle[self.current_color_index % len(self.color_cycle)]
        self.current_color_index += 1

        self.ax_spectrum.plot(
            self.energy,
            mean_spectrum,
            label=f"Mean {self.current_color_index}",
            color=color,
        )
        self.ax_spectrum.set_title("Mean Spectrum — Circular Image Area (972 px diameter)")
        self.ax_spectrum.set_xlabel("Energy (eV)" if self.axis_is_energy else "Frame number")
        self.ax_spectrum.set_ylabel("Intensity")
        self.ax_spectrum.legend()
        self.spectrum_canvas.draw()
        self.metadata_label.setText("Displayed mean spectrum from the central circular image area (972 px diameter).")

    def get_circular_mask(self, shape, diameter):
        """Returns a circular mask centered in the image with the given diameter."""
        h, w = shape
        center_y, center_x = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        radius = diameter / 2
        mask = dist_from_center <= radius
        return mask

    def compute_mean_spectrum(self, stack):
        """
        stack: 3D numpy array with shape (frames, height, width)
        Returns mean spectrum from the central circular region (972 px diameter).
        """
        mean_image = np.nanmean(stack, axis=0)
        mask = self.get_circular_mask(mean_image.shape, diameter=972)
        masked_pixels = stack[:, mask]
        mean_spectrum = np.nanmean(masked_pixels, axis=1)
        return mean_spectrum

    def mousePressEvent(self, event):
        if (
            self.crop_mode
            and event.button() == Qt.LeftButton
            and self.image_label.underMouse()
        ):
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()

    def mouseMoveEvent(self, event):
        if self.crop_mode and self.rubber_band.isVisible():
            rect = QRect(self.origin, event.pos()).normalized()
            self.rubber_band.setGeometry(rect)

    def mouseReleaseEvent(self, event):
        if self.crop_mode and self.rubber_band.isVisible():
            self.crop_mode = False
            self.rubber_band.hide()
            rect = self.rubber_band.geometry()
            self.crop_rect = rect
            self.apply_crop()

    def save_processed_stack_hdf5(self):
        """Save the current working stack losslessly for reloading in this GUI.

        Data are stored as float32 so subpixel interpolation and NaN borders from
        drift correction are preserved.  The energy axis and, when available,
        drift-shift metadata are saved alongside the stack.
        """
        if self.data is None:
            QMessageBox.warning(
                self, "No data loaded", "Please load an image stack first."
            )
            return

        stack = np.asarray(self.data, dtype=np.float32)
        if stack.ndim != 3:
            QMessageBox.warning(
                self,
                "Invalid stack",
                f"Expected a 3-D stack (energy, y, x), found {stack.shape}.",
            )
            return

        if self.energy is not None and len(self.energy) == stack.shape[0]:
            energy = np.asarray(self.energy, dtype=np.float64)
        else:
            energy = np.arange(stack.shape[0], dtype=np.float64)

        if getattr(self, "filename", None):
            stem = Path(self.filename).stem
            default_name = f"{stem}_processed.hdf5"
        else:
            default_name = "xpeem_processed_stack.hdf5"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed XPEEM Stack",
            os.path.join(self.last_directory, default_name),
            "HDF5 Files (*.hdf5 *.h5)",
        )
        if not save_path:
            return

        if not save_path.lower().endswith((".hdf5", ".h5")):
            save_path += ".hdf5"

        try:
            ny, nx = stack.shape[1], stack.shape[2]
            sample_x = np.arange(nx, dtype=np.float64)
            sample_y = np.arange(ny, dtype=np.float64)
            nx_time = time.strftime("%Y-%m-%dT%H:%M:%S")

            with h5py.File(save_path, "w") as fout:
                fout.attrs["NeXus_version"] = "4.3.0"
                fout.attrs["creator"] = f"XPEEM Viewer v{__version__}"
                fout.attrs["file_name"] = save_path
                fout.attrs["file_time"] = nx_time
                fout.attrs["processed_stack"] = True

                entry = fout.create_group("entry1")
                entry.attrs["NX_class"] = "NXentry"
                entry.create_dataset("definition", data=np.bytes_("NXstxm"))
                entry.create_dataset("start_time", data=np.bytes_(nx_time))
                entry.create_dataset("end_time", data=np.bytes_(nx_time))
                entry.create_dataset("title", data=np.bytes_("Processed XPEEM stack"))

                xpeem = entry.create_group("xpeem")
                xpeem.attrs["NX_class"] = "NXdata"
                xpeem.attrs["signal"] = "data"
                xpeem.attrs["axes"] = np.array(
                    [b"energy", b"sample_y", b"sample_x"]
                )

                dset = xpeem.create_dataset(
                    "data",
                    data=stack,
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                dset.attrs["long_name"] = "Processed XPEEM image stack"
                dset.attrs["dimension_order"] = "energy,y,x"
                dset.attrs["preserves_nan"] = True

                e = xpeem.create_dataset("energy", data=energy)
                e.attrs["units"] = "eV"
                e.attrs["axis"] = 0

                sy = xpeem.create_dataset("sample_y", data=sample_y)
                sy.attrs["units"] = "pixel"
                sy.attrs["axis"] = 1

                sx = xpeem.create_dataset("sample_x", data=sample_x)
                sx.attrs["units"] = "pixel"
                sx.attrs["axis"] = 2

                xpeem.create_dataset(
                    "count_time",
                    data=np.full(stack.shape[0], np.nan, dtype=np.float32),
                )
                xpeem.create_dataset(
                    "stxm_scan_type", data=np.bytes_("sample image")
                )

                processing = entry.create_group("processing")
                processing.attrs["NX_class"] = "NXprocess"
                processing.create_dataset(
                    "program", data=np.bytes_(f"XPEEM Viewer v{__version__}")
                )
                processing.create_dataset(
                    "note",
                    data=np.bytes_(
                        "Current GUI working stack saved as float32; "
                        "NaNs and subpixel-interpolated values are preserved."
                    ),
                )

                source_name = (
                    os.path.basename(self.filename)
                    if getattr(self, "filename", None)
                    else "unknown"
                )
                processing.create_dataset(
                    "source_file", data=np.bytes_(source_name)
                )

                if self.i0_spectrum is not None:
                    i0_ds = processing.create_dataset(
                        "I0_spectrum", data=np.asarray(self.i0_spectrum, dtype=np.float32)
                    )
                    i0_ds.attrs["description"] = "Mean I0 spectrum from selected background polygon"
                    if self.i0_vertices is not None:
                        processing.create_dataset(
                            "I0_polygon_xy_pixels",
                            data=np.asarray(self.i0_vertices, dtype=np.float32),
                        )
                    if self.i0_normalization_mode is not None:
                        processing.attrs["I0_normalization"] = self.i0_normalization_mode

                if self.drift_shifts is not None:
                    shifts = np.asarray(self.drift_shifts, dtype=np.float64)
                    if shifts.shape == (stack.shape[0], 2):
                        ds = processing.create_dataset(
                            "drift_shifts_yx_pixels", data=shifts
                        )
                        ds.attrs["columns"] = np.array([b"y_shift", b"x_shift"])
                        if self.drift_reference_idx is not None:
                            processing.attrs["drift_reference_slice"] = int(
                                self.drift_reference_idx
                            )
                        if self.drift_algorithm is not None:
                            processing.attrs["drift_algorithm"] = self.drift_algorithm
                        for key, value in self.drift_details.items():
                            processing.attrs[f"drift_{key}"] = value

            self.last_directory = os.path.dirname(save_path)
            self.settings.setValue("last_directory", self.last_directory)
            self.metadata_label.setText(
                f"Processed stack saved:\n{save_path}\n"
                f"Shape: {stack.shape}; dtype: float32\n"
                "This file can be loaded back with Load HDF5 / NXS."
            )
            QMessageBox.information(
                self,
                "Processed stack saved",
                "The current stack was saved as float32 HDF5.\n\n"
                "Subpixel values and NaN borders are preserved, and the file "
                "can be loaded back into this GUI.",
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "HDF5 save error",
                f"Could not save the processed stack:\n{exc}",
            )

    def save_for_axis(self):

        if self.filename:

            nxsFile = h5py.File(self.filename, "r")
            theEntry = list(nxsFile.keys())[0]
            if self.data is None:
                theData = nxsFile[theEntry]["scan_data"]["data_02"][()]
            else:
                theData = self.data

            try:
                energy = nxsFile[theEntry]["scan_data"]["actuator_1_1"][()]
            except:
                energy = np.zeros(len(theData))

            sampleX = np.linspace(0, np.shape(theData)[1] - 1, np.shape(theData)[1])
            sampleY = np.linspace(0, np.shape(theData)[2] - 1, np.shape(theData)[2])

            # hdfOutFileName = nxsFileName.split(".")[0] + "_axis.hdf5"
            savefilename, _ = QFileDialog.getSaveFileName(
                self, "Save axis readable HDF5 File", "", "HDF5 Files (*.hdf5)"
            )
            if savefilename:
                with h5py.File(savefilename, "w") as NXfout:
                    NXfout.attrs["HDF5_Version"] = np.array([b"1.8.4"])
                    NXfout.attrs["NeXus_version"] = np.array([b"4.3.0"])
                    NXfout.attrs["file_name"] = np.array([savefilename.encode("utf-8")])
                    NXtime = time.strftime("%Y-%m-%dT%H:%M:%S+01:00")
                    NXfout.attrs["file_time"] = np.array([NXtime.encode("utf-8")])
                    NXfout.create_group(b"entry1")
                    NXfout["entry1"].create_dataset(
                        "start_time", data=np.array([NXtime.encode("utf-8")])
                    )
                    NXfout["entry1"].create_dataset(
                        "end_time", data=np.array([NXtime.encode("utf-8")])
                    )
                    NXfout["entry1"].attrs["NX_class"] = b"NXentry"
                    NXfout["entry1"].create_dataset(
                        "definition", data=np.array([b"NXstxm"])
                    )
                    NXfout["entry1"]["definition"].attrs["version"] = np.array(b"1.1")
                    NXfout["entry1"].create_group(b"xpeem")
                    NXfout["entry1"]["xpeem"].attrs["NX_class"] = b"NXdata"
                    NXfout["entry1"]["xpeem"].create_dataset(
                        b"data", np.shape(theData), dtype="uint16", data=theData
                    )
                    NXfout["entry1"]["xpeem"].attrs["signal"] = "data"
                    NXfout["entry1"]["xpeem"].attrs["axes"] = [
                        "energy",
                        "sample_x",
                        "sample_y",
                    ]
                    NXfout["entry1"]["xpeem"].attrs["sample_y_indices"] = np.array(
                        [0], dtype="uint32"
                    )
                    NXfout["entry1"]["xpeem"].create_dataset(
                        b"sample_y",
                        (np.shape(theData)[2],),
                        dtype="float",
                        data=sampleY,
                    )
                    NXfout["entry1"]["xpeem"].attrs["sample_x_indices"] = np.array(
                        [1], dtype="uint32"
                    )
                    NXfout["entry1"]["xpeem"].create_dataset(
                        b"sample_x",
                        (np.shape(theData)[1],),
                        dtype="float",
                        data=sampleX,
                    )
                    NXfout["entry1"]["xpeem"].create_dataset(b"energy", data=energy)
                    NXfout["entry1"]["xpeem"].create_dataset(
                        "stxm_scan_type", data=np.array([b"sample image"])
                    )
                    NXfout["entry1"]["xpeem"]["sample_x"].attrs["axis"] = 2
                    NXfout["entry1"]["xpeem"]["sample_y"].attrs["axis"] = 1
                    NXfout["entry1"]["xpeem"]["energy"].attrs["units"] = "eV"
                    NXfout["entry1"]["xpeem"].create_dataset(
                        b"count_time", data=np.linspace(0.1, 0.1, np.shape(theData)[0])
                    )
                    NXfout["entry1"].create_dataset("title", data=[b"XPEEM data"])

    def apply_crop(self):
        if self.crop_rect and self.data is not None:
            xmin, xmax, ymin, ymax = self.crop_rect
            if self.slice_type == "energy":
                self.data = self.data[:, ymin:ymax, xmin:xmax]
            elif self.slice_type == "x":
                self.data = self.data[ymin:ymax, :, xmin:xmax]
            elif self.slice_type == "y":
                self.data = self.data[ymin:ymax, xmin:xmax, :]

            # Update sliders and image
            self.energy_slider.setMaximum(self.data.shape[0] - 1)
            self.x_slider.setMaximum(self.data.shape[1] - 1)
            self.y_slider.setMaximum(self.data.shape[2] - 1)
            self.update_image()
            self.metadata_label.setText("Data cropped to selected ROI.")

    def remove_norm(self):
        from PIL import Image  # Only import when needed

        tiff_path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF Image", self.last_directory, "TIFF Files (*.tif *.tiff)"
        )
        if not tiff_path:
            return

        try:
            tiff_image = Image.open(tiff_path)
            tiff_array = np.array(tiff_image)

            # Ensure the TIFF image matches the spatial dimensions
            if self.data is None:
                self.metadata_label.setText("Load an HDF5 file first.")
                return

            if tiff_array.shape != self.data.shape[1:]:
                self.metadata_label.setText(
                    f"TIFF shape {tiff_array.shape} does not match data shape {self.data.shape[1:]}."
                )
                return

            # Divide the TIFF image from each energy slice
            # Avoid division by zero in the normalization image.
            denom = tiff_array.astype(np.float32)
            denom[denom == 0] = np.nan
            self.data = self.data.astype(np.float32) / denom[np.newaxis, :, :]
            self.last_normalized_image = self.data[self.energy_slider.value()].copy()
            self.save_tiff_button.setEnabled(True)
            self.metadata_label.setText(
                "Normalization applied to all slices. "
                "Use 'Save TIFF Image' to export the displayed normalized slice."
            )
            self.update_image()  # Refresh image view

        except Exception as e:
            self.metadata_label.setText(f"Error loading TIFF: {str(e)}")

    def sum_stack_and_save_tiff(self):
        """
        Sum all images in the currently loaded stack along the first
        (energy/image) axis and save the result as a 32-bit float TIFF.

        NaNs from the circular PEEM mask or drift correction are ignored.
        Pixels that are NaN in every frame remain NaN in the summed image.
        """
        if self.data is None:
            QMessageBox.warning(
                self, "No data", "Please load an HDF5 image stack first."
            )
            return

        stack = np.asarray(self.data, dtype=np.float32)

        if stack.ndim != 3:
            QMessageBox.warning(
                self,
                "Invalid stack",
                f"Expected a 3D stack (images, y, x), found {stack.shape}.",
            )
            return

        valid_count = np.sum(np.isfinite(stack), axis=0)
        summed_image = np.nansum(stack, axis=0, dtype=np.float64).astype(np.float32)
        summed_image[valid_count == 0] = np.nan

        # Display the result before asking where to save it.
        self.ax_image.clear()
        self.ax_image.imshow(summed_image, cmap="gray", origin="lower")
        self.ax_image.set_title(f"Sum of {stack.shape[0]} images")
        self.canvas.draw_idle()

        self.metadata_label.setText(
            f"Summed {stack.shape[0]} images from the current stack. "
            "Choose a filename to save the 32-bit TIFF."
        )

        if hasattr(self, "filename") and self.filename:
            stem = Path(self.filename).stem
            default_name = f"{stem}_sum_{stack.shape[0]}images.tif"
        else:
            default_name = f"xpeem_sum_{stack.shape[0]}images.tif"

        self._save_float_tiff(summed_image, default_name)

    def _save_float_tiff(self, array, default_name):
        """Save a 2D array as a 32-bit floating-point TIFF."""
        if array is None:
            return
        arr = np.asarray(array, dtype=np.float32)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save TIFF Image",
            os.path.join(self.last_directory, default_name),
            "TIFF Files (*.tif *.tiff)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith((".tif", ".tiff")):
            file_path += ".tif"
        Image.fromarray(arr, mode="F").save(file_path)
        self.last_directory = os.path.dirname(file_path)
        self.settings.setValue("last_directory", self.last_directory)
        self.metadata_label.setText(f"32-bit TIFF saved to:\n{file_path}")

    def save_current_tiff(self):
        """
        Export the currently displayed energy image as a 32-bit float TIFF.
        This preserves normalized values rather than rescaling them to 8/16 bit.
        """
        if self.data is None:
            QMessageBox.warning(self, "No data", "Please load a data stack first.")
            return

        if self.slice_type != "energy":
            QMessageBox.information(
                self,
                "Energy image required",
                "TIFF export currently saves an energy image. "
                "Move the energy slider or display an energy slice first.",
            )
            return

        idx = self.energy_slider.value()
        image = self.data[idx, :, :]
        if self.energy is not None and len(self.energy) > idx:
            default_name = f"xpeem_{self.energy[idx]:.2f}eV_normalized.tif"
        else:
            default_name = f"xpeem_slice_{idx:04d}_normalized.tif"
        self._save_float_tiff(image, default_name)

    def _load_2d_tiff(self, title):
        path, _ = QFileDialog.getOpenFileName(
            self, title, self.last_directory, "TIFF Files (*.tif *.tiff)"
        )
        if not path:
            return None, None
        try:
            arr = np.asarray(Image.open(path), dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Expected a 2D TIFF, found shape {arr.shape}.")
            self.last_directory = os.path.dirname(path)
            self.settings.setValue("last_directory", self.last_directory)
            return arr, path
        except Exception as exc:
            QMessageBox.critical(self, "TIFF error", f"Could not load TIFF:\n{exc}")
            return None, None

    def align_cl_cr_and_save_stack(self):
        """
        Load normalized CL and CR TIFF images, register CR to CL using
        translation-only subpixel phase correlation, preview the result,
        and save a two-frame 32-bit TIFF stack.

        Frame 0 = CL reference
        Frame 1 = CR aligned to CL
        """
        cl, cl_path = self._load_2d_tiff("Load normalized CL TIFF image")
        if cl is None:
            return

        cr, cr_path = self._load_2d_tiff("Load normalized CR TIFF image")
        if cr is None:
            return

        if cl.shape != cr.shape:
            QMessageBox.warning(
                self,
                "Shape mismatch",
                f"CL shape {cl.shape} does not match CR shape {cr.shape}.",
            )
            return

        try:
            cl_reg = self._registration_image(cl)
            cr_reg = self._registration_image(cr)

            shift_yx, error, phasediff = phase_cross_correlation(
                cl_reg,
                cr_reg,
                upsample_factor=20,
            )

            cr_aligned = ndi_shift(
                cr.astype(np.float32),
                shift=shift_yx,
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "CL/CR alignment failed",
                f"Could not align the two TIFF images:\n{exc}",
            )
            return

        self.aligned_cl_cr_stack = np.stack(
            [cl.astype(np.float32), cr_aligned.astype(np.float32)],
            axis=0,
        )
        self.cl_cr_shift = np.asarray(shift_yx, dtype=float)

        # Preview the aligned CR image.
        self.ax_image.clear()
        self.ax_image.imshow(cr_aligned, cmap="gray", origin="lower")
        self.ax_image.set_title(
            f"CR aligned to CL: dy={shift_yx[0]:.3f}, dx={shift_yx[1]:.3f} px"
        )

        # Preview residual difference CL - aligned CR.
        self.ax_spectrum.clear()
        difference = cl.astype(np.float32) - cr_aligned
        finite = difference[np.isfinite(difference)]
        if finite.size:
            self.ax_spectrum.hist(finite, bins=100)
        self.ax_spectrum.set_title("CL − aligned CR difference distribution")
        self.ax_spectrum.set_xlabel("Intensity difference")
        self.ax_spectrum.set_ylabel("Pixels")
        self.canvas.draw_idle()

        self.metadata_label.setText(
            "CL/CR alignment calculated.\n"
            f"CL reference: {os.path.basename(cl_path)}\n"
            f"CR moving: {os.path.basename(cr_path)}\n"
            f"Applied CR shift: dy={shift_yx[0]:.3f}, "
            f"dx={shift_yx[1]:.3f} pixels"
        )

        reply = QMessageBox.question(
            self,
            "Save aligned CL/CR stack",
            f"Calculated CR → CL shift:\n"
            f"dy = {shift_yx[0]:.3f} pixels\n"
            f"dx = {shift_yx[1]:.3f} pixels\n\n"
            "Save the aligned two-image TIFF stack?\n\n"
            "Frame 0 = CL reference\n"
            "Frame 1 = aligned CR",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        default_name = "CL_CR_aligned_stack.tif"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save aligned CL/CR TIFF stack",
            os.path.join(self.last_directory, default_name),
            "TIFF Files (*.tif *.tiff)",
        )
        if not file_path:
            return

        if not file_path.lower().endswith((".tif", ".tiff")):
            file_path += ".tif"

        try:
            # Pillow supports multi-page floating-point TIFFs through save_all.
            frame_cl = Image.fromarray(
                self.aligned_cl_cr_stack[0].astype(np.float32), mode="F"
            )
            frame_cr = Image.fromarray(
                self.aligned_cl_cr_stack[1].astype(np.float32), mode="F"
            )

            frame_cl.save(
                file_path,
                save_all=True,
                append_images=[frame_cr],
                compression="tiff_deflate",
                description=(
                    "XPEEM aligned CL/CR stack; "
                    "frame 0=CL reference; frame 1=CR aligned to CL; "
                    f"CR shift dy={shift_yx[0]:.6f}px, dx={shift_yx[1]:.6f}px"
                ),
            )

            self.last_directory = os.path.dirname(file_path)
            self.settings.setValue("last_directory", self.last_directory)
            self.metadata_label.setText(
                f"Aligned CL/CR TIFF stack saved:\n{file_path}\n"
                f"Frame 0 = CL; Frame 1 = aligned CR; "
                f"shift dy={shift_yx[0]:.3f}, dx={shift_yx[1]:.3f} px"
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "TIFF save error",
                f"Could not save the aligned TIFF stack:\n{exc}",
            )

    def run_xmcd_analysis(self):
        """Calculate XMCD from an aligned 2-frame TIFF stack or separate CL/CR TIFFs."""
        box = QMessageBox(self)
        box.setWindowTitle("XMCD Analysis")
        box.setText("Choose the XMCD input:")
        stack_btn = box.addButton("Load aligned CL/CR TIFF stack", QMessageBox.AcceptRole)
        separate_btn = box.addButton("Load separate CL and CR TIFFs", QMessageBox.ActionRole)
        cancel_btn = box.addButton(QMessageBox.Cancel)
        box.exec()

        clicked = box.clickedButton()
        if clicked is None or clicked == cancel_btn:
            return

        if clicked == stack_btn:
            stack_path, _ = QFileDialog.getOpenFileName(
                self, "Load aligned CL/CR TIFF stack",
                self.last_directory, "TIFF Files (*.tif *.tiff)"
            )
            if not stack_path:
                return
            try:
                with Image.open(stack_path) as img:
                    n_frames = getattr(img, "n_frames", 1)
                    if n_frames != 2:
                        QMessageBox.warning(
                            self, "Invalid TIFF stack",
                            f"The TIFF contains {n_frames} frame(s). XMCD expects exactly 2:\n"
                            "Frame 0 = CL\nFrame 1 = aligned CR"
                        )
                        return
                    img.seek(0)
                    cl = np.asarray(img, dtype=np.float32).copy()
                    img.seek(1)
                    cr = np.asarray(img, dtype=np.float32).copy()
                self.last_directory = os.path.dirname(stack_path)
                self.settings.setValue("last_directory", self.last_directory)
                source_text = (f"Aligned stack: {os.path.basename(stack_path)}\n"
                               "Frame 0 = CL; Frame 1 = aligned CR")
            except Exception as exc:
                QMessageBox.critical(self, "TIFF stack error",
                                     f"Could not read the TIFF stack:\n{exc}")
                return
        else:
            cl, cl_path = self._load_2d_tiff("Load normalized CL TIFF image")
            if cl is None:
                return
            cr, cr_path = self._load_2d_tiff("Load normalized CR TIFF image")
            if cr is None:
                return
            source_text = (f"CL: {os.path.basename(cl_path)}\n"
                           f"CR: {os.path.basename(cr_path)}")

        if cl.ndim != 2 or cr.ndim != 2 or cl.shape != cr.shape:
            QMessageBox.warning(
                self, "Invalid XMCD input",
                f"CL and CR must be matching 2D images.\nCL: {cl.shape}\nCR: {cr.shape}"
            )
            return

        cl = np.asarray(cl, dtype=np.float32)
        cr = np.asarray(cr, dtype=np.float32)
        denominator = cl + cr
        valid = (np.isfinite(cl) & np.isfinite(cr) &
                 (np.abs(denominator) > np.finfo(np.float32).eps))

        xmcd = np.full(cl.shape, np.nan, dtype=np.float32)
        xmcd[valid] = (cl[valid] - cr[valid]) / denominator[valid]
        self.xmcd_image = xmcd

        finite = xmcd[np.isfinite(xmcd)]
        if finite.size == 0:
            QMessageBox.warning(self, "XMCD calculation",
                                "No finite XMCD pixels could be calculated.")
            return

        max_abs = float(np.nanmax(np.abs(finite)))
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = 1.0

        if hasattr(self, "_xmcd_colorbar") and self._xmcd_colorbar is not None:
            try:
                self._xmcd_colorbar.remove()
            except Exception:
                pass
            self._xmcd_colorbar = None

        self.ax_image.clear()
        im = self.ax_image.imshow(xmcd, cmap="seismic", origin="lower",
                                  vmin=-max_abs, vmax=max_abs)
        self.ax_image.set_title("XMCD asymmetry: (CL - CR) / (CL + CR)")
        self._xmcd_colorbar = self.canvas.figure.colorbar(
            im, ax=self.ax_image, fraction=0.046, pad=0.04
        )
        self._xmcd_colorbar.set_label("XMCD asymmetry")

        self.ax_spectrum.clear()
        self.ax_spectrum.hist(finite, bins=100)
        self.ax_spectrum.set_title("XMCD asymmetry distribution")
        self.ax_spectrum.set_xlabel("(CL - CR) / (CL + CR)")
        self.ax_spectrum.set_ylabel("Pixels")
        self.canvas.draw_idle()

        self.metadata_label.setText(
            f"XMCD analysis\n{source_text}\nValid pixels: {finite.size}"
        )

        reply = QMessageBox.question(
            self, "Save XMCD TIFF",
            "XMCD analysis is complete.\n\nSave the XMCD asymmetry image as a 32-bit TIFF?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save XMCD TIFF",
            os.path.join(self.last_directory, "XMCD_asymmetry.tif"),
            "TIFF Files (*.tif *.tiff)"
        )
        if not save_path:
            return
        if not save_path.lower().endswith((".tif", ".tiff")):
            save_path += ".tif"

        try:
            Image.fromarray(xmcd.astype(np.float32), mode="F").save(save_path)
            self.last_directory = os.path.dirname(save_path)
            self.settings.setValue("last_directory", self.last_directory)
            self.metadata_label.setText(f"XMCD TIFF saved:\n{save_path}\n{source_text}")
        except Exception as exc:
            QMessageBox.critical(self, "XMCD save error",
                                 f"Could not save the XMCD TIFF:\n{exc}")

    def cluster_cube(self, data_cube, n_clusters=4):
        E, Y, X = data_cube.shape
        flat_data = data_cube.reshape(E, -1).T  # shape: (Y*X, E)

        # Find valid (non-NaN) pixels
        valid_mask = ~np.isnan(flat_data).any(axis=1)
        flat_data_valid = flat_data[valid_mask]

        # Normalize (zero mean, unit variance) to focus on shape
        flat_data_valid = flat_data_valid.astype(np.float32)
        flat_data_norm = (
            flat_data_valid - flat_data_valid.mean(axis=1, keepdims=True)
        ) / (flat_data_valid.std(axis=1, keepdims=True) + 1e-8)

        # UMAP dimensionality reduction
        scaled = StandardScaler().fit_transform(flat_data_norm)
        # embedding = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(scaled)
        embedding = umap.UMAP(n_neighbors=30, min_dist=0.1).fit_transform(scaled)
        # KMeans clustering
        labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(embedding)

        # Create a full label image, initialized with -1
        label_img = np.full((Y * X,), -1, dtype=np.int32)
        label_img[valid_mask] = labels
        label_img = label_img.reshape(Y, X)

        # Average spectra per cluster (using the original, non-normalized spectra)
        mean_spectra = []
        for i in range(n_clusters):
            cluster_indices = labels == i
            if np.any(cluster_indices):
                mean_spectra.append(flat_data_valid[cluster_indices].mean(axis=0))
            else:
                mean_spectra.append(np.zeros(E))
        mean_spectra = np.array(mean_spectra)

        return label_img, mean_spectra

    def save_clustered_spectra(self):
        if (
            hasattr(self, "cluster_results")
            and self.cluster_results is not None
            and np.size(self.cluster_results) > 0
        ):
            spectra_to_save = self.cluster_results
            spectra_type = "Clustered Spectra"
        elif (
            hasattr(self, "pca_components")
            and self.pca_components is not None
            and np.size(self.pca_components) > 0
        ):
            spectra_to_save = self.pca_components
            spectra_type = "PCA Components"
        else:
            self.metadata_label.setText("No spectra (cluster or PCA) to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Save {spectra_type}", self.last_directory, "Text Files (*.txt)"
        )
        if file_path:
            self.last_directory = os.path.dirname(file_path)
            with open(file_path, "w") as f:
                f.write(
                    f"# Energy (eV), {spectra_type.replace(' ', '_')}_1, {spectra_type.replace(' ', '_')}_2, ...\n"
                )
                for i in range(len(self.energy)):
                    line = f"{self.energy[i]:.4f}"
                    for spectrum in spectra_to_save:
                        line += f", {spectrum[i]:.6f}"
                    f.write(line + "\n")
            self.metadata_label.setText(f"{spectra_type} saved to {file_path}")


if __name__ == "__main__":
    # Reuse an existing QApplication if IPython/Spyder already created one.
    # Always run the Qt event loop for this standalone viewer.  This prevents
    # the window/dialogs from appearing only when the IPython session exits.
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    try:
        window = HDF5Viewer()
        window.showMaximized()
        window.raise_()
        window.activateWindow()
        app.exec()
    except Exception as e:
        print(f"Error: {e}")
