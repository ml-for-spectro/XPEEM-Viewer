# XPEEM Viewer — Installation Guide

XPEEM Viewer is a Python/Qt application for loading, visualising, and processing XPEEM HDF5/NXS and TIFF image stacks.

The viewer currently includes:
- HDF5/NXS and TIFF stack loading
- Image and stack visualisation
- ROI-based spectroscopy
- I0 normalisation
- Drift correction
- Phase cross-correlation alignment
- Igor/SPELEEM-style FFT drift correction
- CL/CR image alignment
- XMCD analysis
- Export of processed data

## Recommended installation

The recommended installation uses Miniforge/Conda to create an isolated Python environment. This avoids conflicts with other Python installations and is particularly suitable for beamline and laboratory computers.

## Requirements

- Windows 10/11, macOS, or Linux
- 64-bit Python
- Miniforge/Conda recommended
- Internet connection during installation


## 1. Install Miniforge

If Miniforge or Conda is already installed on your computer, skip to Section 2.

Download Miniforge from:

https://github.com/conda-forge/miniforge

Install the appropriate Miniforge3 version for your operating system.

On Windows, after installation, open **Miniforge Prompt**.

On macOS or Linux, open a terminal.

## 2. Create the XPEEM Viewer environment

Create a dedicated Python environment:

    conda create -n xpeem-viewer python=3.12 -y

Activate the environment:

    conda activate xpeem-viewer

After activation, your terminal prompt should begin with:

    (xpeem-viewer)


## 3. Download XPEEM Viewer

Download the XPEEM Viewer repository from GitHub.

You can either download the repository as a ZIP file using:

**Code → Download ZIP**

or clone the repository using Git:

    git clone <XPEEM-VIEWER-REPOSITORY-URL>

Then enter the XPEEM Viewer directory:

    cd XPEEM-Viewer

## 4. Install the required Python packages

Make sure that the XPEEM Viewer environment is active:

    conda activate xpeem-viewer

Install the required packages:

    pip install -r requirements.txt

This installs the main dependencies required by XPEEM Viewer, including NumPy, h5py, PySide6, Matplotlib, SciPy, scikit-image, and Pillow.

## 5. Start XPEEM Viewer

From inside the XPEEM-Viewer directory, run:

    python xpeem_viewer.py

The XPEEM Viewer graphical interface should open.

## 6. Starting XPEEM Viewer after installation

The Python packages only need to be installed once.

For normal use, open Miniforge Prompt or a terminal and activate the environment:

    conda activate xpeem-viewer

Navigate to the XPEEM-Viewer directory:

    cd <PATH-TO-XPEEM-VIEWER>

Then start the application:

    python xpeem_viewer.py

## 7. Windows launcher

For regular use on a Windows beamline or laboratory computer, a launcher can be used so that XPEEM Viewer can be started without manually entering the commands each time.

The repository includes:

    launch_xpeem_viewer.bat

After the Python environment has been installed correctly, double-click this file to start XPEEM Viewer.

If the launcher does not work, first verify that XPEEM Viewer starts normally from Miniforge Prompt using:

    conda activate xpeem-viewer
    python xpeem_viewer.py

## 8. Verify the installation

After XPEEM Viewer opens:

1. Load an HDF5/NXS file or a TIFF image stack.
2. Confirm that the image and frame/energy navigation controls are displayed.
3. Open the drift-correction interface.
4. Confirm that the two drift-correction approaches are available:
   - Phase cross-correlation
   - Igor/SPELEEM-style FFT correlation
5. Open Processing Help to access the built-in workflow documentation.


## 9. Updating XPEEM Viewer

If XPEEM Viewer was installed by cloning the GitHub repository, navigate to the XPEEM-Viewer directory and run:

    git pull

Then activate the environment:

    conda activate xpeem-viewer

If the dependencies have changed, update them with:

    pip install -r requirements.txt

## 10. Troubleshooting

### ModuleNotFoundError

Make sure the correct environment is active:

    conda activate xpeem-viewer

Then reinstall the required packages:

    pip install -r requirements.txt

### PySide6 or Qt error

First verify that XPEEM Viewer is being launched from the dedicated environment:

    conda activate xpeem-viewer
    python xpeem_viewer.py

If necessary, reinstall PySide6:

    pip uninstall PySide6 PySide6_Addons PySide6_Essentials shiboken6 -y
    pip install PySide6

### The application starts from the terminal but not from the Windows launcher

First confirm that the application works from Miniforge Prompt.

The Windows launcher must be able to activate the `xpeem-viewer` Conda environment before starting the application.

### Installation behind an institutional proxy

If your computer accesses the internet through an institutional proxy, use the proxy configuration provided by your institution or IT service.

## 11. Uninstalling

To remove the dedicated XPEEM Viewer Python environment:

    conda deactivate
    conda env remove -n xpeem-viewer

The downloaded XPEEM-Viewer directory can then be deleted separately.
