# Partial Coherence Imaging Simulator

This is a personal hobby project for performing partial coherence analysis on optical systems with a defined Numerical Aperture (NA). It simulates the aerial images typically found in semiconductor exposure tools (lithography scanners) using Abbe's method. I built this while trying out Antigravity for the first time.

If you are curious about exposure devices, optics, or computational lithography, please give it a try!

## Features

- **Interactive UI**: A Tkinter-based GUI for real-time parameter configuration, with Matplotlib visualizations.
- **Customizable Optical Parameters**: Adjust Wavelength ($\lambda$), Lens NA, and Illumination coherence ($\sigma$).
- **Mask Definition**: Simulate Line and Space (L&S) patterns with adjustable width, number of lines, and orientation (Vertical, Horizontal, or Both).
- **Aberrations**: Full support for 36 standard Fringe Zernike Coefficients (in waves) for analyzing aberrations like Coma, Astigmatism, and Spherical.
- **Through-Focus Sweep**: Simulate images through various focus steps to evaluate the depth of focus and contrast curves.
- **Visualizations**: View 1D intensity profiles, 2D aerial images, through-focus contrast curves, and through-focus intensity heatmaps.
- **Data Export**: Easily export the resulting 1D profiles, contrast curves, and heatmaps to CSV format.

## Requirements

The application requires Python 3 and the following dependencies:
- `numpy`
- `matplotlib`

You can install the required packages using pip:
```bash
pip install numpy matplotlib
```

## Usage

Run the main application from the terminal:

```bash
python main.py
```

### Optical Parameters
1. **Wavelength $\lambda$ (nm)**: Source illumination wavelength (e.g., 365.0 for i-line, 193.0 for ArF).
2. **Lens NA**: Numerical Aperture of the objective lens.
3. **Illumination $\sigma$**: Partial coherence factor of the illumination source (0 to 1).
4. **Focus & Sweep**: Set the central focus position and perform a sweep over a defined range and step size.
5. **L&S Width (nm)**: Width of the lines corresponding to your mask pattern.
6. **Precision**: Choose between `Fast (Rough)` for quick explorations or `High (Slow)` for detailed, high-resolution rendering.

### Zernike Coefficients
Expand the bottom left section to input values (in waves) for up to 36 Fringe Zernike aberrations to see their impact on the aerial image and contrast.

### Simulation Modes
- **Run Full Simulation**: Computes the through-focus calculations, providing heatmaps and contrast curves.
- **Run 2D & Profile Only**: Computes only the specific focus position (faster, great for tuning).
- **Export to CSV**: Saves the currently displayed simulation data to a CSV file.

## Acknowledgements
Thanks to the Antigravity system for the development experience.
