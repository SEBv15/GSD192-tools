# Calibration

A jupyter notebook with example code can be found [here](https://github.com/SEBv15/GSD192-tools/blob/master/example_notebooks/Calibration.ipynb)

## Method

The energy calibration is done by taking data from sources with known energy peaks like Cd-109 and Co-57 and having the user manually input their locations in the energy spectrum of one strip. Then, using dynamic time warping, their location in all the other strips is determined and refined by fitting a gaussian to every peak. Then a calibration curve is calculated using the locations of the peaks and their energies. Using that, x values in keV (or any other energy unit) can be found for every point in the data.

## Using the library

Import all the relevant classes and functions

```python
from gsd192_tools.calibration import *
```

Load the calibration data from a `.mca` file

```python
strips = Strips(path="calibration-data.mca")
```
The `Strips` class is basically just a list of `Strip` classes with some methods to perform bulk operations on them.

Assuming data where strip `80` has peaks at `450` and `780` with energies `30` and `45` keV, to set the peak locations for the strip and dynamic time warp them to all the other you can either do

```python
strips.set_energies([(450, 30), (780, 45)], 80)
```
or
```python
strips[80].set_energies([(450, 30), (780, 45)])
for strip in strips:
    if not strip.number == 80:
        strip.energies_from(strips[80])
```

To find their more exact locations using curve fitting you can do

```python
strips.refine_energies()
```

Now that all strips have the exact location and energies of their peaks, you can get the x values for every point in the data

```python
x = strips.calibrated_x()
```

or interpolate the data so you don't have to worry about x values

```python
data, xmin, xmax = strips.interpolate()
```

or just look at it

```python
strips.imshow()
```