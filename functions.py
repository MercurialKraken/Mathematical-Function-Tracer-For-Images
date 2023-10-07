import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import math

# Load the input image
img = cv2.imread('imag2.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding to the image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours in the image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Initialize an empty list to hold the contour coordinates
contour_coords = []

# Loop through the contours and extract their coordinates
for contour in contours:
    coords = np.squeeze(contour)
    contour_coords.append(coords)

# Initialize an empty list to hold the spline functions
splines = []
#plt.plot(coords[:,0], coords[:,1], color='b')
#plt.show()
# Loop through the contour coordinates and fit a cubic spline to each one
for coords in contour_coords:
    x, y = coords[:, 0], coords[:, 1]
    t = np.arange(len(x))
    interp_spline = CubicSpline(t, np.column_stack((x, y)))
    splines.append(interp_spline)

# Plot the original image
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Plot the spline curves over the image
for spline in splines:
    t_new = np.linspace(0, len(x)-1, 1000)
    xy_new = spline(t_new)
    plt.plot(xy_new[:, 0], -xy_new[:, 1], color='r')
    
for i in xy_new:
    print("({}, {})".format((i[0]-955.811)/250, (-i[1]+558.794)/250))
    
xy_final = [[(i[0]-955.811)/250, (-i[1]+558.794)/250] for i in xy_new]

print(xy_final)
xy_final = np.array(xy_final)


    # Get the coefficients of the cubic polynomial
a, b, c, d = spline.c
e = a[:,1]
f = b[:,1]
g = c[:,1]
h = d[:,1]
a = a[:,0]
b = b[:,0]
c = c[:,0]
d = d[:,0]
    
idk = spline.x

    # Print the piecewise function
    #for a,b,c,d in spline.c[:4]:
    #print('Spline coefficients: a = {:.4f}, b = {:.4f}, c = {:.4f}, d = {:.4f}'.format(float(a), float(b), float(c), float(d)))
for i in range(len(a)):
    xL, xR = spline.x[i], spline.x[i+1]
    ai, bi, ci, di = a[i], b[i], c[i], d[i]
    ei, fi, gi, hi = e[i], f[i], g[i], h[i]
    print('x = {:.4f}(t-{})^3 + {:.4f}(t-{})^2 + {:.4f}(t-{}) + {:.4f}'.format(ai, idk[i], bi, idk[i], ci, idk[i], di, ) + ' \left\{' + '{} < z < {}'.format(xL, xR) + '\\right\}')
    print('y = {:.4f}(t-{})^3 + {:.4f}(t-{})^2 + {:.4f}(t-{}) + {:.4f}'.format(ei, idk[i], fi, idk[i], gi, idk[i], hi, ) + ' \left\{' + '{} < z < {}'.format(xL, xR) + '\\right\}')
plt.show()

