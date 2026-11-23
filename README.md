# Exp-6--Record-EDGE-DETECTION
# Aim
To perform edge detection using Sobel, Roberts, Prewitt, Laplacian, and Canny edge detectors.

# Software Required
Anaconda – Python 3.7 Jupyter Notebook / VS Code OpenCV (cv2) NumPy Matplotlib

# Algorithm
# Step 1:
Import all the necessary modules for the program.

# Step 2:
Load an image using cv2.imread().

# Step 3:
Convert the image to grayscale.

# Step 4:
Apply Sobel operator using OpenCV to detect edges.

# Step 5:
Apply Prewitt operator using custom kernels.

# Step 6:
Apply Roberts operator using custom kernels.

# Step 7:
Apply Laplacian operator using OpenCV.

# Step 8:
Apply Canny edge detector using OpenCV.

# Step 9:
Display all edge-detected images for comparison.

Developed By Name: Suvetha R Register No: 212225220113

# ORIGINAL IMAGE

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("BMW4.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()
```
# OUTPUT
<img width="222" height="225" alt="image" src="https://github.com/user-attachments/assets/6c7ff794-db4f-4494-bb2b-d9c58cea21a7" />

# Sobel Edge Detector
```
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)  
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)  
sobel_combined = cv2.magnitude(sobel_x, sobel_y)  
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Edge Detection')
plt.axis('off')
```
# OUTPUT
<img width="225" height="217" alt="image" src="https://github.com/user-attachments/assets/54beb219-55e5-4124-8794-bb4bb2c86ddd" />

# Prewitt Edge Detector

```
image = cv2.imread("BMW4.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

prewitt_x = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0, -1]])

prewitt_y = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [-1, -1, -1]])

prewitt_x_edge = cv2.filter2D(gray, cv2.CV_32F, prewitt_x)
prewitt_y_edge = cv2.filter2D(gray, cv2.CV_32F, prewitt_y)

prewitt = cv2.magnitude(prewitt_x_edge, prewitt_y_edge)

plt.imshow(prewitt, cmap='gray')
plt.title('Prewitt Edge Detection')
plt.axis('off')
plt.show()
```
# OUTPUT
<img width="222" height="222" alt="image" src="https://github.com/user-attachments/assets/bce6e3d2-f603-4113-b92c-06ab634cb6c7" />

# Roberts Edge Detector

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(BMW4.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

roberts_x = np.array([[1, 0],
                      [0, -1]], dtype=np.float32)

roberts_y = np.array([[0, 1],
                      [-1, 0]], dtype=np.float32)

gx = cv2.filter2D(gray, cv2.CV_32F, roberts_x)
gy = cv2.filter2D(gray, cv2.CV_32F, roberts_y)

roberts = cv2.magnitude(gx, gy)
roberts = cv2.normalize(roberts, None, 0, 255, cv2.NORM_MINMAX)
roberts = roberts.astype(np.uint8)

plt.imshow(roberts, cmap="gray")
plt.title("Roberts Edge Detection")
plt.axis("off")
plt.show()
```

# OUTPUT
<img width="226" height="222" alt="image" src="https://github.com/user-attachments/assets/f89cda12-1603-4cba-b114-fb28d09fd337" />

# Laplacian Edge Detector

```
import cv2
import matplotlib.pyplot as plt
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
laplacian_8bit = cv2.convertScaleAbs(laplacian)
plt.imshow(laplacian_8bit, cmap='gray')
plt.title('Laplacian Edge Detection')
plt.axis('off')
plt.show()
```

# OUTPUT
<img width="222" height="225" alt="image" src="https://github.com/user-attachments/assets/df6361ec-f15a-4541-a713-2e2ba38af21f" />

# Canny Edge Detector
```
import cv2
import matplotlib.pyplot as plt
canny_edges = cv2.Canny(gray_image, 50, 150)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')
plt.show()
```

# OUTPUT
<img width="220" height="230" alt="image" src="https://github.com/user-attachments/assets/e9c3bd22-1ae2-4c08-90d5-89adef28518f" />

# Result
Thus, edges are successfully detected using Sobel, Prewitt, Roberts, Laplacian, and Canny edge detection techniques. Each method highlights edges differently based on gradient and intensity variations, improving feature extraction and analysis.
