## Self-Driving Car Engineer Nanodegree

### Advanced Lane Finding Project

### Overview
The end goal of this project is to track lane lines in a video from a front-facing camera on a car.

### Goals/Steps

The goals or steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_chessboard.png "Undistorted Chessboard"
[image2]: ./output_images/undistorted_test_image.png "Undistorted Road"
[image3]: ./output_images/gradient_binary.png "Gradient Binary Example"
[image4]: ./output_images/magnitude_binary.png "Magnitude Threshold"
[image5]: ./output_images/direction_binary.png "Directional Threshold"
[image6]: ./output_images/s_channel_binary.png "S Channel Threshold"
[image7]: ./output_images/combined_binary.png "Combined Thresholds"
[image8]: ./output_images/warped_image.png "Warped Image"
[image9]: ./output_images/polynomial_fit.png "Polynomial Fit"
[image10]: ./output_images/unwarped_image.png "Unwarped Image"
[video1]: ./project_video_output_full.mp4 "Output Video"

ALL OF THE CODE REFERRED TO BELOW IS IN THE FILE `Main.ipynb`.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Code: Line `11` to `30`

I used multiple chessboard images to calibrate the camera, method for which is implemented in the function `calibrate()`. Corners of the chessboard pattern are found using opencv method `findChessboardCorners()`. The same corner points are then used to draw them up using opencv method `drawChessboardCorners()`. All corners detected are appended to a list called `imgpoints[]` (which is an array of 2D points in image plane). Another array, `objpoints[]`, will have 3D points, `objp`, in real world space denoting the object coordinates of 9x6 chessboard corners. 

These imgpoints and objpoints are then fed into the opencv method `calibrateCamera()` to calculate `cameraMatrix` (a matrix mapping the 2D and 3D points) and `distCoeffs` (distortion coefficients).

Next, the `cameraMatrix` and `distCoeffs` are used in an opencv method `undistort()` to give us an undistorted version of the input image. One such example is as follows.

![alt text][image1]

### Pipeline (images)

#### 1. Provide an example of a distortion-corrected image.

An example of a distortion corrected test image is as follows.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Code: Line `33` to `44`

The undistorted image is then used to get further processed. Thresholded binary images are created to identify lane lines. First, I use gradient thresholding in x direction since the lines we are looking for run vertically through the image. This is achieved by using an opencv method `Sobel`. Implementation is done in `abs_sobel_thresh()`. Have a look at the results here.

![alt text][image3]

Code: Line `47` to `56`

Next, I perform magnitude thresholding on test images which is implemented in `mag_thresh()`. This allows me to see the magnitude of each pixel in the image hence helping in the identification of the lane lines. Have a look at the results here.

![alt text][image4]

Code: Line `59` to `69`

Then comes directional thresholding which is implemented in `dir_thresh()`. This helps me to see which pixels are aligned between 0 and 90deg both left and right. Have a look at the results here.

![alt text][image5]

Code: Line `72` to `78`

Finally, I move on to color thresholding which is one of the most important steps in correctly finding the lines when it comes to varying road shades. I used HLS color space to extract the S channel which is able to identify the lane lines better than other channels. Code for this is implemented in `color_thresh()`. The S channel binary image is displayed below.

![alt text][image6]

With all thresholds combined, the binary image looks like this.

![alt text][image7]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Code: Line `98` to `114`

Perspective transformation is implemented in `persp_transform()`. First, we create a 3x3 transformation matrix `M` using opencv method `getPerspectiveTransform()`. This method takes in four pairs of corresponding points which are called source and destination points. 

```python
    src = np.float32([[200, 720],
                      [593, 450],
                      [693, 450],
                      [1150, 720]])
    
    dst = np.float32([[300, 720],
                      [300, 0],
                      [990, 0],
                      [990, 720]])
```

This resulted in the following source and destination points:

| Source         | Destination   | 
|:--------------:|:-------------:| 
| 200,  720      | 300, 720      | 
| 593,  450      | 300, 0        |
| 693,   450     | 990, 0        |
| 1150, 720      | 990, 720      |

Next, I apply perspective transformation to an image by using the opencv method `warpPerspective()`. It takes in the source image, transformation matrix `M` and image size to output a warped image which has the same size and type as the source image. Check out my warped image example below.

![alt text][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Code: Line `117` to `195`

Implementation of this part of the code is done in `find_lane_pixels()`. It takes in a binary warped image. A histogram is created using bottom half the image. Left and right lane line positions are found out using numpy's `argmax()` method. X positions of the lane lines are calculated by seeing where the highest pixel value lies in the image. These are referred to as `leftx_base` and `rightx_base`.

Next, I create 9 windows to be plotted on the lane lines. This helps to visualize and construct the lane lines. Width of each window is the same. They are constructed as follows.

```
    win_y_low = binary_warped.shape[0] - (window + 1) * window_height
    win_y_high = binary_warped.shape[0] - window * window_height
    
    win_leftx_low = leftx_current - margin
    win_leftx_high = leftx_current + margin
    
    cv2.rectangle(out_img, (win_leftx_low, win_y_low), (win_leftx_high, win_y_high), (0,255,0), 2)
    
```

`win_y_low` and `win_y_high` are the lower and upper bounds of each window in y direction respectively. `win_leftx_low` and `win_leftx_high` are the lower and upper bounds of the window in x direction respectively. These are for the left line. Similarly values are calculated for the right line as well. A rectangle is plotted out using these values using an opencv method `rectangle()`. It takes in two pairs of points (x,y) and an image to plot the result on.

Inside each window, good indices are found by using the code snipet below (left indices shown only here).

```
   good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                    & (nonzerox >= win_leftx_low) & (nonzerox < win_leftx_high)).nonzero()[0]

```
`nonzerox` and `nonzeroy` are all the nonzero pixels at x and y locations respectively. All these indices are appended to a list. If the `good_left_inds` is greater than a pre-set minimum pixel value (50 in this case), then the `leftx_current` is moved about mean position of these good x indices which is implemented as follows.

```
    if len(good_left_inds) > min_pix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
```

This process is repeated for all windows until we have all the left and right lane indices. Finally, the lane-line pixels are identified as follows (left line shown only).

```
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
```

This function `find_lane_pixels()` returns pixel positions and an image: `leftx`, `lefty`, `rightx`, `righty`, `out_img`.

Next comes fitting a polynomial to these pixel positions. That is implemented in `fit_poly()`. Numpy's `polyfit()` method is used to get polynomial coefficients related to left and right lines. After which they are fitted as follows,

```
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
```

where `left_fitx` and `right_fitx` are left and right line polynomial coefficients.

![alt text][image9]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Code: Line `265` to `271`

Curvature radius is implemented in `measure_curvature_real()`. It takes in `left_fit_real` and `right_fit_real`.

`
    left_fit_real = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_real = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
`

where `ym_per_pix = 30/720` is the meter per pixel ratio in y direction and `xm_per_pix = 3.7/700` is the ratio in x direction. Left, right and curvature radiuses are calculated as follows.

`
    left_curve_rad = ((1 + (2*left_fit_real[0]*ymax + left_fit_real[1])**2)**1.5) / np.absolute(2*left_fit_real[0])
    right_curve_rad = ((1 + (2*right_fit_real[0]*ymax + right_fit_real[1])**2)**1.5) / np.absolute(2*right_fit_real[0])
    curve_rad = (left_curve_rad + right_curve_rad) / 2
`

Code: Line `273` to `281`

Vehicle position is implemented in `vehicle_position()`. It takes in `left_fitx`, `right_fitx` and `xm_per_pix`. Left and right line x positions at the base of the image are found out and lane midpoint is calculated. Offset of the car is then deduced by subtracting the lane midpoint from the image center.

`
    left_x_pos = left_fitx[-1]*xm_per_pix
    right_x_pos = right_fitx[-1]*xm_per_pix
    lane_midpoint = (left_x_pos + right_x_pos) / 2
    image_center = (warped.shape[1]//2) * xm_per_pix
    
    offset = image_center - lane_midpoint
`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Code: Line `284` to `333`

The result is implemented and plotted back down onto the road in `draw_final_image()`. Then the text is overlayed onto the final image in `text_overlay()`.

![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Enjoy my video output here!

![alt text][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced issues when the color shade of the road changed. Lane detections were not happening accurately. That was mostly corrected by not detecting lane lines in each and every frame, but using an estimation from the previous frame and search for the lane lines in the same approximate region.

A possible issue will be if the lane was curving too much such that it was going out of camera view window. The lane lines will be going out of a view frame and tracking it will be an issue.