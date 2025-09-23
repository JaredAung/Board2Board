# Board2Board

## Welcome to Board2Board
**Board2Board** leverages Computer Vision and Deep Learning to convert a physical chessboard into Forsyth-Edwards Notation (FEN) with **94% accuracy**.  

FEN is a standardized, line-by-line description of a chessboard (piece + position), which can be fed into many existing platforms and services to recreate the board digitally. This allows over-the-board (OTB) chess positions to be seamlessly transferred into the online environment.

---

## Goal
A majority of chess players today prefer to play online. According to a **2020 Chess.com survey**, about **70%** of their user base identified as casual players, and among them, roughly **50% reported that they only play online**.  

Several factors drive this trend:  

1. **Accessibility of tools** – Online platforms provide built-in evaluation, analysis, coaching, and practice against AI bots. Finding opponents is fast and effortless.  
2. **High cost of OTB digitization** – To enable similar functionalities for OTB play, users need a digital chessboard capable of transmitting board states to a computer. While this approach is used in professional tournament broadcasting, these boards cost upwards of **$100**, making them impractical for casual players.  

**Board2Board aims to bridge this gap** by eliminating the need for expensive hardware. Instead, it uses AI to convert a physical chessboard into a digital one, giving OTB players access to the same powerful tools enjoyed by online players.

---

## Approach

**Technology Stack:** OpenCV, Keras, Python, Scikit-Learn, ResNet50  

The computer vision pipeline is a **reimagined and extended version** of an approach outlined in a [Medium article by <Author’s Name>]. Additional features were added for greater robustness and adaptability across varying lighting conditions and image qualities.  

**Pipeline Overview:**  
1. **Preprocessing** – Apply OpenCV’s `GaussianBlur` for smoothing, `Canny` for edge detection, and `HoughLines` to detect lines.  
2. **Line Sorting** – Separate detected lines into vertical and horizontal categories.  
3. **Intersection Detection** – Compute intersections between vertical and horizontal lines.  
4. **Clustering** – Use Sci-Kit Learn’s Agglomerative Clustering to merge intersections within a given threshold, reducing noise and identifying true grid points.  

---

## Issues & Fixes

During implementation, several issues were encountered and addressed:  

1. **Extraneous Line Detection**  
   - *Problem:* `HoughLines` often detected lines outside the chessboard, disrupting the pipeline.  
   - *Fix:* Images are first manually cropped, and the pipeline was split into **Preprocessing** and **Secondary Processing** stages.  

2. **Two-Stage Processing**  
   - **Preprocessing Stage:**  
     - Detects the four outer corners of the board using the pipeline (`GaussianBlur + Canny + HoughLines + Intersections + Clustering`).  
     - The image is then cropped, perspective-transformed, and warped.  
   - **Secondary Processing Stage:**  
     - Runs the same pipeline again on the warped board to detect internal grid lines.  
     - Final intersections are sorted into an accurate 9×9 grid representing the chessboard squares.  

3. **Error Correction**  
   - *Problem:* The dataset was custom-built and limited in size, with only about 1,200 training images. While the model achieved **94% accuracy**, it remained prone to occasional misclassifications.  
   - *Fix:* Introduced a user-input correction feature at the end of the classification process, allowing players to manually fix errors before the final FEN is generated.  

---

## How to Use
1. Download the trained models from the shared Google Drive.  
2. Run the provided preprocessing and recognition scripts on an image of a chessboard.  
3. The system outputs the board state in **FEN notation**, ready for use on any chess platform.  
