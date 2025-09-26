# Board2Board

**Board2Board** leverages Computer Vision and Deep Learning to convert a physical chessboard into Forsyth-Edwards Notation (FEN) with **94% accuracy**.  

![Python](https://img.shields.io/badge/python-3.10-blue.svg)  
![License](https://img.shields.io/badge/license-MIT-green.svg)  

---

## Demo  
📷 Input (OTB Chessboard Photo) → ♟ Output (Digital FEN + SVG)  

| Input Image | Output (SVG Board) |
|-------------|---------------------|
| ![Sample Input](images/sample_input.jpg) | ![Sample Output](images/sample_output.svg) |

---

FEN is a standardized, line-by-line description of a chessboard (piece + position), which can be fed into many existing platforms and services to recreate the board digitally. This allows over-the-board (OTB) chess positions to be seamlessly transferred into the online environment.

---

## Goal
A majority of chess players today prefer to play online. According to a **2020 Chess.com survey**, about **70%** of their user base identified as casual players, and among them, roughly **50% reported that they only play online**.  

Several factors drive this trend:  

1. **Accessibility of tools** – Online platforms provide built-in evaluation, analysis, coaching, and practice against AI bots. Finding opponents is fast and effortless.  
2. **High cost of OTB digitization** – To enable similar functionalities for OTB play, users need a digital chessboard capable of transmitting board states to a computer. While this approach is used in professional tournament broadcasting, these boards cost upwards of **$100**, making them impractical for casual players.  

**Board2Board aims to bridge this gap** by eliminating the need for expensive hardware. Instead, it uses AI to convert a physical chessboard into a digital one, giving OTB players access to the same powerful tools enjoyed by online players.

---

## Installation  

Clone the repository and install dependencies:  

```bash
git clone https://github.com/JaredAung/Board2Board.git
cd Board2Board
pip install -r requirements.txt

---

## Approach

**Technology Stack:** OpenCV, Keras, Python, Scikit-Learn, ResNet50, Python-Chess, Scikit-Image, Tensorflow, SciPy, Joblib, Jupyter Notebook  

**Computer Vision Pipeline Overview:**  
1. **Preprocessing** – Apply OpenCV’s `GaussianBlur` for smoothing, `Canny` for edge detection, and `HoughLines` to detect lines.  
2. **Line Sorting** – Separate detected lines into vertical and horizontal categories.  
3. **Intersection Detection** – Compute intersections between vertical and horizontal lines.  
4. **Clustering** – Use Scikit-Learn’s Agglomerative Clustering to merge intersections within a given threshold, reducing noise and identifying true grid points.  

**Deep Learning Pipeline Overview:**
1. **Square Extraction** – Crop and warp each of the 64 squares into 224×224 images
2. **ResNet50 Model** – Classify each square as one of 13 classes (12 piece types + empty)
3. **Threshold Regression Models** – Predict optimal pipeline thresholds (Hough + clustering) using ML models trained on logged features
4. **Prediction & Correction** – User validates grid detection thresholds, CNN predicts pieces, and manual corrections can be applied
5. **Final Visualization** – Convert the final board state into Forsyth-Edwards Notation (FEN) and into the SVG of the converted digital board. 
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
   - *Effect:* Four thresholds must be managed (primary HoughLines, primary clustering, secondary HoughLines, secondary clustering).  

3. **Error Correction**  
   - *Problem:* The dataset was custom-built and limited in size, with only about 1,200 training images. While the model achieved **94% accuracy**, it remained prone to occasional misclassifications.  
   - *Fix:* Introduced a user-input correction feature at the end of the classification process, allowing players to manually fix errors before the final FEN is generated.  

---

## How to Use

There are three main scripts in the repository. All use the same OpenCV pipeline, but their purposes differ:  

### `generate_training_data.py`  
- Extracts training images (cropped squares) and generates four CSV files containing image features and threshold values for processing.  
- Thresholds are initially set and managed by the user.  
- After user approval, 64 cropped images are saved and corresponding features/thresholds are appended to their CSVs.  
- Images must then be **manually sorted into labeled folders** (e.g., `black-knight/`, `white-king/`, `empty/`).  
- For best results:  
  - Ensure each square contains the full piece (not cropped out).  
  - Provide at least 100 images per class.  
  - Keep class counts balanced; otherwise, the model may bias predictions toward overrepresented classes.  

### `trainer.ipynb`  
- Run in Google Colab (or another preferred training environment).  
- Trains two sets of models:  
  - **ResNet50** for piece classification using the cropped square images.  
  - **Keras Linear Regression models** for predicting the four pipeline thresholds using the CSV files.  
- On GPU, training is significantly faster.  
- Outputs saved trained models for later use.  

### `classifier.py`  
- Tests the pipeline and generates FEN output.  
1. Download trained models from Google Drive (or adjust file paths if stored locally).  
2. Run `classifier.py` on a test image of a chessboard (provided in the `testing/` folder).  
3. The Linear Regression models predict the four thresholds; user approval is requested at each stage.  
4. After final approval, cropped images are classified by the ResNet50 model.  
5. The system displays piece predictions for each square and allows manual corrections via a simple input prompt.  
6. Once confirmed, the board state is constructed in **FEN notation**, which can be copied directly into chess engines (e.g., [Chess.com Analysis](https://www.chess.com/analysis)).  

---

## Results & Future Work

### Results
- Achieved **94% classification accuracy** on a custom dataset of ~1,200 images across all chess pieces and empty squares.  
- Successfully generated valid FEN notations for OTB chessboard images.  
- Implemented a **hybrid automation + user-correction pipeline**, reducing the impact of misclassifications.  

### Limitations
- Dataset size was relatively small and manually curated, leading to occasional class imbalance.  
- Performance can degrade under poor lighting conditions or with unusual chess piece designs.  
- Manual threshold adjustment is still required (with model predictions as guidance).  

### Future Work
- Expand dataset with **tens of thousands of labeled images** for better generalization.  
- Improve pipeline robustness with **deep-learning–based board detection** (instead of HoughLines).  
- Implement **automatic error correction** using probabilistic validation rules (e.g., both sides must have exactly one king).  
- Optimize for **mobile deployment** so casual players can snap a photo of their board and instantly get a digital reconstruction.  
- Explore integration with platforms like **Lichess or Chess.com** via API for seamless import of positions.  

---
