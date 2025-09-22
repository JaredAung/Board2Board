# Board2Board

## Welcome to Board2Board
Board2Board uses the capabilities of Computer Vision and Deep Learning to convert a physical chessboard into a Forsyth-Edwards Notation (FEN) with a 94% accuracy. FEN is a line-by-line description of a chessboard (piece+position), which can then be feed into many preexisting services and website to recreate the board into a computer space.

## Goal
A majority of people who play chess prefer to play chess online. According to the Chess.com survery in 2020, about 70% of their user base described themselves as casual players, and among them, about 50% said that they only play online. There could be due to multiple factors. 

1. Players have easy access to built-in evaluation and analysis tools, especially on platforms like Chess.com. There also can get access to coaching, practice AI bots and finding opponenets is very easy. 
2. In order to have the functionalities for OTB chess players, they need to invest in an expensive digital board that would can connected to a computer and can send infornmation about the board to the computer digitally. While this is very sound approach and it is used in broadcasting of professional chess tournments, it is impractial for a casual player as the boards cost upwards of $100.

#### Board2Board seek to bridge this gap and eliminate the need for a expensive digital board by using the capability of AI to convert a physical chessboard into a digital board, which will give the user access to all the powerful chess utility tools available online. 

## Approach 

Technology Stack: OpenCV, Keras, Python, Scikit-Learn, RestNet50

The approach for computer vision pipeline is a reimaginged version of the appraoch taken in the Medium article by ... (name), but some additional features were added to give the pipeline more robustness
and adaptability to different types of images (different lightings, conditions, etc). The pipeline described starts with OpenCV's GaussianBlur (smoothing), Canny (edges detection), and
Houghline (detecting lines). The lines are sorted into vertical and horizontal lines. The intesections of the two types of lines are calculated and clustered using Sci-kit Learn's Agglomerate Clustering, which 
clusters, intersections within a certain thresholds as one cluster or one intersection. 

However during implementing, a lot of problems came to light with the pipeline such as lines outside the board were being detected and throwing up the whole process. 


## Outcome

The computer vision pipeline is able to slice any grid into 9 rows x 9 columns (forming a 8x8 digital board). The thresholds ResNet50 model was able to predict the images 

## How to Use

### Download trained models from the shared google drive. 

 