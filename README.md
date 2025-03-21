#### VR_Project1- Anukriti Singh, Susmita Roy, Mohd. Danish Rabbani
## Face Mask Segmentation

## Overview
This project explores face mask segmentation using both traditional image processing techniques and deep learning-based approaches. The goal is to compare different segmentation methods and evaluate their performance.

## Dataset Structure

The dataset consists of two main folders; however, only **Folder 1** was used for training and evaluation due to the lack of ground truth in **Folder 2**. The dataset was split into **80% training** and **20% testing**.

```
├─ /1
│　├─ face_crop              # Contains cropped face images
│　├─ face_crop_segmentation # Contains segmented face images
│　├─ img                    # Raw images
│　└─ dataset.csv            # CSV file containing dataset metadata
├─ /2 (Not Used)
│　└─ img                    # Unused image directory
```

## Methods Used
### Traditional Methods
1. **Region Growing Segmentation**: Expands regions based on pixel similarity to segment the mask area.
2. **Edge Contour Segmentation**: Detects mask boundaries using edge-based techniques.
3. **Thresholding Segmentation**: Segments the mask region based on intensity thresholds.

### Deep Learning Method
- **UNet**: A fully convolutional neural network designed for pixel-wise segmentation, trained to accurately segment face masks.

## Challenges Faced
- Traditional methods require fine-tuning of parameters for different lighting conditions.
- Region growing and edge detection methods may fail on complex backgrounds.
- UNet requires a large dataset and computational power for training.
- Data preprocessing and augmentation were necessary to improve generalization.

## Results
| Method                  | Accuracy | Intersection over Union     | Dice Coefficient|
|-------------------------|--------- |---------------------------- |-----------------|
| Region Growing          | 0.6730   | 0.2755                      | 0.3652          |
| Edge Contour            | 0.6305   | 0.1563                      | 0.2552          |
| Thresholding Segment.   | 0.5291   | 0.2909                      | 0.4052          |
| UNet                    | 0.6119   | 0.8944                      | 0.9390          |

## Conclusion
- Traditional methods work well for simple cases but struggle with complex backgrounds.
- UNet outperforms traditional methods in terms of segmentation accuracy.
- Future improvements could include using more advanced architectures like DeepLabV3+ or Transformer-based segmentation models.

## Usage
1. Install dependencies:
    ```bash
    pip install torch torchvision numpy opencv-python matplotlib tqdm
    ```
2. Run the segmentation.ipynb notebook

## References
- UNet Paper: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- Image Processing Techniques: [OpenCV Documentation](https://docs.opencv.org/)
