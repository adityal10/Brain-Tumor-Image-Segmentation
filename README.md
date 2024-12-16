# Brain Tumor Image Segmentation Using PyTorch

This project focuses on segmenting brain tumors from MRI images using deep learning and image processing techniques.

## About the Project
Brain tumor segmentation is an important step in analyzing medical images. This project uses:
1. **Clustering techniques** like K-Means, Mean-Shift, and Normalized Cuts.
2. A **deep learning model (SegNet)** built with PyTorch for segmentation tasks.

The dataset used is from Kaggle: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/bilalakgz/brain-tumor-mri-dataset/data).

---

## What’s Inside?
- **Data Preprocessing:** Load and visualize MRI images and labels.
- **Image Segmentation:** Use traditional clustering methods for segmenting images.
- **Deep Learning Model:** Build and evaluate a SegNet model for improved segmentation.
- **Evaluation:** Check the model’s performance using IoU and SSIM metrics.

---

## How to Use It?
1. **Install Dependencies:**
   ```bash
   pip install torch torchvision matplotlib scikit-image scikit-learn pillow
   ```
2. **Run the Code:**
   - Visualize MRI images:
     ```python
     plot_images(num_images=1, val_dir=train_dir, val_imgs_files=train_imgs_files)
     ```
   - Perform segmentation using clustering:
     ```python
     # K-Means
     plt.imshow(segmented_image)
     ```
   - Use SegNet for segmentation:
     ```python
     model = SegNet()
     output = model(input_tensor)
     ```

3. **Dataset Structure:**
   Ensure the dataset follows this format:
   ```plaintext
   brain_tumor_dataset/
     ├── train/
     ├── test/
     ├── valid/
   ```

---

## Example Results
- **K-Means Clustering:**
  - Groups similar regions in the image.
- **SegNet Output:**
  - Binary segmentation mask for tumor regions.

---

## Future Improvements
- Fine-tune the SegNet model for better accuracy.
- Add more data augmentation for better generalization.

---

### Acknowledgements
- **Dataset:** [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/bilalakgz/brain-tumor-mri-dataset/data)
- **Frameworks:** PyTorch, OpenCV, and scikit-learn.

---

This project is for learning purposes. Feel free to use it or improve upon it!
