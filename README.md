# A Linear SVM Classifier Trained on Intel Image Classification Dataset

This project demonstrates the training of a Linear SVM Classifier on the Intel Image Classification Dataset. The classifier is trained on extracted HOG (Histogram of Oriented Gradients) features and achieves satisfactory results on training, validation, and test datasets.

## Table of Contents
- [Setup](#setup)
- [Dependencies](#dependencies)
- [Dataset Structure](#dataset-structure)
- [Feature Extraction](#feature-extraction)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)

---

## Setup

Ensure you have the required libraries installed and the dataset downloaded:

```bash
!pip install torch
!pip install kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

The dataset should be placed in the following directory structure:

```
intel_image_classification/
    seg_train/
        seg_train/
            class_1/
            class_2/
            ...
    seg_test/
        seg_test/
            class_1/
            class_2/
            ...
```

---

## Dependencies

Below are the primary libraries used in this project:

- `torch`
- `torchvision`
- `scikit-learn`
- `numpy`
- `opencv-python`
- `matplotlib`
- `seaborn`

---

## Dataset Structure

The dataset is organized into training and testing sets, each containing images categorized into the following six classes:

1. Buildings
2. Forest
3. Glacier
4. Mountain
5. Sea
6. Street

---

## Feature Extraction

HOG features are extracted for each image to represent its structural and textural information. The following parameters are used:

- `Image Dimensions`: 64x64
- `Orientations`: 9
- `Pixels per Cell`: (12, 12)
- `Cells per Block`: (3, 3)

The HOG feature extraction function is implemented as:

```python
def get_hog_data(images_path, width, height, orient=9, pixelsxcell=(12, 12), cellsxblock=(3, 3)):
    ...
```

After extraction, the training set contains 14,034 features, and the test set contains 3,000 features.

---

## Training and Evaluation

### Data Preparation

The training dataset is split into training and validation sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train_numeric, test_size=0.2, random_state=42)
```

### Model Training

The SVM classifier is trained with the following function:

```python
def SVM(Xtrain, Ytrain, Xval, Yval):
    clf = SVC(random_state=0)
    clf.fit(Xtrain, Ytrain)
    return clf
```

Training achieves:
- **Training Accuracy**: 85.04%
- **Validation Accuracy**: 72.96%

![Train and Val Confusion Matrix](/Images/train_val_result.png)

---

## Test Results

The trained SVM model achieves the following on the test set:

- **Test Accuracy**: 70.6%

![Test Confusion Matrix](/Images/test_result.png)

---

## Conclusion

This project demonstrates how to train a Linear SVM Classifier using HOG features for image classification. While the results are promising, further optimization could improve the classifier's performance. Feel free to explore and enhance this project!

---

## License

This project is licensed under the MIT License.

--- 

**Contributions**: 

[Huzaifah Tariq Ahmed](https://github.com/huzaifahtariqahmed). 





