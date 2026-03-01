import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from config import ConfigDataset, ConfigPlot
from dataset.math.plot import DatasetPlot
from dataset.gen.dataframe import DataframeGenerator

class DatasetGenerator:
    """
    Generates TensorFlow data generators for image datasets.
    
    Implements the dataset preprocessing pipeline from Figure 6:
    1. Load images and labels from paths
    2. Apply sample range filtering (optional subsets)
    3. Train/validation/test splitting (stratified)
    4. Standardize image size and apply preprocessing
    5. Create TensorFlow data generators for batch feeding to models
    
    Related to section 2.1.2 (Dataset Preprocessing and Figure 6)
    """
    
    def generate(self, paths, labels, indicies, dataset_config, plot_config, image_processing, known_classes=None):
        """
        Generates train, validation and test data iterators for image dataset.
        
        Implements the data pipeline:
        - Creates a DataFrame with image paths and labels
        - Splits into train (50%), validation (25%), test (25%)
        - Resizes images to input_shape
        - Converts to grayscale (medical imaging convention)
        - Applies preprocessing (contrast enhancement, sharpening, clipping)
        - Returns TensorFlow ImageDataGenerators for batch training
    
        Args:
            paths (str): Paths to all directories, which includes images.
            labels (str): List of labels to corresponding path to directory.
            indicies (list): List of sample ranges [start, end] to extract subsets. Empty list = all samples.
            dataset_config (ConfigDataset): Image size, batch size, split ratio
            plot_config (ConfigPlot): Plotting configuration for sample visualization
            image_processing (func): Optional custom preprocessing function

        Returns:
            tuple: (train_data, validation_data, test_data) as ImageDataGenerators
        """
        
        df = DataframeGenerator.generate(paths, labels, indicies)
        
        if(dataset_config.debug):
            print("\nDataframe:")
            print(df.head())
            print("")
            print(df.tail())
            print("")
            
        train_df, test_df = train_test_split(df, 
                                             test_size=dataset_config.split, 
                                             random_state=42,
                                             stratify=df.labels if df.labels.nunique() > 1 else None)
        train_df, val_df = train_test_split(train_df, 
                                            test_size=(dataset_config.split-0.1), 
                                            random_state=42,
                                            stratify=train_df.labels if train_df.labels.nunique() > 1 else None)
        
        if(dataset_config.debug):
            print("\nDataframe shapes (split): \n")
            print("Training dataset   : ", train_df.shape)
            print("Test dataset       : ",test_df.shape)
            print("Validation dataset : ", val_df.shape)
            print("")
        
        # Fix pre-proccessing for single dataset. 
        if(image_processing != None):
            image_gen = ImageDataGenerator(
                rescale=1./255,
                preprocessing_function=lambda image: self.default_processing(image)
        )
        else:      
            image_gen = ImageDataGenerator(
                rescale=1./255,
                preprocessing_function=lambda image: self.default_processing(image)
        )
        
       # Ensure all splits use the same class list (fixes shape mismatch when a split has only 1 class).
        all_classes = sorted(df["labels"].unique().tolist())
        if known_classes:
            for cls in known_classes:
                if cls not in all_classes:
                    all_classes.append(cls)
            all_classes = sorted(all_classes)

       # CHANGED COLOR MODE TO GRAYSCALE. 
        train_data = image_gen.flow_from_dataframe(dataframe=train_df,x_col="filepaths",y_col="labels",
                                            target_size=(dataset_config.image_size,dataset_config.image_size),
                                            color_mode='grayscale',
                                            class_mode="categorical", 
                                            classes=all_classes,
                                            batch_size=dataset_config.batch_size,
                                            shuffle=True)
        
        validation_data = image_gen.flow_from_dataframe(dataframe=val_df,x_col="filepaths", y_col="labels",
                                            target_size=(dataset_config.image_size,dataset_config.image_size),
                                            color_mode= 'grayscale',
                                            class_mode="categorical",
                                            classes=all_classes,
                                            batch_size=dataset_config.batch_size,
                                            shuffle=False)
        
        test_data = image_gen.flow_from_dataframe(dataframe=test_df,x_col="filepaths", y_col="labels",
                                            target_size=(dataset_config.image_size,dataset_config.image_size),
                                            color_mode='grayscale',
                                            class_mode="categorical",
                                            classes=all_classes,
                                            batch_size=dataset_config.batch_size,
                                            shuffle=False)
        
        if(plot_config.plot and dataset_config.debug):
            plot = DatasetPlot()
            plot.plot_classes(train_data, train_df, plot_config)
        
        return train_data, validation_data, test_data

    def default_processing(self, image):
        """
        Default preprocessing for medical images.
        
        Steps:
        1. Convert RGB to grayscale if needed
        2. Contrast enhancement (brighten + shadow)
        3. Sharpening filter kernel
        4. Clip values to valid range [0, 255]
        5. Restore channel dimension to (H, W, 1)
        
        Improves model's ability to extract relevant features from medical images
        by enhancing contrast and sharpness of anatomical structures.

        Args:
            image: Input image (H, W) or (H, W, C)

        Returns:
            image: Preprocessed image with shape (H, W, 1)
        """
        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = image[:, :, 0]

        image = cv2.addWeighted(image, 1.5, image, -0.5, 0)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)

        image = np.clip(image * 1.25, 0, 255)

        # Restore channel dimension so shape is (H, W, 1)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        return image