# Meal Nutrition Analysis

This project involves developing a multimodal machine learning model to estimate lunch calorie intake using a dataset comprising diverse modalities such as meal photographs, motion data, demographic attributes, micro gut health parameters, and breakfast-related information.

## Problem Statement

The increasing prevalence of diet-related health issues necessitates scalable and accurate calorie estimation methods. Traditional manual tracking methods are prone to inaccuracies and inefficiencies. Leveraging machine learning and multimodal data integration, this project aims to provide an effective solution for personalized nutrition and dietary monitoring.

## Objectives

- Develop a multimodal model for accurate lunch calorie estimation.
- Integrate heterogeneous data modalities into a unified dataset.
- Achieve predictions surpassing benchmark performance on Kaggle.

## Dataset Overview

The dataset includes:

| Modality                 | Training File            | Testing File               |
|--------------------------|--------------------------|----------------------------|
| Meal Images             | `img_train.csv`         | `img_test.csv`            |
| Demographics & Gut Data | `demo_viome_train.csv`  | `demo_viome_test.csv`     |
| Continuous Glucose Data | `cgm_train.csv`         | `cgm_test.csv`            |
| Labels for Calories     | `label_train.csv`       | `label_test_breakfast_only.csv` |

## Detailed Methodology

### Data Preprocessing

1. **Demographic and Gut Health Data**:
   - Aggregated Viome statistics (mean, max, min) as new features for better representation of gut health.
   - Applied one-hot encoding for categorical variables such as race and gender to make them machine-readable.
   - Standardized continuous numerical features using StandardScaler to normalize their distributions.

2. **Meal Image Data**:
   - Processed raw images by resizing them to 224x224 pixels to fit the input size requirements of EfficientNetB0.
   - Used a pretrained EfficientNetB0 model to extract feature embeddings from the images, capturing critical visual details like portion size and texture.
   - Reduced dimensionality of extracted embeddings using Principal Component Analysis (PCA) to retain only the most significant features and improve computational efficiency.

3. **Continuous Glucose Monitor (CGM) Data**:
   - Cleaned and aligned glucose data timestamps to ensure consistency with meal times.
   - Engineered temporal features such as the interval between breakfast and lunch to capture glucose variability.
   - Computed rolling variance over a 30-minute window to quantify glucose fluctuations as a key metabolic feature.

4. **Data Integration**:
   - Merged features from all modalities (images, demographics, CGM) into a single unified dataset indexed by participant ID and date.
   - Handled missing values through appropriate imputation techniques or removal, depending on modality significance.
   - Applied feature scaling to ensure uniform importance across modalities during model training.

### Model Architecture

The model integrates diverse modalities into a unified neural network. The major components include:

1. **EfficientNetB0**:
   - Extracts meaningful visual features from meal images.

2. **Temporal and Statistical Features**:
   - Utilizes rolling variance from CGM data to capture glucose dynamics.
   - Combines Viome statistics and one-hot encoded demographics to represent participant health profiles.

3. **Fully Connected Neural Network**:
   - Fuses all modality-specific features into a single joint embedding.
   - Contains layers optimized for regression tasks, trained using RMSRE to minimize proportional errors in calorie estimation.

### Model Training

- **Loss Function**: Root Mean Square Relative Error (RMSRE), ensuring proportional error minimization.
- **Optimizer**: Adam optimizer for efficient convergence.
- **Regularization**: Dropout layers and early stopping to prevent overfitting.
- Monitored training using validation loss curves to ensure model generalization.

### Experiments and Hyperparameter Tuning

- Conducted grid search for learning rate, dropout rates, and batch sizes.
- Experimented with different activation functions (ReLU, Leaky ReLU) and optimizers (Adam, SGD).
- Achieved optimal performance with a learning rate of 0.001, batch size of 64, and a dropout rate of 0.3.

### Results

- The model achieved a public RMSRE score of 0.2954, significantly outperforming the Kaggle benchmark of 0.5258.
- Identified key predictors: 
  - **Breakfast Protein**: Most influential feature, indicating its strong correlation with lunch calorie intake.
  - **Rolling Variance**: Highlighted the importance of glucose dynamics in metabolic responses.
  - **Image Embeddings**: Captured crucial visual information about meal composition.

### Complete Workflow

The following outlines the entire approach implemented in this project:

1. **Data Preparation**:
   - Loaded datasets from CSV files and performed modality-specific preprocessing.
   - Ensured all features were aligned by participant ID and date.

2. **Feature Engineering**:
   - Designed temporal and statistical features for CGM data.
   - Extracted meaningful embeddings from meal images.
   - Applied scaling and one-hot encoding for demographics and gut health data.

3. **Model Development**:
   - Implemented EfficientNetB0 for image feature extraction.
   - Built a custom neural network to merge features from all modalities.
   - Trained the model using RMSRE as the loss function.

4. **Evaluation and Results Analysis**:
   - Evaluated model performance using validation RMSRE.
   - Fine-tuned hyperparameters and retrained for optimal performance.

### Video Explanation

For a detailed walkthrough of the project, refer to the video demonstration: [Project Video](https://drive.google.com/file/d/1BuZkIPveY7Qb6t8KGPqV1t7jcfITvBZE/view?usp=sharing)

## Challenges

1. **Heterogeneous Data**:
   - Addressed inconsistencies in data formats by designing separate preprocessing pipelines for each modality.

2. **High Dimensionality**:
   - Applied PCA to reduce meal image embeddings and maintain computational feasibility.

3. **Temporal Data**:
   - Engineered robust temporal features from CGM data to represent glucose behavior effectively.

4. **Small Dataset**:
   - Mitigated overfitting through regularization and data augmentation techniques.

## Contributions

| Team Member       | Contributions                                      |
|--------------------|---------------------------------------------------|
| Priyal Khapra     | Data preprocessing and integration.               |
| Dishant Zaveri    | Model training and optimization.                  |
| Kanishk Chhabra   | Results analysis, visualization, and documentation. |

## Acknowledgments

Special thanks to Prof. Bobak Mortazavi for guidance and Texas A&M University for computational resources. 

## License

This project is licensed under the MIT License. See `LICENSE` for details.
