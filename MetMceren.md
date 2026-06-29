Materials and Methods
Dataset

We used the MURA (Musculoskeletal Radiographs) dataset released by the Stanford Machine Learning Group, which consists of upper extremity radiographs labeled at the study level as normal or abnormal. The dataset contains 40,561 radiographs grouped into 14,656 studies. In accordance with the official split, 36,808 images from 13,457 studies were used for training and 3,197 images from 1,199 studies were used for validation. Each study corresponds to one patient examination and contains between one and four radiographic projections, depending on the anatomical region (elbow, finger, forearm, hand, humerus, shoulder, or wrist). The task is formulated as a binary classification of study-level abnormality with an imbalanced label distribution of approximately 38.5% abnormal and 61.5% normal cases.

Formally, the dataset is defined as:

D = { (Sᵢ, yᵢ) } for i = 1 … N

where:

Sᵢ = { xᵢ^(v) } for v = 1 … nᵢ represents the i-th study

xᵢ^(v) is the v-th radiographic view of that study

nᵢ ∈ {1, 2, 3, 4} is the number of available views

yᵢ ∈ {0, 1} is the binary abnormality label (0 = normal, 1 = abnormal)

Due to computational constraints (CPU-only training), experiments were conducted on a stratified subset of 1,000 training studies and 200 validation studies, maintaining a balanced class distribution.

Preprocessing and Data Augmentation

All radiographs were originally stored as grayscale images and were converted to three-channel images using a grayscale-to-RGB duplication to match the input format expected by the pretrained convolutional backbone. Pixel intensities were normalized using a fixed mean of (0.5, 0.5, 0.5) and standard deviation of (0.5, 0.5, 0.5) per channel.

During training, on-the-fly data augmentation was applied. The augmentation pipeline included random resized cropping (scale ∈ [0.8, 1.0], aspect ratio ∈ [0.9, 1.1]) to a target size of 224 × 224, random horizontal flips (p = 0.5), small random rotations (up to ±10°), and color jitter (brightness and contrast varied by ±0.2).

Let T(·) denote the composition of all training transformations. For each study Sᵢ, the preprocessed views are defined as:

x̃ᵢ^(v) = T(xᵢ^(v)), for v = 1 … nᵢ

During validation, no data augmentation was applied. Images were resized to 256 × 256 pixels and center-cropped to 224 × 224, followed by normalization only.

To handle the variable number of views per study, a maximum of Vₘₐₓ = 4 views was fixed. Studies with fewer views were zero-padded to obtain tensors of shape [Vₘₐₓ, 3, 224, 224]. A binary mask mᵢ ∈ {0,1}^Vₘₐₓ indicates which positions correspond to real views and which correspond to padding.

Model Architecture

The proposed system is a multi-view convolutional classifier designed to aggregate information from multiple radiographic projections of the same anatomical region. It consists of a shared ResNet18 backbone for per-view feature extraction, followed by a masked average pooling fusion module and a study-level classification head.

Feature Extraction Backbone

Each radiographic view is processed independently by a shared ResNet18 backbone pretrained on ImageNet. The final classification layer of ResNet18 is removed, and the output of the global average pooling layer is used as the feature representation.

For a given view x̃ᵢ^(v), the backbone produces a 512-dimensional embedding:

hᵢ^(v) = fθ(x̃ᵢ^(v)) ∈ ℝ⁵¹²,

where fθ denotes the ResNet18 model with parameters θ.

For each study, up to Vₘₐₓ embeddings are obtained and stacked into a matrix:

Hᵢ ∈ ℝ^(Vₘₐₓ × 512)

Rows corresponding to padded views contain zeros. The same backbone is shared across all views and all studies.

Masked Average Pooling Fusion

Multi-view fusion is achieved using a masked average pooling operation. Given the view embeddings Hᵢ and the binary validity mask mᵢ, the fused study-level representation is computed as:

zᵢ = (1 / Σᵥ mᵢ^(v)) · Σᵥ₌₁^Vₘₐₓ mᵢ^(v) · hᵢ^(v)

This ensures that only valid (non-padded) views contribute to the study representation, and the result is independent of the number of available views.

Classification Head

The fused representation zᵢ is passed to a classification head consisting of a single hidden layer with ReLU activation and dropout regularization, followed by a linear output layer:

ŷᵢ = W₂ · Dropout(ReLU(W₁ · zᵢ + b₁)) + b₂

where W₁ ∈ ℝ^(256 × 512), W₂ ∈ ℝ^(1 × 256) are trainable weight matrices, b₁ and b₂ are bias terms, and dropout is applied with rate 0.3. A sigmoid activation is applied externally (via BCEWithLogitsLoss during training, or explicitly during inference) to obtain the predicted probability of abnormality ŷᵢ ∈ (0, 1).

Training Procedure
Loss Function and Class Imbalance

To address class imbalance, a weighted binary cross-entropy loss with logits (BCEWithLogitsLoss) is used. The positive class weight w₊ is set to 1.597, derived from the ratio of negative to positive samples (0.615 / 0.385).

The loss over a mini-batch of size B is:

L = − (1 / B) · Σᵢ₌₁ᴮ [ w₊ · yᵢ · log(σ(ẑᵢ)) + (1 − yᵢ) · log(1 − σ(ẑᵢ)) ]

where ẑᵢ denotes the raw logit output of the model and σ is the sigmoid function.

Optimization

The model was trained using the Adam optimizer. Seven experimental configurations were compared by varying the learning rate (3 × 10⁻⁴, 1 × 10⁻³, and 3 × 10⁻³), batch size (2, 4, and 8), and training set size (500, 1,000, and 2,000 studies). All configurations were trained for 10 epochs. Model selection was based on the best validation AUC-ROC achieved during training.

Evaluation Metrics

Model performance was evaluated on the validation set using the following metrics:

- AUC-ROC (primary metric): area under the receiver operating characteristic curve, computed by varying the decision threshold over [0, 1].
- Accuracy: (TP + TN) / (TP + TN + FP + FN)
- Sensitivity (recall): TP / (TP + FN)
- Specificity: TN / (TN + FP)
- Precision: TP / (TP + FP)
- F1 score: 2 · (Precision × Sensitivity) / (Precision + Sensitivity)

Binary predictions were obtained at a fixed decision threshold of 0.5. An optimal threshold was also computed using Youden's J statistic (sensitivity + specificity − 1).
