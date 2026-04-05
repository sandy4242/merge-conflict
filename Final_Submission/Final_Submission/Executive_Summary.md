# Executive Summary: High-Performance Flood Detection Pipeline

**Project Objective:** 
To develop a resilient, highly accurate automated machine learning pipeline capable of segmenting flood events using 6-band multispectral and SAR satellite imagery (SAR-HH, SAR-HV, Green, Red, NIR, SWIR). Designed explicitly for the ANRF AISEHack Theme 1 (Phase 2), the pipeline accurately categorizes geographical terrain across three distinct classes: *No Flood*, *Flood*, and *Water Body*.

## 1. Architectural Overview & Innovations

The core of the solution relies on a robust Multi-Model Ensemble strategy. Recognizing the complexity of multi-sensor geospatial data, the pipeline synthesizes predictions from three diverse, cutting-edge architectures:

### 1.1 Earth Observation Foundation Model: Prithvi EO v2
- **Architecture**: A 300 million parameter temporal-spatial transformer specifically designed by IBM/NASA for Earth Observation, coupled with an UperNet decoder. 
- **Training Strategy**: Deployed via a specialized dual-phase curriculum. 
  - *Phase A (Warm-up)*: Frozen transformer backbone to strictly optimize decoder weights without catastrophically forgetting the foundational representations (5 epochs).
  - *Phase B (Fine-tuning)*: End-to-end unfreezing with a severely reduced learning rate to attune the model to specific regional flood distributions (25 epochs).

### 1.2 Powerful Convolutional Models: UNet++ & FPN
- **UNet++ (EfficientNet-B4)**: Highly structurally aware architecture with dense internal skip pathways, optimal for fine-grained boundary delineation of convoluted floodplains.
- **FPN (Feature Pyramid Network with ResNet-50)**: Excels at multi-scale feature synthesis, effectively capturing both broad spatial context and localized anomalies.
- **Parallel Optimization**: Both models were trained leveraging PyTorch Automatic Mixed Precision (AMP) to maximize the hardware throughput inside constrained memory limits.

## 2. Technical Methodology

### 2.1 Data Pipeline & Augmentation Strategy
The ingestion of 6-channel TIFF imagery poses unique challenges. Our data loader natively handles multi-channel tensors safely and optimally.
- **Aggressive Augmentation**: To prevent model overfitting and promote spatial invariance, the pipeline applies dynamic `Albumentations` transforms during training, including: `ShiftScaleRotate`, `GaussNoise`, `RandomBrightnessContrast`, alongside geometric flips and multi-band normalization.

### 2.2 Addressing Severe Class Imbalance
Flood events naturally constitute a severe minority of geographical image pixels compared to dry land.
- **Dynamic Re-Weighting**: Class distributions were algorithmically measured across the training set (No Flood: ~65%, Flood: ~15%, Water Body: ~19%). Inverse proportional weighting was seamlessly incorporated to force the model to prioritize challenging, critical flood categorizations.
- **Hybrid Loss Formulation**: A compound objective function heavily combining `DiceLoss` (prioritizing structural structural-similarity and IoU metrics) and `Weighted Cross-Entropy Loss` (enforcing strict, pixel-by-pixel categorical accuracy).

### 2.3 Constrained Resource Engineering
Operating within the standard competition T4 GPU hardware limitations necessitated strict computational execution:
- Implemented **Gradient Accumulation** to simulate effectively larger batch sizes on lightweight memory.
- Utilized **Cosine Annealing** dynamic learning rate schedulers to aggressively yet safely map out local minima.
- Auto-resolved underlying platform dependency bottlenecks (notably dynamically bypassing breaking `NumPy` version mismatching intrinsic to the host Kaggle runtime).

## 3. Robust Inference and Post-Processing 

Maximum predictive confidence was realized through advanced processing heuristics applied automatically at test-time.

- **Test-Time Augmentation (TTA)**: The system evaluates every single unseen frame across 4 varying spatial arrangements (Original, Horizontal Flip, Vertical Flip, Diagonal/Both Flip) to forcefully average out single-perspective edge-case biases.
- **Weighted Ensemble Probabilities**: `(0.50 × Prithvi) + (0.35 × UNet++) + (0.15 × FPN)`. The overarching temporal foundation model accounts for the heavy lifting while diverse convolutional models actively resolve localized boundary uncertainties.
- **Morphological Scrubbing**: Employs spatial cleanup to algorithmically erase isolated artifacts and orphaned micro-clusters (under 50 total pixels) from the final flood prediction raster, enforcing contiguous, geographically sound water pooling.

## 4. Final Deliverables & Resources

The comprehensive execution yields single-band, georeferenced TIFF format output predictions that faithfully parallel the precise spatial coordinate definitions of the source datasets. Finally, the automated pipeline programmatically isolates the `Flood` mask layer and compresses it into the compliant column-major Run-Length Encoding (RLE) CSV `submission.csv` format universally required for target evaluation.

### Submission Links
* **Prithvi Checkpoint:** `prithvi-phB-best.ckpt` (Generated at `/kaggle/working/checkpoints/`)
* **UNet++ Checkpoint:** `unetpp_best.pt` (Generated at `/kaggle/working/checkpoints/`)
* **FPN Checkpoint:** `fpn_best.pt` (Generated at `/kaggle/working/checkpoints/`)

---

## Appendix A: Strategies That Did Not Work
During our extensive empirical testing leading up to the final pipeline, several plausible strategies were explored and ultimately discarded due to their negative impacts on the evaluation metrics or hardware stability constraints.

1. **Static Heuristic Probability Multipliers (e.g., `(p * 1.5)`)**: Early attempts locally scaling the raw output probabilities for the `Flood` class independently inside the ensemble induced a severe false positive explosion. This logic mathematically overrode the naturally learned spatial priors, incorrectly predicting normal structural boundaries as flood events. We reverted back strictly to dynamically weighted CrossEntropy at train time to instill organic class distribution logic.
2. **NDWI-Based Post-Processing Disambiguation**: Implementing the Normalized Difference Water Index (NDWI) heuristic mathematically at test-time to rigidly differentiate permanent water bodies from flooded zones actively introduced large-scale false negatives. Real-world flood spectral signatures proved too variable for deterministic multi-spectral indexing at post-process, conflicting heavily with the deep features extrapolated intrinsically by Prithvi constraint mapping.
3. **Implicit Foundation Caching / Silent Downloads**: Forcing Kaggle to inherently download and cache foundation network structures blindly often deadlocked the web-notebook IDE. A specialized, brute-force download loop tracking tokenized progression via the HuggingFace API was implemented to solve this structural barrier, preserving notebook interactivity.
