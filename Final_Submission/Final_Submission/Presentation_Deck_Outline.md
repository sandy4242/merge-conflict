# AISEHack Phase 2 - 5-Slide Presentation Deck Outline

*(Strictly Maps to the 5-Minute 5-Slide Template from the Submission Guidelines)*

---

### Slide 1: Title & Abstract (Overview)
**Title:** Deep-Temporal Ensembles for Advanced Flood Segmentation
**Subtitle:** Aggregating Foundation Models & ConvNets under Hardware Constraints
**Key Talking Points:**
* **Objective**: Automate multi-class geographical segmentation using highly disparate 6-band inputs (Multispectral + SAR) to predict extreme flood events.
* **Our Core Differentiator**: Instead of choosing between the structural perfection of Convolutional Networks OR the broad context of Foundation Models, our pipeline runs a mathematically weighted ensemble of both simultaneously, carefully engineered to squeeze inside a 15GB GPU.
* *Visual suggestion*: Quick map of input images transforming into our 3 distinct classes (No Flood, Flood, Water Body).

---

### Slide 2: Challenges & Architecture
**Heading:** The Dual-Modality Approach
**Bullet Points:**
* **Challenge 1 (Class Imbalance):** Flood representations were heavily overwhelmed (~15%) relative to standard non-flooded terrain (~65%).
* **Challenge 2 (Memory Constraints):** Loading 300 million parameter foundation models breaks Kaggle nodes historically.
* **The Architecture:**
  * Base Foundation: *Prithvi EO v2 (300M-TL)* providing generalized spatial-temporal context logic natively learned from planetary scales.
  * Local Structure Checkers: *UNet++ (EfficientNet-B4)* and *FPN (ResNet50)*.
* *Visual suggestion*: A simple tri-split diagram showing an image going into three discrete models and summing together.

---

### Slide 3: Innovations (The Pipeline Engine)
**Heading:** Engineering Beyond the Baseline
**Key Technical Elements:**
* **Dynamic Loss:** We ditched standard cross-entropy and forged a hybrid combining **DiceLoss** (driving the shape and intersection of floods) + **Weighted CrossEntropy** (scaling penalties by real-time pixel deficit).
* **2-Phase Warmup Training**: 
  1. We completely froze the complex Prithvi backbone to adapt the structural decoder heads without catastrophic forgetting.
  2. End-to-end unfreezing via aggressive Cosine Annealing Learning Rate scheduling.
* **Post-Processing Heuristics**: 4-way spatial Test-Time Augmentation (TTA) with programmatic morphological scrubbing of < 50px spatial artifacts effectively removed false-positive noise natively.

---

### Slide 4: Results & Failed Strategies (Lessons)
**Heading:** What Worked... and What Didn't
**Successes (The Results):**
* Ensemble weighted probability `[0.50 (Prithvi) + 0.35 (UNet) + 0.15 (FPN)]` provided superior stability versus single-model inference, ensuring pixel boundaries matched the geographical realities smoothly.
**Failed Strategies (Lessons Learned):**
* *Heuristic Weighting Failed:* Hand-hacking the raw output probabilities (e.g. padding Flood class uniformly) aggressively increased false positives. Organic neural learning was statistically superior.
* *NDWI Disambiguation Failed:* Attempting to use the established NDWI formula iteratively removed actual silt-heavy flooded zones. Mathematical index masking was incompatible effectively with deep temporal features produced by Prithvi operations.

---

### Slide 5: Conclusion & Future Scope
**Heading:** Conclusion 
**Bullet Points:**
* **Robust Delivery**: Open-source, easily deployable Kaggle framework perfectly synchronized with ANRF submission bounds.
* **Impact**: System successfully automates the discrimination of persistent water bodies from highly variable transient flooding, a notoriously complex remote sensing task.
* **Future Work**: Integration of live Temporal weather data to enforce temporal constraints against dry-spells vs. heavy rain events via the unutilized Prithvi framing system.

*(Prepare for 3 Minutes of Host Q&A)*
