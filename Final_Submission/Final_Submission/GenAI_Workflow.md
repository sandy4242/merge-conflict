# GenAI Tools Workflow & Prompts

*This document complies with the AISEHack Phase 2 Guidelines mandating the disclosure of Generative AI tool usage during the solution development.*

## Overview
During the AISEHack Phase 2 event, the team leveraged **Antigravity (a deep-agentic coding assistant)** as an autonomous copilot. The agent was primarily responsible for debugging environmental dependencies on Kaggle, refactoring PyTorch architectures to fit within T4 GPU limits, and orchestrating the final ensemble inference logic.

## Key Workflows & Chain of Thought (CoT)

### 1. Environmental Bug Resolution (`NumPy` vs. Kaggle constraints)
* **Prompt/Objective:** "Fix our pipeline crash. `segmentation-models-pytorch` is failing due to a NumPy 2.x breaking change when installed via `pip install terratorch`. We need a script that ensures the pipeline runs automatically on Kaggle without manual UI restarts."
* **Agent CoT:** The agent analyzed the standard package resolution that forces `NumPy 2.x`. CoT noted that Kaggle instances preload NumPy `<= 1.26.x` globally, and `pip` conflicting here triggers fatal exceptions in Python modules. The agent proposed a cell script to pre-cache the native environment's version (`import numpy; np_ver = numpy.__version__`), let `pip install terratorch` wipe it, and then enforce a `--force-reinstall --no-deps numpy=={np_ver}` to statically revert to the Kaggle-safe version dynamically.

### 2. Overcoming Foundation Model Silent-Download Freezing
* **Prompt/Objective:** "Cell 2 freezing infinitely in Kaggle when loading Prithvi. Fix this, or make it log progress forcefully."
* **Agent CoT:** The agent evaluated HuggingFace's telemetry behaviors inside non-interactive pseudo-ttys (like Kaggle kernel environments). Recognizing the implicit download was stalling the UI, the agent engineered a brute-force caching script using `huggingface_hub` and `urllib` to forcefully stream and store `Prithvi_EO_V2_300M_TL.pt` chunks, overriding the silent black-box function from `terratorch`.

### 3. Hardware Scaling within Constraints
* **Prompt/Objective:** "How do we fit our 3-model ensemble training (Prithvi, UNet++, FPN) on a single 15GB T4 GPU? We keep getting CUDA OOM."
* **Agent CoT:** The agent decomposed the memory footprint of the models. It structured a workflow implementing a 2-phase frozen/unfrozen curriculum for Prithvi to minimize gradient memory maps early on. It independently separated the convolutional models (`UNet++` and `FPN`) into a secondary sequential training loop forcing PyTorch native `Automatic Mixed Precision (AMP)` via `torch.amp.GradScaler` and rigorous garbage collection (`gc.collect(); torch.cuda.empty_cache()`) between architectures.

### 4. Advanced Post-Processing Formulation
* **Prompt/Objective:** "Write the logic to execute 4-way Test-Time Augmentation across 3 models. Also, add something to clean the prediction mask since we only get scored on the flood class via RLE."
* **Agent CoT:** 
  - Formulated a standard multi-tensor rotation loop for 4 orientations. 
  - Realized through cross-referencing metrics that strict heuristic logic (like Static Probability multipliers) mathematically destroyed performance, and elected instead for Weighted Multiplicative Blending (`0.50` foundation + `0.5` convs).
  - Drafted a fast `scipy.ndimage` morphological scrubber algorithm to eliminate isolated pixel blobs under 50px without using NDWI formulas, correctly predicting that NDWI introduces false negatives under heavy silt conditions.
