# Sparse Sensor-Based Turbulent Flow Reconstruction using EPOD

A MATLAB implementation of **Extended Proper Orthogonal Decomposition (EPOD)** for reconstructing turbulent flow fields from sparse sensor measurements.

This project was developed during my **Summer Undergraduate Research Internship at the Indian Institute of Technology Hyderabad (IIT Hyderabad)**. It investigates different sensor placement strategies and reconstruction techniques to improve turbulent flow reconstruction accuracy while minimizing the number of required measurements.

---

## Overview

High-resolution flow measurements are expensive and difficult to obtain in real-world engineering applications. This project explores how sparse measurements can be used to reconstruct complete turbulent flow fields using **Extended Proper Orthogonal Decomposition (EPOD)**.

Several sensor selection strategies were implemented and compared to determine the most informative measurement locations for accurate reconstruction.

---

## Objectives

- Reconstruct turbulent flow fields from sparse measurements.
- Compare multiple sensor placement strategies.
- Improve reconstruction accuracy using temporal embedding and filtering.
- Evaluate reconstruction performance using quantitative error metrics.

---

## Methodology

The reconstruction pipeline consists of:

1. Loading turbulent flow data
2. Sensor selection
3. POD/SVD decomposition
4. EPOD reconstruction
5. Temporal embedding
6. Gaussian filtering
7. Error evaluation

---

## Sensor Selection Strategies

- Random Crop
- Coverage-Based Selection
- Sliding Window Search
- DG Leverage Method
- DEIM-Based Sensor Placement

---

## Technologies Used

- MATLAB
- Extended Proper Orthogonal Decomposition (EPOD)
- Proper Orthogonal Decomposition (POD)
- Singular Value Decomposition (SVD)
- Gaussian Filtering

---

## Results

The final reconstruction pipeline achieved approximately **8.5% reconstruction error**, demonstrating significant improvements through optimized sensor placement and temporal filtering.

---

## Repository Structure

```
Sparse-Sensor-Based-Turbulent-Flow-Reconstruction
│
├── src/
│   ├── turbulent_main.m
│   ├── code1coveragemethod.m
│   ├── code4randomcrop.m
│   ├── code5slidingwindowbruteforce.m
│   └── code6deimbest.m
│
├── report/
│   └── FINAL_EPOD_REPORT.pdf
│
├── images/
│
├── results/
│
└── README.md
```

---

## Future Improvements

- Deep learning-based reconstruction models
- Adaptive sensor placement
- Real experimental PIV datasets
- GPU acceleration
- Digital Twin integration

---

## Internship

This work was completed during my **Summer Undergraduate Research Internship** at **Indian Institute of Technology Hyderabad (IIT Hyderabad)**.

---

## Author

**Adhithya S R**

Mechanical Engineering Undergraduate  
National Institute of Technology Karnataka (NITK), Surathkal

LinkedIn: https://www.linkedin.com/in/adhithya-s-r-b2343021a/
