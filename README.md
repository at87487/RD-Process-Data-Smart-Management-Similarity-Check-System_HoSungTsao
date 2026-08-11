# 💾 R&D Experiment Data Smart Management System

An intelligent data management system for R&D laboratories. It features **customizable experiment parameters**, **persistent SQLite storage**, **automatic similarity checks** to prevent duplicate experiments, and **2D spatial visualization** for clear historical data comparison.

---

## 🌟 Key Features

*   **📋 Step 1: Define Custom Experiment Templates**
    *   Dynamically define control parameters (e.g., Temperature, Pressure) and response metrics (e.g., Thickness, Hardness).
    *   The system automatically builds the backend database schema based on your input.
*   **🧪 Step 2: Dynamic Data Worksheet & Persistent Storage**
    *   Interactive spreadsheet interface with double-click editing and one-click SQLite database sync.
*   **🔍 Smart Similarity & Duplicate Check (Euclidean Distance)**
    *   Calculates standardized Euclidean distance between new parameters and historical records to detect direct duplicates (`Distance = 0`) or highly similar experiments.
*   **📈 Experimental Space Visualization**
    *   Plots historical data alongside current target parameters (marked as a red star `★`) in 2D space for instant visual evaluation.
