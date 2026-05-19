# Smart Agriculture AI

An end-to-end deep learning system for **agricultural image segmentation**, built with **FastAPI + Streamlit + OpenCV + Deep Learning models**.

##  Project Overview

This project detects and segments agricultural regions (such as crops or affected areas) from images using a trained deep learning model.

It includes:
- AI model for image segmentation
- FastAPI backend for inference
- Streamlit frontend dashboard
-  Real-time visualization with overlay results

---

## Architecture
User Image → Streamlit UI → FastAPI Backend → Deep Learning Model → Mask Output → Visualization
##  Tech Stack

- Python 
- Streamlit 
- FastAPI 
- NumPy
- PyTorch 

---

##  Project Structure
smart_agriculture/
│
├── app.py # Streamlit frontend
├── backend/ # FastAPI backend
├── notebooks/ # Training & experiments
├── utils/ # helper functions
└── README.md

---

##  Features

-  Upload agricultural images
-  AI-powered segmentation
-  Real-time inference via API
-  Overlay visualization (mask on image)
-  Metrics dashboard (pixels, size, inference time)
-  Download results

