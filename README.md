# InfraGuard AI: Integrated Infrastructure Monitoring

A vibrant, AI-powered system for structural integrity monitoring.

## Features
- **AI-Powered Diagnostics**: Detects cracks in reinforced concrete and other structures using a deep learning model.
- **Risk Assessment**: Automatically categorizes structural health and provides maintenance recommendations.
- **Vibrant UI**: Built with Streamlit, featuring glassmorphism, 3D GIS mapping, and real-time surveillance simulations.
- **Edge Analytics**: Visualizes structural discontinuities via Canny Edge Detection.

## Project Structure
- `frontend/app.py`: Main Streamlit application.
- `ai/predict.py`: Image prediction logic using the crack detector model.
- `backend/risk_assessment.py`: Business logic for structural risk categorization.
- `utils/image_processing.py`: Image preprocessing and computer vision utilities.
- `model/crack_detector.h5`: Trained TensorFlow model.

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   streamlit run frontend/app.py
   ```

## Integration Details
The system correctly bridges the gap between the frontend UI, the AI inference pipeline, and the backend decision-making logic.
- **Frontend** (Streamlit) -> **AI** (TensorFlow/OpenCV) -> **Backend** (Risk Assessment Logic) -> **UI Results**
