# Human vs. AI Face Detector

An interactive Streamlit game developed for CS6180: Generative AI.

The application compares a human player's ability to distinguish real faces from AI-generated faces against a fine-tuned MobileNetV3Small classifier.

## Features

- 10-round Human vs. Model game
- Balanced rounds with 5 Real and 5 AI-generated images
- Human prediction tracking
- Live MobileNetV3Small inference
- Model confidence score
- Human vs. model scoreboard
- Final game result and replay option

## Model

The classifier is based on MobileNetV3Small with ImageNet-pretrained weights.

The model was first trained with the convolutional backbone frozen and then selectively fine-tuned using a small learning rate.

Binary labels:

- `0` = AI-Generated
- `1` = Real

The final classifier uses a sigmoid output with a threshold of `0.5`.

## Project Structure

```text
real_vs_ai_streamlit/
├── app.py
├── best_mobilenet_finetuned.keras
├── labels.csv
├── requirements.txt
├── README.md
└── test_images/
```

## Run Locally

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the application:

```bash
python3 -m streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

## Dataset

The interactive game uses a balanced subset of the fixed test set from the Real vs. AI-Generated Face Detection assignment.

The deployed subset contains:

- 30 Real images
- 30 AI-generated images

Each game randomly selects 10 images while maintaining a 5/5 class balance.

## Course

CS6180: Generative AI