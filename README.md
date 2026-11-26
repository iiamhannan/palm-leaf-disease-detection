# Palm Leaf Disease Detection

Upload a date palm leaf photo and get a diagnosis (Healthy, Brown Spot or White Scale), a heatmap of where the model looked, and spraying and irrigation advice based on the local weather forecast.

## Features
- Classifies a leaf photo as Healthy, Brown Spot or White Scale, with a confidence score and class probabilities.
- Heatmap (class activation map) that shows which parts of the leaf influenced the result.
- Disease-specific management steps, plus spraying and irrigation timing from the 24-hour forecast (Open-Meteo: rain, wind, temperature, humidity).
- Location is required for a scan: use the device location (Auto) or type latitude and longitude (Manual).
- Light and dark mode.

## Run it
1. `pip install -r requirements.txt`
2. `python -m uvicorn main:app --reload`
3. Open http://127.0.0.1:8000 (API docs: http://127.0.0.1:8000/docs)

The model file `palm_model_v6.onnx` must be next to `main.py`. An internet connection is needed for the weather forecast.

```
main.py              FastAPI backend (prediction, heatmap, weather advice)
static/index.html    Web interface (Bootstrap 5)
palm_model_v6.onnx   Model, runs with ONNX Runtime (no PyTorch needed)
notebooks/           Training notebook and the data split
```

## Model
ResNet18 (ImageNet pretrained), fine-tuned on date palm leaf photos at 224 x 224. The model is exported to ONNX together with class activation maps, so the app runs on ONNX Runtime without PyTorch.

## Data
The datasets are not included. Download them from Kaggle:
- Date Palm Data: https://www.kaggle.com/datasets/hadjerhamaidi/date-palm-data (4,582 images, 2,917 unique image groups after near-duplicates were grouped)
- White Cochineal: https://www.kaggle.com/datasets/bouthinarechachi/1-white-cochiniel-disease (100 White Scale images used)
- Own photos: 31 healthy leaves with natural backgrounds and 35 White Scale leaves

## Results
Near-duplicate images are kept in the same split (group-wise split). On the combined test set of 736 images, 730 were correct (99.2%) and the macro-F1 was 0.991.

| Class | Precision | Recall | F1 | Images |
|---|---|---|---|---|
| Brown Spot | 0.988 | 1.000 | 0.994 | 247 |
| Healthy | 0.995 | 0.979 | 0.987 | 192 |
| White Scale | 0.993 | 0.993 | 0.993 | 297 |

The test set includes small groups: our own White Scale photos (7 of 8 correct), natural-background healthy photos (6 of 8 correct) and White Scale images from the second Kaggle dataset (30 of 30 correct). These groups are small, so treat them as rough indications.

## Training
See `notebooks/palm_leaf_training.ipynb`. `notebooks/split_v3.csv` is the train, validation and test split used for the reported results. The notebook documents the pipeline, and a rerun gives slightly different numbers because training is random (two training runs gave 6 and 10 errors out of 736).

## Limitations
- Trained on date palm leaf photos only. Other plants, trunks and fruit are not supported.
- The three classes were photographed in different setups, and only 31 healthy photos have a natural background.
- Confidence shows how sure the model is, not how severe the infection is.
- The irrigation advice comes from the weather forecast, not from the leaf image.
- The advice is general guidance. Follow the product label and ask a local agriculture officer.