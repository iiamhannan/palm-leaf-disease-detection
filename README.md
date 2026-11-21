# Palm Leaf Scanner

1. In Colab, export the model to ONNX (see chat) and put `palm_model_v3.onnx` next to `main.py`.
2. pip install -r requirements.txt
3. python -m uvicorn main:app --reload
4. Open http://127.0.0.1:8000  (API docs: /docs)
