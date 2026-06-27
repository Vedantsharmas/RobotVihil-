import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("RestoreModel")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "data", "emotion-ferplus-8.onnx")
    model_url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"

    logger.info("Restoring pre-trained high-accuracy ONNX Model Zoo model (85%+ accuracy)...")
    
    if os.path.exists(model_path):
        try:
            os.remove(model_path)
            logger.info("Removed custom CPU model.")
        except Exception as e:
            logger.error(f"Failed to remove existing model: {e}")

    try:
        req = urllib.request.Request(
            model_url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            with open(model_path, 'wb') as f:
                f.write(data)
        logger.info("Pre-trained model downloaded and restored successfully!")
    except Exception as e:
        logger.error(f"Failed to download pre-trained model: {e}")

if __name__ == "__main__":
    main()
