import streamlit as st
import subprocess
import uuid
from PIL import Image
from concurrent.futures import ThreadPoolExecutor as Pool
from logs import logger


def run_detector(input_folder: str, output_folder: str):
    pool = Pool(max_workers=1)
    f = pool.submit(
        subprocess.call,
        f"""python3 test.py \
            --trained_model="/app/craft_mlt_25k.pth" \
            --test_folder="{input_folder}" \
            --output_folder="{output_folder}" \
            --save_bboxes true \
            --save_result false \
            --cuda false""",
        cwd="/app/CRAFT-pytorch",
        shell=True,
    )
    f.add_done_callback(callback)
    pool.shutdown(wait=True)


def run_predictor(input_folder: str):
    pool = Pool(max_workers=1)
    f = pool.submit(
        subprocess.call,
        f"""python predict.py \
            --sensitive --Transformation TPS --FeatureExtraction ResNet \
            --SequenceModeling BiLSTM --Prediction Attn \
            --saved_model /app/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth \
            --workers 1 \
            --image_folder {input_folder}""",
        cwd="/app/deep-text-recognition-benchmark",
        shell=True,
    )
    f.add_done_callback(callback)
    pool.shutdown(wait=True)


def callback(future):
    if future.exception() is not None:
        logger.error(f"process exception: {future.exception()}")
    else:
        logger.info(f"process returned {future.result()}")


def main():
    st.title("Image text extraction")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        task_id = str(uuid.uuid4())
        input_folder = f"/app/data/{task_id}/input"
        output_folder = f"/app/data/{task_id}/output"
        predictions_file = f"{output_folder}/predictions.json"

        Path(input_folder).mkdir(parents=True, exist_ok=True)
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        image.save(f"{input_folder}/image.jpg", "JPEG")
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Running detector...")
        run_detector(input_folder, output_folder)
        st.write("Running predictor...")
        run_predictor(output_folder)
        if Path(predictions_file).is_file():
            with open(predictions_file) as f:
                st.write(f.read())


if __name__ == "__main__":
    main()
