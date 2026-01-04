# app.py
import streamlit as st
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import joblib
from PIL import Image
import io
import os

st.set_page_config(page_title="Digit Recognizer", layout="wide")

# ---------------- Config ----------------
DIGIT_WIDTH = 10
DIGIT_HEIGHT = 20
IMG_WIDTH = 28
IMG_HEIGHT = 28
CLASS_N = 10

# ---------------- Helpers ----------------
def imresize(img, size):
    """Replace for old imresize: expects grayscale (H,W) or color (H,W,3)."""
    h, w = img.shape[:2]
    new_w, new_h = size
    # cv2.resize uses (width, height)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w // sx) for row in np.vsplit(img, h // sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def pixels_to_hog_20(img_array):
    hog_featuresData = []
    for img in img_array:
        fd = hog(img,
                 orientations=10,
                 pixels_per_cell=(5, 5),
                 cells_per_block=(1, 1),
                 feature_vector=True)
        hog_featuresData.append(fd)
    hog_features = np.array(hog_featuresData, 'float64')
    return np.float32(hog_features)

def get_contour_precedence(contour, cols):
    return contour[1] * cols + contour[0]

def get_digits(contours, hierarchy):
    # expects contours and hierarchy from findContours
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    bounding_rectangles = [cv2.boundingRect(ctr) for ctr in contours]
    final_bounding_rectangles = []
    # find the most common heirarchy level
    u, indices = np.unique(hierarchy[:, -1], return_inverse=True)
    most_common_heirarchy = u[np.argmax(np.bincount(indices))]

    for r, hr in zip(bounding_rectangles, hierarchy):
        x, y, w, h = r
        if ((w * h) > 250) and (10 <= w <= 200) and (10 <= h <= 200) and hr[3] == most_common_heirarchy:
            final_bounding_rectangles.append(r)
    return final_bounding_rectangles

def load_digits_custom(img_file):
    """Extract digits from a custom training image that contains many digits arranged
    in rows of 10 (similar to the original script). Returns (images, labels)."""
    im = cv2.imread(img_file)
    if im is None:
        raise FileNotFoundError(f"Can't read {img_file}")
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    digits_rectangles = get_digits(contours, hierarchy)
    # sort rectangles row-wise left-to-right
    digits_rectangles.sort(key=lambda x: get_contour_precedence(x, im.shape[1]))

    train_data = []
    train_target = []
    start_class = 1
    for index, rect in enumerate(digits_rectangles):
        x, y, w, h = rect
        im_digit = imgray[y:y + h, x:x + w]
        im_digit = (255 - im_digit)
        im_digit = imresize(im_digit, (IMG_WIDTH, IMG_HEIGHT))
        train_data.append(im_digit)
        train_target.append(start_class % 10)
        if index > 0 and (index + 1) % 10 == 0:
            start_class += 1
    return np.array(train_data), np.array(train_target)

def process_user_image_and_predict(img_bytes, model_pipeline):
    """Given image bytes and a trained pipeline, detect digits and predict.
       Returns annotated_image (RGB), blank_digits_image (RGB), list of (digit_img, pred)."""
    file_bytes = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
    im = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("Could not decode image")
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blank_image = np.ones_like(im) * 255

    kernel = np.ones((5, 5), np.uint8)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    digits_rectangles = get_digits(contours, hierarchy)

    predictions = []
    for rect in digits_rectangles:
        x, y, w, h = rect
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        im_digit = imgray[y:y + h, x:x + w]
        im_digit = (255 - im_digit)
        im_digit = imresize(im_digit, (IMG_WIDTH, IMG_HEIGHT))
        hog_feat = pixels_to_hog_20([im_digit])
        pred = model_pipeline.predict(hog_feat)
        label = int(pred[0])
        cv2.putText(im, str(label), (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        cv2.putText(blank_image, str(label), (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
        predictions.append((im_digit, label))
    # convert BGR->RGB for display
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    blank_rgb = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
    return im_rgb, blank_rgb, predictions

# ---------------- Streamlit UI ----------------
st.title("📟 تشخیص ارقام دست‌نویس — Streamlit UI")
st.write("با این رابط می‌تونی دیتاست دست‌نویس سفارشی رو بارگذاری، مدل آموزش بدی یا یک مدل ذخیره‌شده رو بارگزاری و روی تصویر تست اجرا کنی.")

# Sidebar controls
st.sidebar.header("عملیات")
mode = st.sidebar.radio("انتخاب حالت:", ("بارگذاری و آموزش مدل", "بارگذاری مدل آماده و پیش‌بینی"))

# common file uploader for training image
if mode == "بارگذاری و آموزش مدل":
    st.sidebar.subheader("تنظیمات آموزش")
    uploaded_train = st.sidebar.file_uploader("تصویر آموزش (مثلاً custom_train_digits.jpg)", type=['jpg', 'jpeg', 'png'])
    test_size = st.sidebar.slider("نسبت تست (test_size)", 0.1, 0.5, 0.33, step=0.01)
    pca_n = st.sidebar.slider("تعداد مولفه‌های PCA", 10, 100, 40, step=1)
    knn_k = st.sidebar.number_input("k برای KNN", min_value=1, max_value=15, value=5, step=1)
    svm_C = st.sidebar.number_input("C برای SVM", min_value=0.01, value=10.0, step=0.01, format="%.2f")
    svm_gamma = st.sidebar.number_input("gamma برای SVM", min_value=0.0001, value=0.001, step=0.0001, format="%.4f")
    train_button = st.sidebar.button("شروع آموزش")

    if train_button:
        if uploaded_train is None:
            st.sidebar.error("ابتدا تصویر آموزش را بارگذاری کنید.")
        else:
            with st.spinner("در حال استخراج ارقام و آموزش مدل..."):
                # save uploaded file to temp path
                train_bytes = uploaded_train.read()
                tmp_path = "tmp_custom_train.jpg"
                with open(tmp_path, "wb") as f:
                    f.write(train_bytes)
                try:
                    digits, labels = load_digits_custom(tmp_path)
                except Exception as e:
                    st.error(f"خطا در استخراج ارقام از تصویر آموزش: {e}")
                    st.stop()

                st.write("شکل داده‌های استخراج شده:", digits.shape, labels.shape)
                digits, labels = shuffle(digits, labels, random_state=256)
                train_digits_data = pixels_to_hog_20(digits)
                X_train, X_test, y_train, y_test = train_test_split(train_digits_data, labels, test_size=test_size, random_state=42)

                # build pipeline
                knn = KNeighborsClassifier(n_neighbors=int(knn_k))
                svm = SVC(kernel="rbf", C=float(svm_C), gamma=float(svm_gamma), probability=True)
                voting_model = VotingClassifier(estimators=[("knn", knn), ("svm", svm)], voting="soft")
                model_pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=int(pca_n))),
                    ("voter", voting_model)
                ])
                model_pipeline.fit(X_train, y_train)
                preds = model_pipeline.predict(X_test)
                acc = accuracy_score(y_test, preds)
                st.success(f"آموزش تمام شد — دقت روی مجموعهٔ تست: {acc:.4f}")
                # save model
                save_name = st.text_input("نام فایل ذخیره مدل (.pkl)", value="combo_knn_svm_custom_digits.pkl")
                if st.button("ذخیره مدل"):
                    joblib.dump(model_pipeline, save_name)
                    st.success(f"مدل در `{save_name}` ذخیره شد.")
                # show a few extracted digits
                st.write("نمونهٔ چند رقم استخراج‌شده:")
                cols = st.columns(6)
                for i in range(min(6, len(digits))):
                    with cols[i]:
                        d = digits[i]
                        st.image(d, width=64, caption=str(labels[i]))
                # keep model in session state for immediate use
                st.session_state['model_pipeline'] = model_pipeline

elif mode == "بارگذاری مدل آماده و پیش‌بینی":
    st.sidebar.subheader("بارگذاری مدل")
    uploaded_model = st.sidebar.file_uploader("بارگذاری فایل مدل (.pkl)", type=['pkl', 'joblib'])
    load_model_btn = st.sidebar.button("بارگذاری مدل")
    if load_model_btn:
        if uploaded_model is None:
            st.sidebar.error("ابتدا فایل مدل را بارگذاری کنید.")
        else:
            # save to disk and load
            bytes_data = uploaded_model.read()
            tmp_model = "tmp_loaded_model.pkl"
            with open(tmp_model, "wb") as f:
                f.write(bytes_data)
            try:
                model_pipeline = joblib.load(tmp_model)
                st.success("مدل با موفقیت بارگذاری شد.")
                st.session_state['model_pipeline'] = model_pipeline
            except Exception as e:
                st.error(f"خطا در بارگذاری مدل: {e}")

# Prediction panel (common)
st.header("پیش‌بینی روی تصویر کاربری")
uploaded_img = st.file_uploader("تصویر تست برای شناسایی ارقام (img_hand.jpg یا هر عکس دیگری)", type=['jpg', 'jpeg', 'png'])
run_pred = st.button("پیش‌بینی")

if run_pred:
    if 'model_pipeline' not in st.session_state:
        st.error("ابتدا یک مدل آموزش‌دیده یا آماده را بارگذاری کنید (از سایدبار).")
    else:
        if uploaded_img is None:
            st.error("تصویر تست را آپلود کنید.")
        else:
            try:
                model_pipeline = st.session_state['model_pipeline']
                im_rgb, blank_rgb, predictions = process_user_image_and_predict(uploaded_img, model_pipeline)
                st.subheader("تصویر با کادر و برچسب عدد پیش‌بینی‌شده")
                st.image(im_rgb, use_container_width=True)
                st.subheader("تصویر سفید فقط با ارقام پیش‌بینی‌شده")
                st.image(blank_rgb, use_container_width=True)

                if len(predictions) == 0:
                    st.info("هیچ رقم قابل تشخیصی در تصویر پیدا نشد.")
                else:
                    st.write("تعداد ارقام شناسایی‌شده:", len(predictions))
                    # show each extracted digit and prediction
                    cols = st.columns(6)
                    for i, (digit_img, label) in enumerate(predictions[:18]):
                        with cols[i % 6]:
                            st.image(digit_img, width=64, caption=f"pred: {label}")
            except Exception as e:
                st.error(f"خطا در پردازش تصویر: {e}")

# Footer: quick tips
st.markdown("---")
st.markdown("""
**نکات:**  
- برای آموزش دقیق‌تر، از یک تصویر آموزش مشابه `custom_train_digits.jpg` استفاده کن: ردیف‌هایی از 10 رقم پشت سر هم (همانطور که در اسکریپت اصلی انتظار می‌رود).  
- اگر می‌خواهی این را بعداً توسعه بدی: می‌توانیم خروجی‌های HOG یا نرمال‌سازی تصویر را تغییر دهیم.  
- مدل‌ها را می‌توانی در سمت چپ بارگذاری یا ذخیره کنی.
""")


