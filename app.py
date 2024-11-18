import os
import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# Thiết lập môi trường PySpark
os.environ['PYSPARK_PYTHON'] = '/Users/hoangnguyenbaotram/opt/anaconda3/bin/python3.9'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/hoangnguyenbaotram/opt/anaconda3/bin/python3.9'

# Tạo SparkSession
spark = SparkSession.builder.appName("Credit Card Fraud Detection").getOrCreate()

# Đường dẫn tới mô hình đã huấn luyện
model_path = "/Users/hoangnguyenbaotram/gianlan/gbt_model"

# Tải mô hình
try:
    gbt_model = GBTClassificationModel.load(model_path)
    st.success("Model loaded successfully.")
    st.write(f"Model expects {gbt_model.numFeatures} features.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Streamlit App
st.title("Credit Card Fraud Detection")

# Upload file CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", data.head())

    # Điền giá trị mặc định cho các cột bị thiếu
    required_columns = ['merchant', 'category', 'amt', 'state', 'job', 'is_fraud']
    for col in required_columns:
        if col not in data.columns:
            data[col] = 'unknown' if col in ['merchant', 'category', 'state', 'job'] else 0

    # Chuyển đổi Pandas DataFrame thành Spark DataFrame
    spark_df = spark.createDataFrame(data)

    # Định nghĩa pipeline
    categorical_features = ['merchant', 'category', 'state', 'job']
    numeric_features = ['amt']
    indexers = [
        StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep")
        for col in categorical_features
    ]
    encoders = [
        OneHotEncoder(inputCol=col + "_index", outputCol=col + "_ohe")
        for col in categorical_features
    ]
    assembler = VectorAssembler(
        inputCols=[col + "_ohe" for col in categorical_features] + numeric_features,
        outputCol="features_unscaled"
    )
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])

    # Áp dụng pipeline
    try:
        processed_data = pipeline.fit(spark_df).transform(spark_df)
    except Exception as e:
        st.error(f"Failed to process the data pipeline: {e}")
        st.stop()

    # UDF để chuyển đổi Vector thành ARRAY
    def vector_to_array(v):
        return v.toArray().tolist() if v is not None else []

    vector_to_array_udf = udf(vector_to_array, ArrayType(DoubleType()))

    # Chuyển đổi cột "features" thành mảng
    processed_data = processed_data.withColumn("features_array", vector_to_array_udf(F.col("features")))

    # Kiểm tra số lượng phần tử trong mảng
    num_features = processed_data.select(F.size(F.col("features_array")).alias("num_features")).head()["num_features"]

    # Thêm padding nếu cần
    if num_features < gbt_model.numFeatures:
        padding_size = gbt_model.numFeatures - num_features
        padding_array = F.array(*[F.lit(0.0) for _ in range(padding_size)])
        processed_data = processed_data.withColumn(
            "features_array_padded",
            F.concat(F.col("features_array"), padding_array)
        )

        # Chuyển ARRAY thành Vector
        def array_to_vector(array):
            return DenseVector(array)

        array_to_vector_udf = udf(array_to_vector, VectorUDT())
        processed_data = processed_data.withColumn("features", array_to_vector_udf(F.col("features_array_padded")))
        processed_data = processed_data.drop("features_array", "features_array_padded")

    # Dự đoán
try:
    predictions = gbt_model.transform(processed_data)
    result = predictions.select("merchant", "category", "amt", "state", "job", "prediction").toPandas()

    # Hiển thị kết quả dự đoán
    st.write("Prediction Results:", result)

    # Tổng hợp số lượng giao dịch gian lận (fraud) và không gian lận (non-fraud)
    fraud_count = result['prediction'].value_counts().get(1, 0)  # Giá trị 1 là gian lận
    non_fraud_count = result['prediction'].value_counts().get(0, 0)  # Giá trị 0 là không gian lận

    # Hiển thị tổng hợp
    st.write(f"Number of Fraudulent Transactions Detected: {fraud_count}")
    st.write(f"Number of Non-Fraudulent Transactions Detected: {non_fraud_count}")

except Exception as e:
    st.error(f"Prediction failed: {e}")


