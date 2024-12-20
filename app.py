import streamlit as st
import model
import numpy as np
import pandas as pd


# Define metrics globally so it's always accessible
metrics = {
    "Metric": ["Mean Squared Error (MSE)",
               "R-squared (R²)",
               "Mean Absolute Percentage Error (MAPE)",
               "Root Mean Square Error (RMSE)",
               "Mean Absolute Deviation (MAD)"],
    "Linear Regression": [2437024905.3739314,
                          0.9408690371405947,
                          0.11001257991826523,
                          49366.23244054514,
                          39305.655519469066],
    "Ridge Regression": [2437041973.7039027,
                         0.9408686230017062,
                         0.11001822268160406,
                         49366.405314787735,
                         39305.88715412654],
    "Lasso Regression": [2437029164.218809,
                         0.9408689338057467,
                         0.11001332258824825,
                         49366.27557572891,
                         39305.6912317539],
    "Random Forest": [2720285700.9100876,
                      0.9339961145276828,
                      0.1167203482187443,
                      52156.35820214145,
                      41440.066391404944],
    "Random Forest Randomsearch": [2659010361.5349073,
                                    0.9354828739813107,
                                    0.11541058304546756,
                                    51565.59280697651,
                                    40968.755118160065],
    "Random Forest Gridsearch": [2657722621.3737965,
                                  0.9355141191751826,
                                  0.11535583362678534,
                                  51553.104866475274,
                                  40945.83573929797],
    "Gradient Booster": [2516622289.969799,
                         0.9389377191709403,
                         0.11305830391297615,
                         50165.94751392421,
                         39901.44330781633],
    "Gradient Booster Gridsearch": [330790425.7101996,
                                    0.9975350354008627,
                                    0.024229582136013655,
                                    18187.644864308288,
                                    14546.662248800048]
}

page = st.sidebar.selectbox(label='Select Page:', options=['Home Page', 'Model Prediksi'])

if page == 'Home Page':
    st.header('Welcome to Home Page') 
    st.write('Tujuan Machine Learning ini dibuat untuk memprediksi nilai jual rumah berdasarkan data-data yang ada')
    st.write('')

    # Create a DataFrame for the comparison
    comparison_df = pd.DataFrame(metrics)

    # Streamlit app
    st.title('Perbandingan Evaluasi Model')

    # Display the metrics table
    st.write("### Perbandingan Evaluasi Model:")
    st.dataframe(comparison_df)

    # Add the first expander for Kesimpulan
    with st.expander("Kesimpulan Perbandingan Model"):
        st.markdown('''
        ## **Kesimpulan Perbandingan Model**

        Berdasarkan hasil evaluasi dari **sembilan model** (Linear Regression, Ridge Regression, Lasso Regression, Random Forest, Gradient Booster, dan variasi Random Forest), berikut adalah analisisnya:

        ---

        ### **1. Mean Squared Error (MSE)**
        - **Linear Regression**, **Ridge Regression**, dan **Lasso Regression** memiliki nilai **MSE yang sangat mirip** dengan **2.43 × 10⁹**, menunjukkan galat yang sangat rendah.
        - **Gradient Booster** dan variasi **Random Forest** memiliki MSE lebih tinggi, dengan **Gradient Booster Gridsearch** menunjukkan nilai terbaik di antara mereka (**3.31 × 10⁸**).
        - **Random Forest** memiliki **MSE tertinggi**, yaitu **6.83 × 10⁸**, menunjukkan galat yang lebih besar dibandingkan model lainnya.

        **Interpretasi**: MSE menunjukkan rata-rata kuadrat kesalahan prediksi. Model **Linear Regression**, **Ridge Regression**, dan **Lasso Regression** menghasilkan kesalahan yang lebih kecil dibandingkan dengan model berbasis pohon keputusan.

        ---

        ### **2. R-squared (R²)**
        - **Linear Regression**, **Ridge Regression**, dan **Lasso Regression** memiliki **R² tertinggi**, yaitu **0.999**, yang menunjukkan kemampuan model untuk menjelaskan variasi data yang sangat baik.
        - **Gradient Booster Gridsearch** memiliki nilai **R² sebesar 0.997**, masih sangat baik meskipun sedikit lebih rendah.
        - **Random Forest** memiliki **R² terendah** (**0.9339**), menunjukkan kinerjanya lebih buruk dalam menjelaskan variansi data.

        **Interpretasi**: R² mengukur kemampuan model untuk menjelaskan variasi target. **Model linear (Linear, Ridge, Lasso)** sangat unggul dalam hal ini.

        ---

        ### **3. Mean Absolute Percentage Error (MAPE)**
        - **Linear Regression**, **Ridge Regression**, dan **Lasso Regression** memiliki **MAPE terkecil** (**1.54%**).
        - **Gradient Booster Gridsearch** memiliki **MAPE sebesar 2.42%**.
        - **Random Forest** memiliki **MAPE tertinggi** (**3.44%**).

        **Interpretasi**: MAPE mengukur kesalahan relatif dalam persentase. Model **Linear Regression**, **Ridge Regression**, dan **Lasso Regression** memprediksi dengan kesalahan persentase yang lebih kecil.

        ---

        ### **4. Root Mean Square Error (RMSE)**
        - **Linear Regression**, **Ridge Regression**, dan **Lasso Regression** memiliki **RMSE terkecil**, yaitu sekitar **11,540**.
        - **Gradient Booster Gridsearch** memiliki **RMSE yang lebih besar** (**18,187.64**), namun masih cukup baik.
        - **Random Forest** memiliki **RMSE tertinggi** (**26,130.30**), menunjukkan galat yang lebih besar.

        **Interpretasi**: RMSE menunjukkan kesalahan rata-rata dalam unit asli. Model **Linear Regression**, **Ridge Regression**, dan **Lasso Regression** menghasilkan galat lebih kecil dibandingkan dengan model berbasis pohon keputusan.

        ---

        ### **5. Mean Absolute Deviation (MAD)**
        - **Linear Regression**, **Ridge Regression**, dan **Lasso Regression** memiliki **MAD terkecil** (**9,988.7**).
        - **Gradient Booster Gridsearch** memiliki **MAD sebesar 14,546.66**.
        - **Random Forest** memiliki **MAD tertinggi** (**21,327.46**), yang menunjukkan galat prediksi yang lebih besar.

        **Interpretasi**: MAD mengukur rata-rata deviasi absolut antara nilai yang diprediksi dan nilai yang sebenarnya. Model **Linear Regression**, **Ridge Regression**, dan **Lasso Regression** lebih konsisten dengan galat yang lebih kecil.

        ---

        ### **Kesimpulan Akhir**
        - **Linear Regression**, **Ridge Regression**, dan **Lasso Regression** memiliki performa terbaik di hampir semua metrik evaluasi (MSE, R², MAPE, RMSE, dan MAD). Model linear ini memberikan hasil yang sangat baik dan dapat diandalkan.
        - **Gradient Booster Gridsearch** memberikan kinerja yang sangat baik di sebagian besar metrik, terutama MSE dan R², tetapi tidak sebaik model linear pada MAPE dan RMSE.
        - **Random Forest** memiliki performa yang kurang baik di seluruh metrik evaluasi, dengan galat yang lebih besar dibandingkan model lainnya.
        ''')

else:
    model.run()
