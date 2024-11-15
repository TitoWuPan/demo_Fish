import joblib
import streamlit as st
import numpy as np

# Cargar todos los modelos desde el archivo único
modelos = joblib.load('modelos_clasificacion.pkl')
lin_reg = modelos["regresion_lineal"]
svc = modelos["svc"]
log_reg = modelos["regresion_logistica"]


clase_map = {
    0: 'Anabas testudineus',
    1: 'Coilia dussumieri',
    3: 'Otolithoides biauritus',
    4: 'Otolithoides pama',
    5: 'Pethia conchonius',
    6: 'Polynemus paradiseus',
    7: 'Puntius lateristriga',
    8: 'Setipinna taty',
    9: 'Sillaginopsis panijus',
}

def main():
    st.title('Aplicación de Clasificación de Especies')
    st.sidebar.header('Ingrese los valores para clasificar la especie')
    def user_input_parameters():
        length = st.sidebar.slider('Length', 6.36, 7.9, 33.86)
        weight = st.sidebar.slider('Weight', 2.05, 4.4, 6.29)
        w_l_ratio = st.sidebar.slider('w_l_ratio', 0.08, 0.32, 0.64)  
        features = np.array([[length, weight, w_l_ratio]])
        return features

    df = user_input_parameters()
    option = ['Linear Regression', 'Logistic Regression', 'SVM']
    model = st.sidebar.selectbox('Which model you like to use?', option)
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        
        pred_lin_reg = clase_map[abs(lin_reg.predict(df).round().astype(int)[0])]
        pred_svc = clase_map[svc.predict(df)[0]]
        pred_log_reg = clase_map[log_reg.predict(df)[0]]
    
        if model == 'Linear Regression':
            st.write(f"**Regresión Lineal Predice:** {pred_lin_reg}")
        elif model == 'Logistic Regression':
            st.write(f"**SVC Predice:** {pred_svc}")
        else:
           st.write(f"**Regresión Logística Predice:** {pred_log_reg}")


if __name__ == '__main__':
    main()
