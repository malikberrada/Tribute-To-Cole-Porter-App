import streamlit as st
import pandas as pd
import io
import base64
import librosa
import librosa.display
import numpy as np
import pickle
from PIL import Image
from streamlit_option_menu import option_menu
import tensorflow as tf

# configuration

im = Image.open("../pics/Jazz icon.jpg")
st.set_page_config(
                    page_title="Tribute to Cole Porter",
                    page_icon=im,
                    layout="wide",
                    initial_sidebar_state="expanded",
                  )

# sidebar
im_about = Image.open("../pics/image_about.png")
im_prediction = Image.open("../pics/image_prediction_2.png")
im_singer = Image.open("../pics/image_singer_2.png")
im_song = Image.open("../pics/image_song_2.png")
im_search_bar = Image.open("../pics/search bar.png")

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Singers prediction', 'Songs per Singer', 'Singers per song', 'Search'\
                                       ],
        icons=['house', 'bi bi-robot', 'bi bi-music-note-list', 'bi bi-music-note', 'bi bi-search'], menu_icon="cast", default_index=0, \
 \
       styles={
           "nav-link-selected": {"background-color": "#616161"},
       }
    )

# title and description

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(form, png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    form.markdown(page_bg_img, unsafe_allow_html=True)

    form.markdown(page_bg_img, unsafe_allow_html=True)
    return

def features_extractor(file):
    #load the file (audio)
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    #we extract mfcc
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    #in order to find out scaled feature we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

bg1_path = r"../pics/Cole_porter_blur_bg_1.png"
bg2_path = r"../pics/Cole_porter_blur_bg_2.png"

singers = ['Cole Porter',
         'Dionne Warwick',
         'Ella Fitzgerald',
         'Ethel Merman',
         'Frank Sinatra',
         'Harry Connick',
         'Patti Lupone',
         'Sutton Foster',
         'Sutton Foster -  Tap dances',
         'Nat King Cole',
         'Sarah Vaughan',
         'Ray Charles',
         'Louis Armstrong']

if selected == "Home":
    st.markdown('# <font color=#FFFFFF>Tribute to Cole Porter</font>', unsafe_allow_html=True)
    set_background(st, bg1_path)
    st.markdown('## <font color=#FFFFFF>About</font>', unsafe_allow_html=True)
    st.markdown(
        """<div style="text-align: justify;"><p><font color=#FFFFFF>This app was developed as a tribute to Cole Porter. It predicts the singers of the songs sung by Cole Porter and other famous Jazz singers. It can predict singers such as Louis Armstrong, Nat king Cole, Ray Charles, Frank Sinatra, Sarah Vaughan, Ethel Merman and Ella Fitzgerald.</font></p></div>""",
        unsafe_allow_html=True)
    st.markdown("## <font color=#FFFFFF>Who's Cole Porter</font>", unsafe_allow_html=True)
    st.markdown(
        """<div style="text-align: justify;"><p><strong><font color=#FFFFFF>Cole Albert Porter</strong> (June 9, 1891 – October 15, 1964) was an American composer and songwriter. Many of his songs became standards noted for their witty, urbane lyrics, and many of his scores found success on Broadway and in film.\nBorn to a wealthy family in Indiana, Porter defied his grandfather's wishes and took up music as a profession. Classically trained, he was drawn to musical theatre. After a slow start, he began to achieve success in the 1920s, and by the 1930s he was one of the major songwriters for the Broadway musical stage. Unlike many successful Broadway composers, Porter wrote the lyrics as well as the music for his songs. After a serious horseback riding accident in 1937, Porter was left disabled and in constant pain, but he continued to work. His shows of the early 1940s did not contain the lasting hits of his best work of the 1920s and 1930s, but in 1948 he made a triumphant comeback with his most successful musical, <i>Kiss Me, Kate</i>. It won the first Tony Award for Best Musical.\nPorter's other musicals include Fifty Million Frenchmen, DuBarry Was a Lady, Anything Goes, Can-Can and Silk Stockings. His numerous hit songs include "Night and Day", "Begin the Beguine", "I Get a Kick Out of You", "Well, Did You Evah!", "I've Got You Under My Skin", "My Heart Belongs to Daddy" and "You're the Top". He also composed scores for films from the 1930s to the 1950s, including Born to Dance (1936), which featured the song "You'd Be So Easy to Love"; Rosalie (1937), which featured "In the Still of the Night"; High Society (1956), which included "True Love"; and Les Girls (1957). For more additional information about Cole Porter, you can use the Huawei search bar.</font></p></div>""",
        unsafe_allow_html=True)
elif selected == "Singers prediction":
    prediction_form = st.form("prediction")
    set_background(prediction_form, bg2_path)
    prediction_form.markdown('## <font color=#FFFFFF>Singers prediction</font>', unsafe_allow_html=True)
    prediction_form.markdown("""<div style="text-align: left;"><font color=#FFFFFF>Enter a Cole Porter song:</font></div>""", unsafe_allow_html=True)
    model_path = r'../pickle/Cole-Porter-mfcc-neural-network-model-95p.h5'
    l_encoder_path = '../pickle/Cole-Porter-label-encoder.pkl'
    file = prediction_form.file_uploader("",type=['mp3', 'ogg', 'flac', 'm4a'], accept_multiple_files=False)
    is_clk_pred = prediction_form.form_submit_button("Predict")
    if is_clk_pred:
        extracted_features_pred = []
        relative_path = r'Data/Test/' + file.name
        with open(relative_path, mode='wb') as f:
            f.write(file.getvalue())
        data = features_extractor(relative_path)
        file_name = file.name
        extracted_features_pred.append([data, relative_path, file_name])
        pred_extracted_features_df = pd.DataFrame(extracted_features_pred, columns=['feature', 'relative_path', 'File_name'])
        model = tf.keras.models.load_model(model_path)
        input = open(l_encoder_path, 'rb')
        le = pickle.load(input)
        input.close()
        X_pred = np.array(pred_extracted_features_df['feature'].tolist())
        y_pred_test = model.predict(X_pred).argmax(axis=-1)
        pred_extracted_features_df["Predicted_Class"] = y_pred_test
        pred_extracted_features_df['Predicted_Class'] = le.inverse_transform(pred_extracted_features_df['Predicted_Class'])
        pred_extracted_features_df["predicted_Singer"] = pred_extracted_features_df["Predicted_Class"].apply(
            lambda x: ' - '.join(x.split(" - ")[1:]))
        singer = pred_extracted_features_df["predicted_Singer"].values[-1]