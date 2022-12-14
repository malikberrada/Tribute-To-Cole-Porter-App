import streamlit as st
import pandas as pd
import os
import base64
import librosa
import librosa.display
import numpy as np
import pickle
from PIL import Image
from streamlit_option_menu import option_menu
import tensorflow as tf
import glob
import multiprocessing
import platform
import threading
from obs import ObsClient, CompletePart, CompleteMultipartUploadRequest
from streamlit.scriptrunner import add_script_run_ctx, get_script_run_ctx
from threading import Thread
import glob

try:
    im = Image.open("../pics/music.jpg")
except Exception as e:
    root_folders = glob.glob("/*")
    st.error("Can't open App icon :" + "\n"+ str(root_folders))
st.set_page_config(
                    page_title="Tribute to Cole Porter",
                    page_icon=im,
                    layout="wide",
                    initial_sidebar_state="expanded",
                  )

side_bg_ext = "png"
side_bg = "../pics/Happy-Cole-Porter-3.png"

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Songs per Singer', 'Singers per song', 'Storage Cloud', 'Cloud Playback', 'Who sang ?', 'Report a bug', 'Get help'\
                                       ],
        icons=['house', 'bi bi-music-note-list', 'bi bi-music-note', 'bi bi-cloud-upload', 'bi bi-cloud-download', 'bi bi-robot', 'bi bi-bug', 'bi bi-info-circle'], menu_icon="cast", default_index=0, \
 \
       styles={
           "container": {"background-image": f"""url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});background-repeat:repeat"""
    },
           "icon": {"color": "#FFFFFF"},
           "nav-link-selected": {"background-color": "#616161", "opacity": 0.66},
           "nav-link": {"color": "#FFFFFF"},
           "menu-title": {"color":"#FFFFFF"}
       }
    )

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(form, png_file):
    try:
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
    except Exception as e:
        form.error("Can't set background.")

def features_extractor(file):
    try:
        #load the file (audio)
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        #we extract mfcc
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        #in order to find out scaled feature we do mean of transpose of value
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        return mfccs_scaled_features
    except Exception as e:
        st.error("Can't extract features.")

bg1_path = r"../pics/Cole_porter_blur_bg_1_3.png"
bg2_path = r"../pics/Cole_porter_blur_bg_2_5.png"
bg3_path = r"../pics/Cole_porter_blur_bg_3_3.png"
bg4_path = r"../pics/Cole_porter_blur_bg_4_6.png"
bg5_path = r"../pics/Cole_porter_blur_bg_3_5_2.png"
bg6_path = r"../pics/Cole_porter_blur_bg_6_2.png"
bg7_path = side_bg

dict_singers_pics = {}

dict_singers_pics['Cole Porter'] = "../pics/singers_pics/Coleporter.jpg"
dict_singers_pics['Dionne Warwick'] = "../pics/singers_pics/Dionne_Warwick_2021.jpg"
dict_singers_pics['Ella Fitzgerald'] = "../pics/singers_pics/Ella Fitzgerald.jpg"
dict_singers_pics['Ethel Merman'] = "../pics/singers_pics/Ethel_merman_1967.jpg"
dict_singers_pics['Frank Sinatra'] = "../pics/singers_pics/Frank Sinatra.jpg"
dict_singers_pics['Harry Connick'] = "../pics/singers_pics/Harry_Connick.jpg"
dict_singers_pics['Patti Lupone'] = "../pics/singers_pics/Patti-LuPone.jpg"
dict_singers_pics['Sutton Foster'] = "../pics/singers_pics/Sutton-Foster.jpg"
dict_singers_pics['Nat King Cole'] = "../pics/singers_pics/Nat King Cole.jpg"
dict_singers_pics['Sarah Vaughan'] = "../pics/singers_pics/Sarah Vaughan 2.jpg"
dict_singers_pics['Ray Charles'] = "../pics/singers_pics/Ray_Charles.jpg"
dict_singers_pics['Louis Armstrong'] = "../pics/singers_pics/Louis_Armstrong.jpg"
dict_singers_pics['Sutton Foster -  Tap dances'] = "../pics/singers_pics/Sutton-Foster - tap dances-3.gif"

dict_songs = {}
dict_songs['Anything goes'] = 'Anything goes'
dict_songs['Begin the Beguine'] = 'Begin the Beguine'
dict_songs['Blue Skies'] = 'Blue Skies'
dict_songs['C est magnifique'] = "C'est magnifique"
dict_songs['Don t Fence in Me'] = "Don't Fence Me in"
dict_songs['Easy to Love'] = 'Easy to Love'
dict_songs['Ev ry time we say goodbye'] = "Ev'ry time we say goodbye"
dict_songs['I Get A Kick Out Of You'] = 'I Get A Kick Out Of You'
dict_songs['I concentrate on you'] = 'I concentrate on you'
dict_songs['I love Paris'] = 'I love Paris'
dict_songs['I ve got you under my skin'] = "I've got you under my skin"
dict_songs['In the still of the night'] = 'In the still of the night'
dict_songs['It s de-lovely'] = "It's de-lovely"
dict_songs['Let s do it'] = "Let's do it"
dict_songs['Love for Sale'] = 'Love for Sale'
dict_songs['Night and day'] = 'Night and day'
dict_songs['So In Love'] = 'So In Love'
dict_songs['What is this thing called love'] = 'What is this thing called love'
dict_songs['You Do Something To Me'] = 'You Do Something To Me'
dict_songs['You re the top'] = "You're the top"

if selected == "Home":
    st.markdown('# <font color=#FFFFFF>Tribute to Cole Porter</font>', unsafe_allow_html=True)
    set_background(st, bg1_path)
    st.markdown("## <font color=#FFFFFF>Who's Cole Porter</font>", unsafe_allow_html=True)
    st.markdown(
        """<div style="text-align: justify;"><p><font color=#FFFFFF>American composer and lyricist <strong>Cole Porter</strong> brought international momentum to American musical comedy, embodying the sophistication of his songs in his life.</font></p><p><font color=#FFFFFF>Born June 9, 1891 in Peru, Indiana, Porter is the grandson of a millionaire speculator, and the affluence in which he lived likely played a role in the poise and urbanity of his musical style. He began to study the violin at the age of six and the piano at eight; at ten he wrote an operetta in the style of Gilbert and Sullivan and saw his first composition, a waltz, published a year later. While studying at Yale, he composed approximately three hundred songs, including <i>Eli</i>, <i>Bulldog</i>, and <i>Bingo Eli Yale</i>, as well as performing for faculty; he continued his studies at Harvard Law School (1914) and at the Harvard Graduate School of Arts and Sciences in Music (1915-1916). He made his Broadway debut with the musical <i>See America First</i> (1916), which however left the bill after fifteen performances.</font></p><p><font color=#FFFFFF>In 1917, after the United States entered the war, Porter went to France (without joining the Allied troops, as will be said later). He became a traveling playboy in Europe and, although he was quite openly gay, married a wealthy divorced American older than him, Linda Lee Thomas, on December 18, 1919; they spent the next twenty years running social parties and taking group trips, sometimes together, sometimes separately.</p><p>In 1928, Porter composed several songs for a successful Broadway play, <i>Paris</i>. A string of successful musicals followed, including <i>Fifty Million Frenchmen</i> (1929), <i>Gay Divorc??e</i> (1932), <i>Anything Goes</i> (1934), <i>Red, Hot and Blue</i> (1934), <i>Jubilee</i> (1935), <i>Dubarry Was a Lady</i> (1939) , <i>Panama Hattie</i> (1940), <i>Kiss me, Kate</i> (1948, based on Shakespeare's <i>The Taming of the Shrew</i>), <i>Can-Can</i> (1953) and <i>Silk Stockings</i> (1955). At the same time, he worked on the music for several films. Over the years he has written songs and lyrics as brilliant as <i>Night and Day</i>, <i>I Get a Kick out of you</i>, <i>Begin the Beguine</i>, </i>I've Got you Under my Skin</i>, <i>In The Still of The Night</i>, <i>Just One of Those Things</i>, <i>Love for Sale</i>, <i>My Heart Belongs to Daddy</i>, <i>Too Darn Hot</i>, <i>It's Delovely</i>, <i>I Concentrate on you</i>, <i>Always True to You in My Fashion</i>, and <i>I Love Paris</i>. He has the art of making songs that have entered the repertoire, such as <i>Let's Do It</i> and <i>You're the Top</i>, the best known.<br>You can download my list of musics and store them in Huawei storage Cloud.</font></p></div>""",
        unsafe_allow_html=True)
    st.markdown('## <font color=#FFFFFF>About</font>', unsafe_allow_html=True)
    st.markdown(
        """<div style="text-align: justify;"><p><font color=#FFFFFF>This app was developed as a tribute to Cole Porter. It collects his main known songs, as well as the main singers who sang his songs like: Louis Armstrong, Nat king Cole, Ray Charles, Frank Sinatra, Sarah Vaughan, Ethel Merman and Ella Fitzgerald. In addition, if you have a song whose singer you don't know and which can be sung by one of the singers mentioned above, the application can find it for you. Porter is one of the wittiest lyricists in the world, with a subtle expression and a great mastery of inner rhythm. We have developed this application to introduce the rhythms of popular songs to today's youth and awaken hidden talents around the world. Porter's work remains a model of elegance and refinement in the popular song genre.<br>This app can find the singers of the songs sung by Cole Porter and other famous Jazz singers with an accuracy of 85.2%. It can also find the most likely song with 78.6% accuracy.</font></p></div>""",
        unsafe_allow_html=True)




elif selected == "Who sang ?":
    prediction_form = st.form("prediction")
    set_background(prediction_form, bg2_path)
    prediction_form.markdown('## <font color=#FFFFFF>Who sang ?</font>', unsafe_allow_html=True)
    prediction_form.markdown(
        """<div style="text-align: justify;font-size:16px"><font color=#FFFFFF>If you have songs of which you don't know the singer and which can be sung by the following singers: Louis Armstrong, Nat king Cole, Ray Charles, Frank Sinatra, Sarah Vaughan, Ethel Merman and Ella Fitzgerald, you can predict the real singer through this app. It can do this with an accuracy of 85.2%. It can also predict the most likely song with 78.6% accuracy.</font></div>""",
        unsafe_allow_html=True)
    prediction_form.markdown("""<div style="text-align: left;font-size:16px"><font color=#FFFFFF>Enter a Cole Porter song:</font></div>""", unsafe_allow_html=True)
    model_path = r'../pickle/Cole-Porter-mfcc-neural-network-model-95p.h5'
    l_encoder_path = '../pickle/Cole-Porter-label-encoder.pkl'
    try:
        file = prediction_form.file_uploader("",type=['mp3', 'ogg', 'flac', 'm4a'], accept_multiple_files=False, key="precition_file_uploader")
    except Exception as e:
        prediction_form.error("Invalid file format.")
    is_clk_pred = prediction_form.form_submit_button("Who sang ?")
    if is_clk_pred:
        extracted_features_pred = []
        try:
            relative_path = r'../Data/Test/' + file.name
            with open(relative_path, mode='wb') as f:
                f.write(file.getvalue())
            data = features_extractor(relative_path)
            file_name = file.name
            extracted_features_pred.append([data, relative_path, file_name])
            pred_extracted_features_df = pd.DataFrame(extracted_features_pred, columns=['feature', 'relative_path', 'File_name'])
        except Exception as e:
            prediction_form.error("Invalid file format.")
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            prediction_form.error("Can't load the model.")
        try:
            input = open(l_encoder_path, 'rb')
            le = pickle.load(input)
            input.close()
            X_pred = np.array(pred_extracted_features_df['feature'].tolist())
        except Exception as e:
            prediction_form.error("Can't load the encoder.")
        try:
            y_pred_test = model.predict(X_pred).argmax(axis=-1)
            pred_extracted_features_df["Predicted_Class"] = y_pred_test
            pred_extracted_features_df['Predicted_Class'] = le.inverse_transform(pred_extracted_features_df['Predicted_Class'])
            pred_extracted_features_df["predicted_Singer"] = pred_extracted_features_df["Predicted_Class"].apply(
                lambda x: ' - '.join(x.split(" - ")[1:]))
            pred_extracted_features_df["predicted_Song"] = pred_extracted_features_df["Predicted_Class"].apply(
                lambda x: x.split(" - ")[0])
            singer = pred_extracted_features_df["predicted_Singer"].values[-1]
            pred_song = [k for k, v in dict_songs.items() if v == pred_extracted_features_df["predicted_Song"].values[-1]][-1]
            prediction_form.markdown(
                """<div style="text-align: left;font-size:17px"><font color=#FFFFFF>Its singer is </font><font color=#204E1E>""" + singer.split(" - ")[0] + """</font></div>""",
                unsafe_allow_html=True)
            os.remove(relative_path)
            if ".gif" in dict_singers_pics[singer]:
                try:
                    file_ = open(dict_singers_pics[singer], "rb")
                    contents = file_.read()
                    data_url = base64.b64encode(contents).decode("utf-8")
                    file_.close()
                except Exception as e:
                    prediction_form.error("Can't open the singer image.")
                prediction_form.markdown(
                    f'<br><img src="data:image/gif;base64,{data_url}" alt="cat gif">',
                    unsafe_allow_html=True,
                )
            else:
                try:
                    image = Image.open(dict_singers_pics[singer])
                except Exception as e:
                    prediction_form.error("Can't open the singer image.")
                prediction_form.image(image, caption='')
            if "tap dances" in singer.lower():
                prediction_form.markdown(
                    """<div style="text-align: left;font-size:17px"><font color=#FFFFFF>Its a </font><font color=#204E1E>Tap dances</font><font color=#FFFFFF> song !</font></div>""",
                    unsafe_allow_html=True)
            directory = '../Data/songs - ogg - Orig'
            try:
                for song in glob.iglob(f'{directory}/*'):
                    sg = song.replace("\\", "/")
                    if sg.split("/")[-1] == pred_song:
                        for sgr in glob.iglob(f'{song}/*'):
                            sgr = sgr.replace("\\", "/")
                            if sgr.split("/")[-1] == singer:
                                cpt = 0
                                for song_sp in glob.iglob(f'{sgr}/*'):
                                    audio_name = song_sp.replace("\\", "/")
                                    orig_audio_name = audio_name
                                    prediction_form.markdown(
                                        """<div style="text-align: left;font-size:16px"><font color=#FFFFFF>Please, wait for loading the song...</font></div>""",
                                        unsafe_allow_html=True)
                                    try:
                                        audio_file = open(orig_audio_name, 'rb')
                                        audio_bytes = audio_file.read()
                                        mymidia_str = "data:audio/ogg;base64,%s" % (
                                            base64.b64encode(audio_bytes).decode())
                                        mymidia_html = """
                                                        <audio autoplay class="stAudio">
                                                        <source src="%s" type="audio/ogg">
                                                        Your browser does not support the audio element.
                                                        </audio>
                                                    """ % mymidia_str
                                        prediction_form.markdown(mymidia_html, unsafe_allow_html=True)
                                    except Exception as e:
                                        st.error("Songs not found.")
                                    prediction_form.markdown(
                                        """<div style="text-align: left;font-size:16px"><font color=#FFFFFF>You can stop the song by switching tabs.</font></div>""",
                                        unsafe_allow_html=True)
                                    songs_found = True
                                    cpt += 1
                                    if cpt == 1:
                                        break
                                break
                        break
            except Exception as e:
                st.error("Songs not found.")
        except Exception as e:
            prediction_form.error("Can't predict the Data.")
elif selected == 'Songs per Singer':
    dict_singers_pics_2 = {}
    dict_singers_pics_2['Cole Porter'] = "../pics/singers_pics_s_singer/Coleporter_3.jpg"
    dict_singers_pics_2['Dionne Warwick'] = "../pics/singers_pics_s_singer/Dionne_Warwick_2021_3.jpg"
    dict_singers_pics_2['Ella Fitzgerald'] = "../pics/singers_pics_s_singer/Ella Fitzgerald_3.jpg"
    dict_singers_pics_2['Ethel Merman'] = "../pics/singers_pics_s_singer/Ethel_merman_1967_3.jpg"
    dict_singers_pics_2['Frank Sinatra'] = "../pics/singers_pics_s_singer/Frank Sinatra_3.jpg"
    dict_singers_pics_2['Harry Connick'] = "../pics/singers_pics_s_singer/Harry_Connick_3.jpg"
    dict_singers_pics_2['Patti Lupone'] = "../pics/singers_pics_s_singer/Patti-LuPone_3.jpg"
    dict_singers_pics_2['Sutton Foster'] = "../pics/singers_pics_s_singer/Sutton-Foster_3.jpg"
    dict_singers_pics_2['Nat King Cole'] = "../pics/singers_pics_s_singer/Nat King Cole_3.jpg"
    dict_singers_pics_2['Sarah Vaughan'] = "../pics/singers_pics_s_singer/Sarah Vaughan 2_3.jpg"
    dict_singers_pics_2['Ray Charles'] = "../pics/singers_pics_s_singer/Ray_Charles_3.jpg"
    dict_singers_pics_2['Louis Armstrong'] = "../pics/singers_pics_s_singer/Louis_Armstrong_3.jpg"
    dict_singers_pics_2['Sutton Foster -  Tap dances'] = "../pics/singers_pics_s_singer/Sutton-Foster_3.jpg"
    set_background(st, bg3_path)
    st.markdown('## <font color=#FFFFFF>Songs per Singer</font>', unsafe_allow_html=True)
    st.markdown("""<div style="text-align: left;font-size:16px"><font color=#FFFFFF>Choose the singer:</font></div>""", unsafe_allow_html=True)
    singer = st.selectbox('', ('Cole Porter',
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
         'Louis Armstrong'))

    try:
        image = Image.open(dict_singers_pics_2[singer])
    except Exception as e:
        st.error("Can't open the singer image.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image(image, caption='')

    with col3:
        st.write(' ')

    songs_found = False
    # assign directory
    directory = '../Data/songs - ogg - Orig'
    try:
        for song in glob.iglob(f'{directory}/*'):
            for sgr in glob.iglob(f'{song}/*'):
                sgr = sgr.replace("\\", "/")
                if sgr.split("/")[-1] == singer:
                    cpt = 0
                    for song_sp in glob.iglob(f'{sgr}/*'):
                        audio_name = song_sp.replace("\\", "/")
                        orig_audio_name = audio_name
                        st.markdown(
                            """<div style="text-align: left;font-size:16px"><font color=#FFFFFF>""" + orig_audio_name.split("/")[-1].replace(".ogg", "") + """</font></div><br>""",
                            unsafe_allow_html=True)
                        try:
                            audio_file = open(orig_audio_name, 'rb')
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format='audio/ogg', start_time=0)
                        except Exception as e:
                            st.error("Songs not found.")
                        songs_found = True
                        cpt += 1
                        if cpt == 1:
                            break
                    break
    except Exception as e:
        st.error("Songs not found.")
    if not songs_found:
        st.error("Songs not found.")
elif selected == 'Singers per song':
    set_background(st, bg4_path)
    st.markdown('## <font color=#FFFFFF>Singers per song</font>', unsafe_allow_html=True)
    st.markdown("""<div style="text-align: left;font-size:16px"><font color=#FFFFFF>Choose the song:</font></div>""", unsafe_allow_html=True)
    song_name = st.selectbox('', (dict_songs['Anything goes'],
                             dict_songs['Begin the Beguine'],
                             dict_songs['Blue Skies'],
                             dict_songs['C est magnifique'],
                             dict_songs['Don t Fence in Me'],
                             dict_songs['Easy to Love'],
                             dict_songs['Ev ry time we say goodbye'],
                             dict_songs['I Get A Kick Out Of You'],
                             dict_songs['I concentrate on you'],
                             dict_songs['I love Paris'],
                             dict_songs['I ve got you under my skin'],
                             dict_songs['In the still of the night'],
                             dict_songs['It s de-lovely'],
                             dict_songs['Let s do it'],
                             dict_songs['Love for Sale'],
                             dict_songs['Night and day'],
                             dict_songs['So In Love'],
                             dict_songs['What is this thing called love'],
                             dict_songs['You Do Something To Me'],
                             dict_songs['You re the top']))
    songs_found = False
    # assign directory
    directory = '../Data/songs - ogg - Orig'
    try:
        for song in glob.iglob(f'{directory}/*'):
            song = song.replace("\\", "/")
            if song.split("/")[-1] == [k for k, v in dict_songs.items() if v == song_name][-1]:
                for sgr in glob.iglob(f'{song}/*'):
                    sgr = sgr.replace("\\", "/")
                    cpt = 0
                    for song_sp in glob.iglob(f'{sgr}/*'):
                        audio_name = song_sp.replace("\\", "/")
                        orig_audio_name = audio_name
                        st.markdown(
                            """<div style="text-align: left;font-size:16px"><font color=#FFFFFF>""" + orig_audio_name.split("/")[-2] + """</font></div><br>""",
                            unsafe_allow_html=True)
                        try:
                            audio_file = open(orig_audio_name, 'rb')
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format='audio/ogg', start_time=0)
                        except Exception as e:
                            st.error("Exception: "+ e + "Singer not found.")
                        songs_found = True
                        cpt += 1
                        if cpt == 1:
                            break
    except Exception as e:
        st.error("Exception: "+ e + "Singer not found.")
    if not songs_found:
        st.error("Singer not found.")
elif (selected == 'Storage Cloud'):
    upload_form = st.form("upload_form")
    file_uploaded = False
    def doCopyPart(partETags, bucketName, objectKey, partNumber, uploadId, copySource, copySourceRange):
        try:
            if IS_WINDOWS:
                global obsClient
            else:
                obsClient = ObsClient(access_key_id=AK, secret_access_key=SK, server=server)
            resp = obsClient.copyPart(bucketName=bucketName, objectKey=objectKey, partNumber=partNumber, uploadId=uploadId,
                                      copySource=copySource, copySourceRange=copySourceRange)
        except Exception as e:
            st.error("We can' generate a response.")
        try:
            if resp.status < 300:
                partETags[partNumber] = resp.body.etag
                upload_form.markdown(
                    """<div style="text-align: left;font-size:16px"><font color=#FBFBFB>Part """ + str(partNumber) + """ done</font></div>""",
                    unsafe_allow_html=True
                    )
            else:
                upload_form.markdown(
                    """<div style="text-align: left;font-size:16px"><font color=#FBFBFB>Part """ + str(
                        partNumber) + """ failed</font></div>""",
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error("We can't divide song into parts.")
    try:
        ctx = get_script_run_ctx()
        t = Thread(target=doCopyPart)
        add_script_run_ctx(t)
        t.start()
        t.join()
        set_background(upload_form, bg5_path)
        storage_cloud_title=upload_form.markdown('## <font color=#FBFBFB>Storage Cloud</font>', unsafe_allow_html=True)
        storage_cloud_chse_song=upload_form.markdown("""<div style="text-align: left;font-size:16px"><font color=#FBFBFB>Choose a song to upload:</font></div>""",
                    unsafe_allow_html=True)
        relative_path = ''
    except Exception as e:
        st.error("We can't generate thread.")
    try:
        file = upload_form.file_uploader("",type=['mp3', 'ogg', 'flac', 'm4a', 'oga'], accept_multiple_files=False, key='obs_storage_file_uploader')
    except Exception as e:
        upload_form.error("Invalid file format.")
    is_clk_upload = upload_form.form_submit_button("Upload")
    AK = '9WEQIWKRMG0LEICD6CFU'
    SK = 'L1R1KYVgtSjGfuUR7vOH7bwYdW9WAKqgDznal3j4'
    server = 'obs.ap-southeast-1.myhuaweicloud.com'
    bucketName = 'cole-porter-app-storage'
    if is_clk_upload:
        for i in range(6):
            if i== 0:
                try:
                    relative_path = r'../Data/Test/' + file.name
                    sourceBucketName = bucketName
                    sourceObjectKey = file.name
                    objectKey = sourceObjectKey + '-back'
                    sampleFilePath = relative_path
                    with open(relative_path, mode='wb') as f:
                        f.write(file.getvalue())
                    IS_WINDOWS = platform.system() == 'Windows' or os.name == 'nt'
                    # Constructs a obs client instance with your account for accessing OBS
                    obsClient = ObsClient(access_key_id=AK, secret_access_key=SK, server=server)
                    upload_form.markdown("""<div style="text-align: left;font-size:16px"><font color=#FBFBFB>Uploading a song...</font></div>""",
                                         unsafe_allow_html=True
                                         )
                    upload_form.markdown(
                        """<div style="text-align: left;font-size:16px"><font color=#FBFBFB>Please, wait a couple minutes...</font></div><br>""",
                        unsafe_allow_html=True
                        )
                    my_bar = upload_form.progress(0)
                    resp = obsClient.putFile(sourceBucketName, sourceObjectKey, sampleFilePath)
                    if resp.status >= 300:
                        upload_form.error("We can't upload the song")

                    # Claim a upload id firstly
                    resp = obsClient.initiateMultipartUpload(bucketName, objectKey)
                    if resp.status >= 300:
                        upload_form.error("We can't upload the song")

                    uploadId = resp.body.uploadId
                    my_bar.progress(int(100 * (i + 1) / 6))
                    continue
                except Exception as e:
                    st.error("We can't generate a response.")
            elif i == 1:
                try:
                    upload_form.markdown("""<div style="text-align: left;font-size:16px"><font color=#FBFBFB>Claiming an upload id...</font></div>""",
                                         unsafe_allow_html=True)
                    # 5MB
                    partSize = 5 * 1024 * 1024
                    resp = obsClient.getObjectMetadata(sourceBucketName, sourceObjectKey)
                    if resp.status >= 300:
                        upload_form.error("We can't upload the song")

                    header = dict(resp.header)
                    objectSize = int(header.get('content-length'))

                    partCount = int(objectSize / partSize) if (objectSize % partSize == 0) else int(objectSize / partSize) + 1

                    if partCount > 10000:
                        upload_form.error("We can't upload the song")
                    my_bar.progress(int(100 * (i + 1) / 6))
                    continue
                except Exception as e:
                    st.error("We can't get metadata.")
            elif i == 2:
                try:
                    upload_form.markdown("""<div style="text-align: left;font-size:16px"><font color=#FBFBFB>Dividing the song into parts...</font></div><br>""",
                                         unsafe_allow_html=True)
                    # Upload multiparts by copy mode
                    upload_form.markdown("""<div style="text-align: left;font-size:16px"><font color=#FBFBFB>Beginning to upload the parts to the Cloud Storage Service...</font></div>""",
                                         unsafe_allow_html=True)
                    proc = add_script_run_ctx(threading.Thread, ctx) if IS_WINDOWS else multiprocessing.Process

                    partETags = dict() if IS_WINDOWS else multiprocessing.Manager().dict()
                except:
                    st.error("We can't generate the thread.")

                processes = []

                try:
                    for i in range(partCount):
                        rangeStart = i * partSize
                        rangeEnd = objectSize - 1 if (i + 1 == partCount) else rangeStart + partSize - 1

                        p = proc(target=doCopyPart, args=(
                            partETags, bucketName, objectKey, i + 1, uploadId, sourceBucketName + '/' + sourceObjectKey,
                            str(rangeStart) + '-' + str(rangeEnd)))
                        processes.append(p)

                    for p in processes:
                        p.start()

                    for p in processes:
                        p.join()

                    if len(partETags) != partCount:
                        upload_form.error("We can't upload the song")
                    my_bar.progress(int(100 * (i + 1) / 6))
                    continue
                except Exception as e:
                    st.error("We can't generate the thread.")
            elif i == 3:
                try:
                    # View all parts uploaded recently
                    upload_form.markdown("""<div style="text-align: left;font-size:16px"><font color=#FBFBFB>Listing of the song's parts......</font></div>""",
                                         unsafe_allow_html=True)
                    resp = obsClient.listParts(bucketName, objectKey, uploadId)

                    if resp.status < 300:
                        cpt=0
                        for part in resp.body.parts:
                            cpt+=1
                            if part != resp.body.parts[- 1]:
                                upload_form.markdown("""<div style="text-align: left;font-size:16px"><font color=#FBFBFB>Part """ + str(part.partNumber) + "</font></div>",
                                             unsafe_allow_html=True)
                            else:
                                upload_form.markdown(
                                    """<div style="text-align: left;font-size:16px"><font color=#FBFBFB>Part """ + str(
                                        part.partNumber) + "</font></div><br>",
                                    unsafe_allow_html=True)
                        print('\n')
                    else:
                        upload_form.error('listParts failed')

                    # Complete to upload multiparts

                    partETags = sorted(partETags.items(), key=lambda d: d[0])

                    parts = []
                    for key, value in partETags:
                        parts.append(CompletePart(partNum=key, etag=value))
                    my_bar.progress(int(100 * (i + 1) / 6))
                    continue
                except Exception as e:
                    st.error("We can't list parts.")
            elif i == 4:
                try:
                    upload_form.markdown("""<div style="text-align: left;font-size:16px"><font color=#FBFBFB>Completing to upload the song's parts...</font></div>""",
                                         unsafe_allow_html=True)
                    resp = obsClient.completeMultipartUpload(bucketName, objectKey, uploadId, CompleteMultipartUploadRequest(parts))
                    my_bar.progress(int(100 * (i + 1) / 6))
                    continue
                except Exception as e:
                    st.error("We can't complete multiparts.")
            else:
                try:
                    if resp.status < 300:
                        upload_form.markdown("""<div style="text-align: left;font-size:16px"><font color=#FBFBFB>Song uploaded !</font></div><br>""",
                                         unsafe_allow_html=True)
                    else:
                        upload_form.error("We can't upload the song")
                    os.remove(relative_path)
                    progress_bar=my_bar.progress(int(100 * (i + 1) / 6))
                except Exception as e:
                    st.error("We can't upload the song.")

elif selected == "Cloud Playback":
    download_form = st.form("download_form")
    set_background(download_form, bg6_path)
    download_form.markdown('## <font color=#FBFBFB>Cloud Playback</font>', unsafe_allow_html=True)
    download_form.markdown(
        """<div style="text-align: left;font-size:16px"><font color=#FBFBFB>If you've uploaded a song on the Cloud, please enter it's filename:</font></div>""",
        unsafe_allow_html=True)
    try:
        filename = download_form.text_input('', placeholder ='It must end with .mp3, .ogg, .oga, .wav, .flac, .m4a')
        if filename and (not (filename.endswith(".mp3") or filename.endswith(".ogg") or filename.endswith(".oga") or filename.endswith(".wav") or filename.endswith(".flac") or filename.endswith(".m4a"))):
            st.error("Wrong file extension.")
    except Exception as e:
        st.error("We can't get the filename.")
    download_on_cloud = download_form.form_submit_button("Load")
    if download_on_cloud:
        if filename:
            st.markdown(
                """<div style="text-align: left;font-size:16px"><font color=#FBFBFB>Please, wait while we're loading...</font></div><br>""",
                unsafe_allow_html=True)
            AK = '9WEQIWKRMG0LEICD6CFU'
            SK = 'L1R1KYVgtSjGfuUR7vOH7bwYdW9WAKqgDznal3j4'
            server = 'obs.ap-southeast-1.myhuaweicloud.com'
            bucketName = 'cole-porter-app-storage'
            try:
                obsClient = ObsClient(access_key_id=AK, secret_access_key=SK, server=server)
                bucketClient = obsClient.bucketClient(bucketName)
                resp = bucketClient.getObject(filename, loadStreamInMemory=True)
                if resp.status < 300:
                    response = resp.body.buffer
                    if response is not None:
                        try:
                            st.audio(response, format='audio/ogg', start_time=0)
                        except Exception as e:
                            st.error("Exception: " + e + "Singer not found.")
                    else:
                        st.error("Wrong filename.")
                else:
                    st.error("The song isn't available in the Cloud.")
            except Exception as e:
                st.error("We can't download the song.")
        else:
            st.error('Wrong filename.')

elif selected == 'Report a bug':
    set_background(st, bg7_path)
    st.markdown('## <font color=#FBFBFB>Report a bug</font>', unsafe_allow_html=True)
    st.markdown(
        """<div style="text-align: left;font-size:16px"><font color=#FBFBFB>If you notice any bug, please contact the support service at: abdelmalik.berrada@gmail.com</font></div>""",
        unsafe_allow_html=True)

else:
    set_background(st, bg7_path)
    st.markdown('## <font color=#FBFBFB>Get help</font>', unsafe_allow_html=True)
    st.markdown(
        """<div style="text-align: left;font-size:19px"><font color=#FBFBFB>1. "Home" tab:</font></div><br>""",
        unsafe_allow_html=True)
    st.markdown(
        """<div style="text-align: justify;font-size:16px"><font color=#FBFBFB>This tab displays the summary of Cole Porter's biography:</font></div><br>""",
        unsafe_allow_html=True)
    try:
        image = Image.open("../pics/Home.png")
    except Exception as e:
        st.error("Can't open the image.")
    st.image(image, caption="Cole Porter's biography")
    st.markdown(
        """<div style="text-align: left;font-size:18px"><font color=#FBFBFB>2. "Songs per Singer" tab:</font></div><br>""",
        unsafe_allow_html=True)
    st.markdown(
        """<div style="text-align: justify;font-size:16px"><font color=#FBFBFB>This tab displays Cole Porter songs by singer. It allows to choose one of the main singers of Cole Porter, to listen to his songs and to download them:</font></div><br>""",
        unsafe_allow_html=True)
    try:
        image = Image.open("../pics/Songs per Singer - 1.png")
    except Exception as e:
        st.error("Can't open the image.")
    st.image(image, caption="Songs per singer")
    try:
        image = Image.open("../pics/Songs per Singer - 2.png")
    except Exception as e:
        st.error("Can't open the image.")
    st.image(image, caption="Songs per singer 2")
    st.markdown(
        """<div style="text-align: left;font-size:18px"><font color=#FBFBFB>3. "Singers per song" tab:</font></div><br>""",
        unsafe_allow_html=True)
    st.markdown(
        """<div style="text-align: justify;font-size:16px"><font color=#FBFBFB>This tab displays the main singers of Cole Porter by song. It allows you to choose a Cole Porter song, listen to it (sung by several singers) and download them.</font></div><br>""",
        unsafe_allow_html=True)
    try:
        image = Image.open("../pics/Singers per song - 1.png")
    except Exception as e:
        st.error("Can't open the image.")
    st.image(image, caption="Singers per song")
    try:
        image = Image.open("../pics/Singers per song - 2.png")
    except Exception as e:
        st.error("Can't open the image.")
    st.image(image, caption="Singers per song 2")
    st.markdown(
        """<div style="text-align: left;font-size:18px"><font color=#FBFBFB>4. "Storage Cloud" tab:</font></div><br>""",
        unsafe_allow_html=True)
    st.markdown(
        """<div style="text-align: justify;font-size:16px"><font color=#FBFBFB>This tab allows you to upload a Cole Porter song to the Huawei Cloud. You can enter the song and upload it:</font></div><br>""",
        unsafe_allow_html=True)
    try:
        image = Image.open("../pics/Storage Cloud - 1.png")
    except Exception as e:
        st.error("Can't open the image.")
    st.image(image, caption="Storage Cloud")
    try:
        image = Image.open("../pics/Storage Cloud - 2.png")
    except Exception as e:
        st.error("Can't open the image.")
    st.image(image, caption="Storage Cloud 2")
    try:
        image = Image.open("../pics/Storage Cloud - 3.png")
    except Exception as e:
        st.error("Can't open the image.")
    st.image(image, caption="Storage Cloud 3")
    st.markdown(
        """<div style="text-align: left;font-size:18px"><font color=#FBFBFB>5. "Cloud Playback" tab:</font></div><br>""",
        unsafe_allow_html=True)
    st.markdown(
        """<div style="text-align: justify;font-size:16px"><font color=#FBFBFB>This tab allows you to play a song uploaded to the Cloud. You can enter the file name of the song. Then you have to load it. Then you can listen to it and download it:</font></div><br>""",
        unsafe_allow_html=True)
    try:
        image = Image.open("../pics/Cloud Playback - 1.png")
    except Exception as e:
        st.error("Can't open the image.")
    st.image(image, caption="Cloud Playback")
    try:
        image = Image.open("../pics/Cloud Playback - 2.png")
    except Exception as e:
        st.error("Can't open the image.")
    st.image(image, caption="Cloud Playback 2")
    try:
        image = Image.open("../pics/Cloud Playback - 3.png")
    except Exception as e:
        st.error("Can't open the image.")
    st.image(image, caption="Cloud Playback 3")
    try:
        image = Image.open("../pics/Cloud Playback - 4.png")
    except Exception as e:
        st.error("Can't open the image.")
    st.image(image, caption="Cloud downloading")
    st.markdown(
        """<div style="text-align: left;font-size:18px"><font color=#FBFBFB>6. "Who sang ?" tab:</font></div><br>""",
        unsafe_allow_html=True)
    st.markdown(
        """<div style="text-align: justify;font-size:16px"><font color=#FBFBFB>If the user listens to a song of which he does not know the singer, the application can search for it for him, provided that it is sung by the following singers: Louis Armstrong, Nat king Cole, Ray Charles, Frank Sinatra, Sarah Vaughan, Ethel Merman and Ella Fitzgerald. You can enter the song and predict it:</font></div><br>""",
        unsafe_allow_html=True)
    try:
        image = Image.open("../pics/Song Prediction - 1.png")
    except Exception as e:
        st.error("Can't open the image.")
    st.image(image, caption="Singers prediction")
    try:
        image = Image.open("../pics/Song Prediction - 3.png")
    except Exception as e:
        st.error("Can't open the image.")
    st.image(image, caption="Singers prediction 2")
    try:
        image = Image.open("../pics/Song Prediction - 5.png")
    except Exception as e:
        st.error("Can't open the image.")
    st.image(image, caption="Singers prediction 3")