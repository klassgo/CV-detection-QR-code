from io import BytesIO 
import json
from pathlib import Path
import streamlit as st
from reader import CodeReader, DEFAULT_WEIGHTS, load_image, annotate

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Conveyor Vision · QR reader',page_icon='📦',layout='wide')
st.markdown('''<style>
.block-container{max-width:1280px;padding-top:2.5rem}
h1{letter-spacing:-.045em} [data-testid="stMetric"]{background:white;border:1px solid #e1e7ee;border-radius:14px;padding:16px}
div.stButton>button[kind="primary"]{border-radius:10px}
</style>''',unsafe_allow_html=True)

st.title('Поиск QR-кодов на изображении')
st.write('Загрузите фотографию с QR-кодами или попробуйте готовый пример.')


@st.cache_resource
def get_reader(path,modified):
    return CodeReader(path)


with st.sidebar:

    st.header('Параметры чтения')
    mode_label=st.radio('Способ обработки',['YOLO + декодер','Только декодер'])
    mode='yolo' if mode_label=='YOLO + декодер' else 'decoder'

    conf=st.slider('Порог уверенности',.05,.95,.25,.05,disabled=mode=='decoder')
    imgsz=st.select_slider('Размер входа детектора',options=[640,1024,1280],value=1024,disabled=mode=='decoder')

    st.divider()
    st.caption('Обработка на CPU. Фотографии не отправляются во внешние API.')

upload=st.file_uploader('Фотография',type=['jpg','jpeg','png','webp'],max_upload_size=20)
options={}
local_examples=sorted((ROOT/'data/valid/images').glob('*.jpg'))[:4]

for i,path in enumerate(local_examples,1):options[f'Фото из validation · {i}']=path
chosen=st.selectbox('Или готовый пример',list(options)) if options else None

if upload is not None:
    source,source_label=upload.getvalue(),upload.name

elif chosen is not None:
    source,source_label=options[chosen],chosen

else:
    st.info('Загрузите фотографию для чтения QR-кодов.');st.stop()

try:
    image=load_image(source)
except (OSError,ValueError) as error:
    st.error(f'Не удалось открыть изображение: {error}');st.stop()

import hashlib
key=(hashlib.sha256(image.tobytes()).hexdigest(),mode,conf,imgsz)

if st.button('Найти и прочитать QR-коды',type='primary',width='stretch'):
    try:
        with st.spinner('Ищем области и читаем содержимое…'):
            reader=get_reader(str(DEFAULT_WEIGHTS),DEFAULT_WEIGHTS.stat().st_mtime_ns)
            report=reader.read(image,confidence=conf,imgsz=imgsz,mode=mode)
            st.session_state['result']=(key,report)

    except Exception as error:
        st.error(f'Не удалось выполнить обработку: {error}')

saved=st.session_state.get('result')

report=saved[1] if saved and saved[0]==key else None

left,right=st.columns([1.55,1],gap='large')

with left:
    st.subheader('Результат на изображении' if report else 'Исходное изображение')
    st.image(annotate(image,report) if report else image,caption=source_label,width='stretch')
    if report:st.caption('Зелёная рамка — содержимое прочитано. Оранжевая — область найдена, но прочитать её не удалось.')

with right:

    st.subheader('Содержимое кодов')
    if report is None:
        st.info('Нажмите «Найти и прочитать коды», чтобы увидеть результат.')

    else:
        a,b,c=st.columns(3)
        a.metric('Найдено',len(report['detections']))
        b.metric('Прочитано',len(report['codes']))
        c.metric('Время',f"{report['timing_ms']['total']:.0f} мс")

        if not report['detections']:st.info('Коды не найдены. Попробуйте другое изображение или уменьшите порог уверенности.')
        elif not report['codes']:st.warning('Области найдены, но содержимое не прочитано. Попробуйте более чёткий кадр.')

        for detection in report['detections']:

            with st.container(border=True):

                score=detection['confidence']
                st.markdown(f"**Область #{detection['id']}**" + (f" · уверенность {score:.0%}" if score is not None else ''))
                matched=[code for code in report['codes'] if code['id'] in detection['code_ids']]

                for code in matched:

                    st.caption(code['format'])
                    st.code(code['text'] or ('HEX: '+code['bytes_hex']),language=None,wrap_lines=True)
                if not matched:st.caption('Найден, но не прочитан')

        st.download_button('Скачать результат JSON',json.dumps(report,ensure_ascii=False,indent=2),file_name='barcode_result.json',mime='application/json',width='stretch')
        buffer=BytesIO();annotate(image,report).save(buffer,format='PNG')
        st.download_button('Скачать изображение с рамками',buffer.getvalue(),file_name='barcode_result.png',mime='image/png',width='stretch')
        st.caption('Время первого запуска включает прогрев модели; загрузка весов в это значение не входит.')

st.divider()
st.caption('YOLO26m → области QR → ZXing → содержимое.')
