import os
from datetime import datetime
from uuid import uuid4

import pandas as pd
from h2o_wave import ui, Q, app, main

from rainforest.config import N_SPECIES
from rainforest.pann import load_pretrained_model, get_model_predictions_for_clip
from rainforest.utils import show_references, get_candidates, get_annotations, visualize_spectograms, \
    get_next_candidate, get_random_positive_example, get_random_negative_example, ANNOTATION_PATH, \
    AUDIO_IMG_TEMPLATE, TMP_PATH, read_audio_fast, max_probability_bar_plot, AUDIO_TEMPLATE, \
    mel_spectrogram, IMAGE_TEMPLATE, framewise_probability_plot, CHUNK_URI

_ = main

CARDS = {
    'references': ['references'],
    'annotate': ['species_selector', 'method_selector', 'tp_selector', 'buttons', 'new', 'example_tp', 'example_fp'],
    'recognize': ['upload', 'framewise_predictions', 'clipwise_predictions', 'audio', 'mel_spectrogram']
}


@app('/')
async def stg(q: Q):
    print('=' * 10)
    print(q.args)
    await display_main_page(q)
    await q.page.save()


async def del_cards(q: Q, cards):
    for c in cards:
        del q.page[c]
    await q.page.save()


def select_new_candidate(q):
    q.client.rec_id, q.client.start, q.client.prob = get_next_candidate(
        q.client.candidates, q.client.annotations, q.client.spec_id, q.client.method, q.client.tp_selector)
    print('New candidate')


def refresh_original_examples(q):
    q.client.pos_rec_id, q.client.pos_start, q.client.pos_prob = get_random_positive_example(q.client.spec_id)
    q.client.neg_rec_id, q.client.neg_start, q.client.neg_prob = get_random_negative_example(q.client.spec_id)
    print('Refresh')


def update_annotations(q, label):
    q.client.annotations = q.client.annotations.append(
        {'rec_id': q.client.rec_id, 'start': q.client.start, 'spec_id': q.client.spec_id, 'label': label},
        ignore_index=True)
    q.client.annotations.to_csv(ANNOTATION_PATH / q.client.filename, index=False)


async def add_progress(q: Q):
    q.page['progress'] = ui.form_card(
        box='1 2 3 2',
        items=[ui.progress(label='Chirp Chirp!')]
    )
    await q.page.save()


async def show_audio_chunk(q, rec_id, spec_id, start, prob, show_boxes, header):
    fig = visualize_spectograms(rec_id, spec_id, start, prob, show_boxes=show_boxes, fig_size=(8, 12))
    img_tmp_path = TMP_PATH / f'{uuid4()}.png'
    fig.savefig(img_tmp_path, dpi=100)
    uploaded_image, = await q.site.upload([img_tmp_path])
    os.remove(img_tmp_path)
    return AUDIO_IMG_TEMPLATE.format(
        header=header,
        uploaded_audio=CHUNK_URI.format(rec_id=rec_id, start=start),
        uploaded_image=uploaded_image)


async def show_recognize_dashboard(q: Q, audio_path, uploaded, example=True):
    filename = audio_path.split('/')[-1]
    y = read_audio_fast(audio_path)

    q.page['audio'] = ui.form_card(
        box='1 2 3 3',
        items=[
            ui.markup(AUDIO_TEMPLATE.format(header=filename, uploaded_audio=uploaded)),
            ui.button(name='new_recording', label='New recording', primary=True),
        ]
    )
    if example:
        # preds = get_model_predictions_for_clip(y, q.app.model)
        # preds.to_csv('./data/example_preds.csv', index=False)
        preds = pd.read_csv('./data/example_preds.csv')
    else:
        preds = get_model_predictions_for_clip(y, q.app.model)

    q.page['clipwise_predictions'] = ui.frame_card(
        box='1 5 3 7',
        title='Clipwise predictions',
        content=max_probability_bar_plot(preds, 3, 7)
    )

    q.page['framewise_predictions'] = ui.frame_card(
        box='4 8 20 4',
        title='Framewise predictions',
        content=framewise_probability_plot(preds, 20, 4)
    )

    if example:
        # fig = mel_spectrogram(y, fig_size=(26, 4.5))
        example_mel_path = './data/example_mel.png'
        # fig.savefig(example_mel_path, dpi=100)
        uploaded_image, = await q.site.upload([example_mel_path])
    else:
        fig = mel_spectrogram(y, fig_size=(26, 4.5))
        img_tmp_path = TMP_PATH / f'{uuid4()}.png'
        fig.savefig(img_tmp_path, dpi=100)
        uploaded_image, = await q.site.upload([img_tmp_path])
        os.remove(img_tmp_path)

    q.page['mel_spectrogram'] = ui.form_card(
        box='4 2 20 6',
        items=[
            ui.markup(IMAGE_TEMPLATE.format(uploaded_image=uploaded_image)),
        ]
    )


async def display_main_page(q):
    if not q.client.initialized:
        q.client.initialized = True
        q.app.model = load_pretrained_model()
        q.client.current_hash = 'annotate'
        q.client.spec_id = '0'
        q.client.method = 'random'
        q.client.tp_selector = 'TP'
        q.client.filename = f"annotations_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"

        q.client.annotations = get_annotations(q.client.filename)
        q.client.candidates = get_candidates()

        q.page['header'] = ui.header_card(
            box='1 1 4 1',
            title='Rainforest Connection Species Audio Detection',
            subtitle='Annotate bird and frog species in tropical soundscapes',
            icon='MusicInCollectionFill',
            icon_color='yellow',
        )
        q.page['tabs'] = ui.tab_card(
            box='5 1 3 1',
            items=[
                ui.tab(name='#annotate', label='Annotate'),
                ui.tab(name='#recognize', label='Recognize'),
                ui.tab(name='#references', label='References'),
            ]
        )
        q.client.rec_id, q.client.start, q.client.prob = get_next_candidate(
            q.client.candidates, q.client.annotations, q.client.spec_id, q.client.method, q.client.tp_selector)
        q.client.pos_rec_id, q.client.pos_start, q.client.pos_prob = get_random_positive_example(q.client.spec_id)
        q.client.neg_rec_id, q.client.neg_start, q.client.neg_prob = get_random_negative_example(q.client.spec_id)

    if q.args.true_button:
        update_annotations(q, 1)
        select_new_candidate(q)
    if q.args.false_button:
        update_annotations(q, 0)
        select_new_candidate(q)
    if q.args.na_button:
        update_annotations(q, -999)
        select_new_candidate(q)

    if q.args.refresh_button:
        refresh_original_examples(q)

    if q.args.new_recording:
        await del_cards(q, CARDS['recognize'])

    if q.args.spec_id and q.client.spec_id != q.args.spec_id:
        q.client.spec_id = q.args.spec_id
        select_new_candidate(q)
        refresh_original_examples(q)
    if q.args.method and q.client.method != q.args.method:
        q.client.method = q.args.method
        select_new_candidate(q)
    if q.args.tp_selector and q.client.tp_selector != q.args.tp_selector:
        q.client.tp_selector = q.args.tp_selector
        select_new_candidate(q)

    if q.args['#']:
        q.client.current_hash = q.args['#']

    if q.client.current_hash == 'references':
        await del_cards(q, CARDS['annotate'] + CARDS['recognize'] + ['progress'])
        q.page['references'] = ui.form_card(
            box='1 2 8 5',
            items=[ui.text(show_references())]
        )
    if q.client.current_hash == 'recognize':
        await del_cards(q, CARDS['annotate'] + CARDS['references'] + ['progress'])
        q.page['upload'] = ui.form_card(
            box='1 2 3 5', items=[
                ui.file_upload(name='audio_upload', label='Recognize!', multiple=False,
                               file_extensions=['mp3', 'wav', 'flac']),
                ui.button(name='example_audio', label='Use example audio', primary=True),
            ])

        if q.args.example_audio:
            await del_cards(q, ['upload'])
            await add_progress(q)
            audio_path = './data/f97ababc1.flac'
            uploaded, = await q.site.upload([audio_path])
            await show_recognize_dashboard(q, audio_path, uploaded, example=True)

        if q.args.audio_upload:
            await del_cards(q, ['upload'])
            await add_progress(q)
            uploaded = q.args.audio_upload[0]
            audio_path = await q.site.download(uploaded, './tmp/')
            await show_recognize_dashboard(q, audio_path, uploaded, example=False)

    if q.client.current_hash == 'annotate':
        await del_cards(q, CARDS['recognize'] + CARDS['references'] + ['progress'])
        q.page['species_selector'] = ui.form_card(
            box='5 2 1 1', items=[
                ui.dropdown(name='spec_id', tooltip='Species', value=q.client.spec_id, trigger=True,
                            choices=[ui.choice(name=f'{s}', label=f'{s}') for s in range(N_SPECIES)])
            ]
        )
        q.page['method_selector'] = ui.form_card(
            box='6 2 1 1', items=[
                ui.dropdown(name='method', tooltip='Order', value=q.client.method, trigger=True,
                            choices=[
                                ui.choice(name='top', label='Likely'),
                                ui.choice(name='random', label='Random'),
                                ui.choice(name='bottom', label='Unlikely'),
                            ])
            ]
        )
        q.page['tp_selector'] = ui.form_card(
            box='7 2 1 1', items=[
                ui.dropdown(name='tp_selector', tooltip='Predicted', value=q.client.tp_selector, trigger=True,
                            choices=[
                                ui.choice(name='TP', label='Positive'),
                                ui.choice(name='FP', label='Negative'),
                            ])
            ]
        )
        q.page['buttons'] = ui.form_card(
            box='1 2 4 1', items=[
                ui.buttons([
                    ui.button(name='true_button', label='True', primary=True),
                    ui.button(name='false_button', label='False'),
                    ui.button(name='na_button', label='Not sure...', ),
                    ui.button(name='refresh_button', label='Refresh confirmed examples', ),
                ])
            ]

        )

        new_html = await show_audio_chunk(
            q, q.client.rec_id, q.client.spec_id, q.client.start, q.client.prob,
            show_boxes=False, header='Example to check')
        q.page['new'] = ui.form_card(box='1 3 4 10', items=[ui.markup(new_html)])

        tp_html = await show_audio_chunk(
            q, q.client.pos_rec_id, q.client.spec_id, q.client.pos_start, q.client.pos_prob,
            show_boxes=True, header='Confirmed positive example')
        q.page['example_tp'] = ui.form_card(box='5 3 4 10', items=[ui.markup(tp_html)])

        fp_html = await show_audio_chunk(
            q, q.client.neg_rec_id, q.client.spec_id, q.client.neg_start, q.client.neg_prob,
            show_boxes=True, header='Confirmed negative example')
        q.page['example_fp'] = ui.form_card(box='9 3 4 10', items=[ui.markup(fp_html)])
