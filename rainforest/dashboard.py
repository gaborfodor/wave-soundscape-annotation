from datetime import datetime

from h2o_wave import ui, Q, app, main

from rainforest.config import N_SPECIES
from rainforest.utils import show_references, get_candidates, get_annotations, visualize_spectograms, \
    get_next_candidate, fig_to_img, get_random_positive_example, get_random_negative_example, ANNOTATION_PATH

_ = main

CARDS = {
    'references': ['references'],
    'annotate': ['species_selector', 'method_selector', 'tp_selector', 'buttons', 'new', 'example_tp', 'example_fp'],
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


async def display_main_page(q):
    if not q.client.initialized:
        q.client.initialized = True
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
        await del_cards(q, CARDS['annotate'])
        q.page['references'] = ui.form_card(
            box='1 2 8 5',
            items=[ui.text(show_references())]
        )
    if q.client.current_hash == 'annotate':
        await del_cards(q, CARDS['references'])
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

        q.page['new'] = ui.image_card(
            box=f'1 3 4 10',
            title='Example to check',
            type='png',
            image=fig_to_img(visualize_spectograms(q.client.rec_id, q.client.spec_id, q.client.start,
                                                   q.client.prob, show_boxes=False)))

        q.page['example_tp'] = ui.image_card(
            box=f'5 3 4 10',
            title='Confirmed positive example',
            type='png',
            image=fig_to_img(visualize_spectograms(q.client.pos_rec_id, q.client.spec_id, q.client.pos_start,
                                                   q.client.pos_prob)))

        q.page['example_fp'] = ui.image_card(
            box=f'9 3 4 10',
            title='Confirmed negative example',
            type='png',
            image=fig_to_img(visualize_spectograms(q.client.neg_rec_id, q.client.spec_id, q.client.neg_start,
                                                   q.client.neg_prob)))
