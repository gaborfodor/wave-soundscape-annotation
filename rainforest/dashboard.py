from h2o_wave import ui, Q, app, main

from rainforest.config import N_SPECIES
from rainforest.utils import show_references, get_candidates, get_annotations, visualize_spectograms, \
    get_next_candidate, fig_to_img, get_random_positive_example, get_random_negative_example

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


async def display_main_page(q):
    if not q.client.initialized:
        q.client.initialized = True
        q.client.spec_id = '0'
        q.client.method = 'top'
        q.client.tp_selector = 'TP'

        q.client.annotations = get_annotations()
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
            q.client.candidates, q.client.spec_id, q.client.method, q.client.tp_selector)

        q.client.pos_rec_id, q.client.pos_start, q.client.pos_prob = get_random_positive_example(q.client.spec_id)
        q.client.neg_rec_id, q.client.neg_start, q.client.neg_prob = get_random_negative_example(q.client.spec_id)

    if q.args.spec_id:
        q.client.spec_id = q.args.spec_id
    if q.args.method:
        q.client.method = q.args.method
    if q.args.tp_selector:
        q.client.tp_selector = q.args.tp_selector

    if q.args.true_button:
        print(q.client.rec_id, q.client.start, q.client.prob, q.client.spec_id, 'True')
    if q.args.false_button:
        print(q.client.rec_id, q.client.start, q.client.prob, q.client.spec_id, 'False')
    if q.args.na_button:
        print(q.client.rec_id, q.client.start, q.client.prob, q.client.spec_id, 'NA')
    if q.args.refresh_button:
        q.client.pos_rec_id, q.client.pos_start, q.client.pos_prob = get_random_positive_example(q.client.spec_id)
        q.client.neg_rec_id, q.client.neg_start, q.client.neg_prob = get_random_negative_example(q.client.spec_id)

    if q.args['#']:
        print(q.client.current_hash, '->', q.args['#'])
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
            box='7 1 1 1', items=[
                ui.dropdown(name='spec_id', label='', value=q.client.spec_id, required=True, trigger=True,
                            choices=[ui.choice(name=f'{s}', label=f'{s}') for s in range(N_SPECIES)])
            ]
        )
        q.page['method_selector'] = ui.form_card(
            box='8 1 1 1', items=[
                ui.dropdown(name='method', label='', value=q.client.method, required=True, trigger=True,
                            choices=[
                                ui.choice(name='top', label='Likely'),
                                ui.choice(name='random', label='Random'),
                                ui.choice(name='bottom', label='Unlikely'),
                            ])
            ]
        )
        q.page['tp_selector'] = ui.form_card(
            box='9 1 1 1', items=[
                ui.dropdown(name='tp_selector', label='', value=q.client.tp_selector, required=True, trigger=True,
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
                                                   q.client.prob)))

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
