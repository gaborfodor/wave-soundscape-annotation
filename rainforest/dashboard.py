from h2o_wave import ui, Q, app, main

from rainforest.config import N_SPECIES
from rainforest.utils import show_references

_ = main

CARDS = {
    'references': ['references'],
    'annotate': ['species_selector', 'method_selector'],
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
        q.user.spec_id = 's0'
        q.user.method = 'top'

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

    if q.args.spec_id:
        q.client.spec_id = q.args.spec_id
    if q.args.method:
        q.client.method = q.args.method

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
                ui.dropdown(name='spec_id', label='', value=q.user.spec_id, required=True, trigger=True,
                            choices=[ui.choice(name=f's{s}', label=f'{s}') for s in range(N_SPECIES)])
            ]
        )
        q.page['method_selector'] = ui.form_card(
            box='8 1 1 1', items=[
                ui.dropdown(name='method', label='', value=q.user.method, required=True, trigger=True,
                            choices=[
                                ui.choice(name='top', label='Likely'),
                                ui.choice(name='random', label='Random'),
                                ui.choice(name='bottom', label='Unlikely'),
                            ])
            ]
        )
