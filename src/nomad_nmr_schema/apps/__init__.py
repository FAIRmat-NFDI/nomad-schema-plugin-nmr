from nomad.config.models.plugins import AppEntryPoint
from nomad.config.models.ui import App, Column, Columns, FilterMenu, FilterMenus

nmr_app = AppEntryPoint(
    name='NmrApp',
    description='NMR app entry point configuration.',
    app=App(
        label='NMR App',
        path='app',
        category='simulation',
        columns=Columns(
            selected=['entry_id'],
            options={
                'entry_id': Column(),
            },
        ),
        filter_menus=FilterMenus(
            options={
                'material': FilterMenu(label='Material'),
            }
        ),
    ),
)
