def test_importing_app():
    # this will raise an exception if pydantic model validation fails for th app
    from nomad_nmr_schema.apps import nmr_app

    assert nmr_app.app.label == 'NMR App'
