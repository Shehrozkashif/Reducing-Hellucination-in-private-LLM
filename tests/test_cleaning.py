from app.scripts.clean_sop import clean_text

def test_clean_text():
    assert clean_text("Hello\nWorld!") == "Hello World!"

