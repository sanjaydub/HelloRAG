print("Hello, World!")
def test_langchain_installed():
    try:
        import langchain
    except ImportError:
        assert False, "langchain is not installed"
    assert hasattr(langchain, "__version__"), "langchain does not have a __version__ attribute"

def test_langchain_version():
    import langchain
    version = langchain.__version__
    print(f"langchain version: {version}")
    # Split version and check major.minor >= 0.3
    parts = version.split('.')
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    assert (major > 0) or (major == 0 and minor >= 3), f"langchain version must be >= 0.3, found {version}"
