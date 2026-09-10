from importlib.metadata import PackageNotFoundError, version

from asd_gen_gap.paper.reproducibility import generate_reproducibility_statement


def test_reproducibility_statement_reports_packages_without_paths() -> None:
    text = generate_reproducibility_statement()
    try:
        version("torch")
    except PackageNotFoundError:
        pass
    else:
        assert "- torch:" in text
    assert "data/ABIDE_I" not in text
