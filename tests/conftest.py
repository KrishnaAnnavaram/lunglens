import pytest

from lunglens.data.synthetic import make_cohort


@pytest.fixture(scope="session")
def cohort(tmp_path_factory):
    """A small synthetic cohort, written once per test session."""
    return make_cohort(tmp_path_factory.mktemp("cohort"), n_patients=48, seed=5)


@pytest.fixture(scope="session")
def trained(cohort, tmp_path_factory):
    from lunglens.pipeline import train_and_evaluate

    out = tmp_path_factory.mktemp("models")
    results = train_and_evaluate(cohort, ["prior", "size", "radiomics"], out, seed=5, n_boot=100)
    return out, {r.kind: r for r in results}
