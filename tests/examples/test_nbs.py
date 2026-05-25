import os
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError


def run_notebook(notebook_path):
    """
    Run Jupyter Notebook of specified path and
    raises exception if any cell fails to execute.
    """
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_path)}})
    except CellExecutionError:
        msg = 'Error executing the notebook "%s".\n\n' % notebook_path
        msg += 'See notebook "%s" for the traceback.' % notebook_path
        print(msg)
        raise
    finally:
        with open(notebook_path, "wt") as f:
            nbformat.write(nb, f)


def get_all_notebooks():
    examples = os.path.join(os.getcwd(), "examples")
    nbs = [
        os.path.join(examples, f) for f in os.listdir(examples) if f.endswith(".ipynb")
    ]

    return nbs


@pytest.mark.examples
def test_htg_recurrent_demo():
    recurrent = os.path.join(os.getcwd(), "examples", "HTG_recurrent.ipynb")

    try:
        run_notebook(recurrent)
    except Exception as e:
        assert False, f"Error in {recurrent}: {e}"


@pytest.mark.examples
def test_htg_algebra_demo():
    algebra = os.path.join(os.getcwd(), "examples", "HTG_algebra.ipynb")

    try:
        run_notebook(algebra)
    except Exception as e:
        assert False, f"Error in {algebra}: {e}"


@pytest.mark.examples
def test_htg_overview_demo():
    overview = os.path.join(os.getcwd(), "examples", "HTG_overview.ipynb")

    try:
        run_notebook(overview)
    except Exception as e:
        assert False, f"Error in {overview}: {e}"


@pytest.mark.examples
def test_metrics_demo():
    metrics = os.path.join(os.getcwd(), "examples", "metrics_demo.ipynb")

    try:
        run_notebook(metrics)
    except Exception as e:
        assert False, f"Error in {metrics}: {e}"


@pytest.mark.examples
def test_mnist_demo():
    mnist = os.path.join(os.getcwd(), "examples", "mnist_demo.ipynb")

    try:
        run_notebook(mnist)
    except Exception as e:
        assert False, f"Error in {mnist}: {e}"


@pytest.mark.examples
def test_dataloader_demo():
    dataloader = os.path.join(os.getcwd(), "examples", "data_loader_demo.ipynb")

    try:
        run_notebook(dataloader)
    except Exception as e:
        assert False, f"Error in {dataloader}: {e}"


@pytest.mark.examples
def test_soil_moisture_demo():
    soil_moisture = os.path.join(os.getcwd(), "examples", "soil_moisture_demo.ipynb")

    try:
        run_notebook(soil_moisture)
    except Exception as e:
        assert False, f"Error in {soil_moisture}: {e}"


@pytest.mark.examples
def test_batcher_demo():
    batcher = os.path.join(os.getcwd(), "examples", "batcher_demo.ipynb")

    try:
        run_notebook(batcher)
    except Exception as e:
        assert False, f"Error in {batcher}: {e}"


@pytest.mark.examples
def test_archiver_demo():
    archiver = os.path.join(os.getcwd(), "examples", "archiver_demo.ipynb")

    try:
        run_notebook(archiver)
    except Exception as e:
        assert False, f"Error in {archiver}: {e}"
