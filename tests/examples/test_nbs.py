import os
import tempfile
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError


def run_notebook(notebook_path):
    """
    Run Jupyter Notebook of specified path and
    raises exception if any cell fails to execute.
    Output is written to a temporary file; the source notebook is never modified.
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
        suffix = "_" + os.path.basename(notebook_path)
        with tempfile.NamedTemporaryFile(
            mode="wt", suffix=suffix, dir=tempfile.gettempdir(), delete=False
        ) as f:
            nbformat.write(nb, f)


def get_all_notebooks():
    examples = os.path.join(os.getcwd(), "examples")
    nbs = [
        os.path.join(examples, f)
        for f in sorted(os.listdir(examples))
        if f.endswith(".ipynb")
    ]
    return nbs


@pytest.mark.examples
@pytest.mark.parametrize(
    "notebook_path", get_all_notebooks(), ids=lambda p: os.path.basename(p)
)
def test_notebook(notebook_path):
    try:
        run_notebook(notebook_path)
    except Exception as e:
        assert False, f"Error in {notebook_path}: {e}"
