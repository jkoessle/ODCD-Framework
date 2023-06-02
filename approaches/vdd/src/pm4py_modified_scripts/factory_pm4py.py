import os
import shutil
import subprocess
import sys
import pm4py_modified_scripts.pydotplus_pm4py as pydotplus

PYDOTPLUS = "pydotplus"

VERSIONS = {PYDOTPLUS: pydotplus.apply}
VERSIONS1 = {PYDOTPLUS: pydotplus.apply_with_constraints}

def apply(heu_net, parameters=None, variant=PYDOTPLUS):
    """
    Gets a representation of an Heuristics Net

    Parameters
    -------------
    heu_net
        Heuristics net
    parameters
        Possible parameters of the algorithm, including: format

    Returns
    ------------
    gviz
        Representation of the Heuristics Net
    """
    print ('in factory')
    return VERSIONS[variant](heu_net, parameters=parameters)


def apply_with_constraints(heu_net, parameters=None, constr = None,  variant=PYDOTPLUS):
    """
    Gets a representation of an Heuristics Net

    Parameters
    -------------
    heu_net
        Heuristics net
    parameters
        Possible parameters of the algorithm, including: format

    Returns
    ------------
    gviz
        Representation of the Heuristics Net
    """
    print ('in factory')
    return VERSIONS1[variant](heu_net, constr = constr, parameters=parameters)



def view(figure):
    """
    View on the screen a figure that has been rendered

    Parameters
    ----------
    figure
        figure
    """
    try:
        filename = figure.name
        figure = filename
    except AttributeError:
        # continue without problems, a proper path has been provided
        pass

    is_ipynb = False

    try:
        get_ipython()
        is_ipynb = True
    except NameError:
        pass

    if is_ipynb:
        from IPython.display import Image
        image = Image(open(figure, "rb").read())
        from IPython.display import display
        return display(image)
    else:
        if sys.platform.startswith('darwin'):
            subprocess.call(('open', figure))
        elif os.name == 'nt':  # For Windows
            os.startfile(figure)
        elif os.name == 'posix':  # For Linux, Mac, etc.
            subprocess.call(('xdg-open', figure))


def save(figure, output_file_path):
    """
    Save a figure that has been rendered

    Parameters
    -----------
    figure
        figure
    output_file_path
        Path where the figure should be saved
    """
    try:
        filename = figure.name
        figure = filename
    except AttributeError:
        # continue without problems, a proper path has been provided
        pass

    shutil.copyfile(figure, output_file_path)
