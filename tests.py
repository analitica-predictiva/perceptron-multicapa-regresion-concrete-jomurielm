"""
Calificación del laboratorio
-----------------------------------------------------------------------------------------
"""

import sys

import preguntas


def test_01():
    """
    ---< Input/Output test case >----------------------------------------------------
    Pregunta 01
    pip3 install scikit-learn pandas numpy
    python3 tests.py 01
    """

    x_data, y_data = preguntas.pregunta_01()

    assert x_data.shape == (1030, 8)
    assert y_data.shape == (1030,)
    assert "strength" not in x_data.columns


def test_02():
    """
    ---< Input/Output test case >----------------------------------------------------
    Pregunta 02
    pip3 install scikit-learn pandas numpy
    python3 tests.py 02
    """

    x_train, x_test, y_train, y_test = preguntas.pregunta_02()
    assert x_train.shape == (772, 8)
    assert x_test.shape == (258, 8)
    assert y_train.sum().round(2) == 27681.52
    assert y_test.sum().round(2) == 9210.98


def test_03():
    """
    ---< Run command >-----------------------------------------------------------------
    Pregunta 03
    pip3 install scikit-learn pandas numpy
    python3 tests.py 03
    """

    pipeline = preguntas.pregunta_03()
    assert pipeline.steps[0][0] == "minmaxscaler"
    assert pipeline.steps[0][1].__class__.__name__ == "MinMaxScaler"
    assert pipeline.steps[1][0] == "mlpregressor"
    assert pipeline.steps[1][1].__class__.__name__ == "MLPRegressor"


def test_04():
    """
    ---< Run command >--------------------------------------------------------------------
    Pregunta 04
    pip3 install scikit-learn pandas numpy
    python3 tests.py 04
    """

    gridsearchcv = preguntas.pregunta_04()

    assert gridsearchcv.__class__.__name__ == "GridSearchCV"
    assert gridsearchcv.cv == 5
    assert gridsearchcv.scoring == "r2"
    assert gridsearchcv.return_train_score is False


def test_05():
    """
    ---< Run command >--------------------------------------------------------------------
    Pregunta 05
    pip3 install scikit-learn pandas numpy
    python3 tests.py 05
    """

    mse_train, mse_test = preguntas.pregunta_05()

    print(mse_train)
    print(mse_test)

    assert mse_train < 114.0
    assert mse_test < 120.0


test = {
    "01": test_01,
    "02": test_02,
    "03": test_03,
    "04": test_04,
    "05": test_05,
}[sys.argv[1]]

test()
