import pytest
from pydantic import ValidationError

from mt_metadata.processing.prewhitening import Prewhitening, PrewhiteningTypeEnum


def test_prewhitening_defaults():
    prewhitening = Prewhitening()

    assert prewhitening.type == PrewhiteningTypeEnum.first_difference
    assert prewhitening.ar_order == 0
    assert prewhitening.ma_order == 0


def test_prewhitening_accepts_arma_orders():
    prewhitening = Prewhitening(
        type=PrewhiteningTypeEnum.arma,
        ar_order=3,
        ma_order=0,
    )

    assert prewhitening.type == PrewhiteningTypeEnum.arma
    assert prewhitening.ar_order == 3
    assert prewhitening.ma_order == 0


@pytest.mark.parametrize("field_name", ["ar_order", "ma_order"])
def test_prewhitening_rejects_negative_orders(field_name):
    with pytest.raises(ValidationError):
        Prewhitening(**{field_name: -1})


def test_prewhitening_rejects_arma_without_ar_order():
    with pytest.raises(ValidationError, match="ar_order"):
        Prewhitening(type=PrewhiteningTypeEnum.arma)


def test_prewhitening_rejects_unsupported_arma_ma_order():
    with pytest.raises(NotImplementedError, match="ma_order"):
        Prewhitening(type=PrewhiteningTypeEnum.arma, ar_order=1, ma_order=1)
