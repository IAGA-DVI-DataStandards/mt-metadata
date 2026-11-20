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
        ma_order=1,
    )

    assert prewhitening.type == PrewhiteningTypeEnum.arma
    assert prewhitening.ar_order == 3
    assert prewhitening.ma_order == 1


@pytest.mark.parametrize("field_name", ["ar_order", "ma_order"])
def test_prewhitening_rejects_negative_orders(field_name):
    with pytest.raises(ValidationError):
        Prewhitening(**{field_name: -1})
