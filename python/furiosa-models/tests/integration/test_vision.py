import pytest

from furiosa.models.vision import (
    EfficientNetV2_S,
    MLCommonsResNet50,
    MLCommonsSSDMobileNet,
    MLCommonsSSDResNet34,
)
from furiosa.registry import Model

@pytest.mark.asyncio
async def test_resnet50_session_wrapper():
    import cv2
    import os 
    from furiosa.models import SessionWrapper
    from furiosa.common.thread import synchronous

    dir_path = os.path.dirname(os.path.realpath(__file__))
    model: Model = synchronous(MLCommonsResNet50)()
    with SessionWrapper(model) as sess:
        result = sess.inference( cv2.imread( os.path.join(dir_path, 'assets/EgyptianCat.jpg')) )
    assert result == "Egyptian cat"

#@pytest.mark.asyncio
#async def test_resnet50():
#    model: Model = await MLCommonsResNet50()
#    assert model.name == "MLCommonsResNet50"

#@pytest.mark.asyncio
#async def test_mobilenet():
#    model: Model = await MLCommonsSSDMobileNet()
#    assert model.name == "MLCommonsSSDMobileNet"

#@pytest.mark.asyncio
#async def test_resnet34():
#    model: Model = await MLCommonsSSDResNet34()
#    assert model.name == "MLCommonsSSDResNet34"

#@pytest.mark.asyncio
#async def test_efficientnet():
#    model: Model = await EfficientNetV2_S()
#    assert model.name == "EfficientNetV2_S"
