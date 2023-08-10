from .detector3d_template import Detector3DTemplate
from .pointpillar import PointPillar
from .second_net import SECONDNet
from .centerpoint import CenterPoint
from .ssd3d import SSD3D
from .dense_sphere import DenseSphere


__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PointPillar': PointPillar,
    'CenterPoint': CenterPoint,
    'DenseSphere': DenseSphere,
    'SSD3D': SSD3D
}


def build_detector(model_cfg, num_class, dataset, logger):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset, logger=logger
    )

    return model
