from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class FlairOneDataset(CustomDataset):
    """FlairOne dataset.
    """
    CLASSES = ('Building', 'Pervious surface', 'Impervious surface', 'Bare soil', 'Water', 'Coniferous',
               'Deciduous', 'Brushwood', 'Vineyard', 'Herbaceous vegetation', 'Agricultural land', 'Plowed land', 'other')

    PALETTE = [[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7],[8,8,8],[9,9,9],[10,10,10],[11,11,11],[12,12,12]]

    def __init__(self, **kwargs):
        super(FlairOneDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            **kwargs)
