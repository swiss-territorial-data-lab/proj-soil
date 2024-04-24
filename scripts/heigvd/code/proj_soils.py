from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class ProjSoilsDataset(CustomDataset):
    """
    Proj Soils dataset.
    """

    CLASSES = (
        "background", # 0
        "batiment", # 1
        # "toit_vegetalise",
        "surface_non_beton", # 2
        "surface_beton", # 3
        # "eau_bassin",
        "roche_dure_meuble", # 4
        "eau_naturelle", # 5
        "roseliere", # 6
        "sol_neige", # 7
        "sol_vegetalise", # 8
        # "surface_riparienne",
        "sol_divers", # 9
        "sol_vigne", # 10
        "sol_agricole", # 11
        "sol_bache", # 12
        # "sol_serre_temporaire",
        # "serre_permanente"
    )

    PALETTE = [[i, i, i] for i in range(len(CLASSES))]

    def __init__(self, **kwargs):
        super(ProjSoilsDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            **kwargs)
