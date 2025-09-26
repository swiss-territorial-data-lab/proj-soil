from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class ProjSoilsDataset(CustomDataset):
    """
    Proj Soils dataset.
    """

    CLASSES = (
        "background", ## 0
        "batiment", #1,
        "surface_non_beton", #2,
        "surface_beton", #3,
        "roche_dure_meuble", #4,
        "eau_naturelle", #5,
        "roseliere", #6,
        "neige", #7,
        "sol_vegetalise", #8,
        "sol_vigne_verger", #9,
        "sol_agricole", #10,
        "sol_bache", #11,
        "sol_serre_temporaire", #12,
        "serre_permanente", #13,
        "toiture_vegetalisee", #14,
        "eau_bassin", #15,
        "gazon_synthetique", #16,
        # "avant_toit", #17   
    )

    PALETTE = [[i, i, i] for i in range(len(CLASSES))]

    def __init__(self, **kwargs):
        super(ProjSoilsDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            **kwargs)
