import numpy as np

# default split configuration for the datamodule
SPLITS_DICT = {
    "train": {
        "years": np.arange(2004, 2025).tolist(),
        "months": np.arange(1, 13).tolist(),
        "days": np.arange(2, 23).tolist(),
    },
    "val": {
        "years": np.arange(2004, 2025).tolist(),
        "months": np.arange(1, 13).tolist(),
        "days": np.arange(24, 27).tolist(),
    },
    "test": {
        "years": np.arange(2004, 2025).tolist(),
        "months": np.arange(1, 13).tolist(),
        "days": np.arange(28, 32).tolist(),
    },
}


# MSG wavelengths in nanometers
MSG_WAVELENGTHS = {
    "IR_016": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 1578.4,
        "center_wavelength": 1640.0,
        "max_wavelength": 1696.0,
    },  # 1.64
    "IR_039": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 3638.4,
        "center_wavelength": 3920.0,
        "max_wavelength": 4201.6,
    },  # 3.92,
    "IR_087": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 8540.0,
        "center_wavelength": 8700.0,
        "max_wavelength": 8892.0,
    },  # 8.70,
    "IR_097": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 9548.0,
        "center_wavelength": 9660.0,
        "max_wavelength": 9783.2,
    },  # 9.66,
    "IR_108": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 10280.0,
        "center_wavelength": 10800.0,
        "max_wavelength": 11280.0,
    },
    "IR_120": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 11520.0,
        "center_wavelength": 12000.0,
        "max_wavelength": 12440.0,
    },
    "IR_134": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 12680.0,
        "center_wavelength": 13400.0,
        "max_wavelength": 14000.0,
    },
    "VIS006": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 602.0,
        "center_wavelength": 640.0,
        "max_wavelength": 677.0,
    },
    "VIS008": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 782.0,
        "center_wavelength": 810.0,
        "max_wavelength": 838.0,
    },
    "WV_062": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 5854.0,
        "center_wavelength": 6250.0,
        "max_wavelength": 6718.0,
    },
    "WV_073": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 7150.0,
        "center_wavelength": 7350.0,
        "max_wavelength": 7590.0,
    },
}


# GOES wavelengths in nanometers
GOES_WAVELENGTHS = {
    "CMI_C01": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 450.5,
        "center_wavelength": 470.0,
        "max_wavelength": 490.6,
    },  # 0.47,
    "CMI_C02": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 596.3,
        "center_wavelength": 640.0,
        "max_wavelength": 682.1,
    },  # 0.64,
    "CMI_C03": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 846.9,
        "center_wavelength": 870.0,
        "max_wavelength": 882.0,
    },  # 0.87,
    "CMI_C04": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 1366.3,
        "center_wavelength": 1380.0,
        "max_wavelength": 1380.3,
    },  # 1.38,
    "CMI_C05": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 1587.6,
        "center_wavelength": 1610.0,
        "max_wavelength": 1632.4,
    },  # 1.61,
    "CMI_C06": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 2220.2,
        "center_wavelength": 2250.0,
        "max_wavelength": 2265.5,
    },  # 2.25,
    "CMI_C07": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 3802.7,
        "center_wavelength": 3890.0,
        "max_wavelength": 3992.2,
    },  # 3.89,
    "CMI_C08": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 5790.4,
        "center_wavelength": 6170.0,
        "max_wavelength": 6590.7,
    },  # 6.17,
    "CMI_C09": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 6725.0,
        "center_wavelength": 6930.0,
        "max_wavelength": 7142.9,
    },  # 6.93,
    "CMI_C10": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 7242.7,
        "center_wavelength": 7340.0,
        "max_wavelength": 7431.1,
    },  # 7.34,
    "CMI_C11": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 8226.4,
        "center_wavelength": 8440.0,
        "max_wavelength": 8663.3,
    },  # 8.44,
    "CMI_C12": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 9423.3,
        "center_wavelength": 9610.0,
        "max_wavelength": 9800.1,
    },  # 9.61,
    "CMI_C13": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 10177.1,
        "center_wavelength": 10330.0,
        "max_wavelength": 10481.1,
    },  # 10.33,
    "CMI_C14": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 10815.5,
        "center_wavelength": 11190.0,
        "max_wavelength": 11603.6,
    },  # 11.19,
    "CMI_C15": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 11825.9,
        "center_wavelength": 12270.0,
        "max_wavelength": 12747.0,
    },  # 12.27,
    "CMI_C16": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 12990.4,
        "center_wavelength": 13270.0,
        "max_wavelength": 13559.3,
    },  # 13.27,
}


# HIMAWARI wavelengths in nanometers
HIMAWARI_WAVELENGTHS = {
    "B01": {
        "reso_og": 1000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 450.0,
        "center_wavelength": 470.0,
        "max_wavelength": 490.7,
    },
    "B02": {
        "reso_og": 500,
        "band_type": "TOA Reflectance",
        "min_wavelength": 495.1,
        "center_wavelength": 510.0,
        "max_wavelength": 525.9,
    },
    "B03": {
        "reso_og": 1000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 599.1,
        "center_wavelength": 640.0,
        "max_wavelength": 680.6,
    },
    "B04": {
        "reso_og": 2000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 839.1,
        "center_wavelength": 860.0,
        "max_wavelength": 873.5,
    },
    "B05": {
        "reso_og": 2000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 1589.6,
        "center_wavelength": 1600.0,
        "max_wavelength": 1630.3,
    },
    "B06": {
        "reso_og": 2000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 2235.1,
        "center_wavelength": 2300.0,
        "max_wavelength": 2278.9,
    },
    "B07": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 3784.6,
        "center_wavelength": 3900.0,
        "max_wavelength": 3985.0,
    },
    "B08": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 5827.5,
        "center_wavelength": 6200.0,
        "max_wavelength": 6648.9,
    },
    "B09": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 6739.0,
        "center_wavelength": 6900.0,
        "max_wavelength": 7140.3,
    },
    "B10": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 7253.7,
        "center_wavelength": 7300.0,
        "max_wavelength": 7440.5,
    },
    "B11": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 8404.8,
        "center_wavelength": 8600.0,
        "max_wavelength": 8776.6,
    },
    "B12": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 9446.4,
        "center_wavelength": 9600.0,
        "max_wavelength": 9823.2,
    },
    "B13": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 10193.7,
        "center_wavelength": 10400.0,
        "max_wavelength": 10612.3,
    },
    "B14": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 10909.9,
        "center_wavelength": 11200.0,
        "max_wavelength": 11576.8,
    },
    "B15": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 11900.5,
        "center_wavelength": 12400.0,
        "max_wavelength": 12865.0,
    },
    "B16": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 13003.9,
        "center_wavelength": 13300.0,
        "max_wavelength": 13564.8,
    },
}

CLOUDSAT_NAN_FILL_VALUES = {
    "Radar_Reflectivity": -35,
    "CloudTypeMask": 0,
    "CPR_Cloud_mask": 0,
    "RO_liq_effective_radius": 0,
    "RO_ice_effective_radius": 0,
    "RO_liq_number_conc": 0,
    "RO_ice_number_conc": 0,
    "RO_liq_water_content": 0,
    "RO_ice_water_content": 0,
    "re": 0,
    "IWC": 0,
    "QR": 0,
}
