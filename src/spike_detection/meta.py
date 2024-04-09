from pathlib import Path

GAIN_TO_UV = SAMP_FREQ = None

# region Neuropixels mice
_SI_MOUSE_PATHS = [
    Path("/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/probe_773592315"),
    Path("/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/probe_773592318"),
    Path("/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/probe_773592320"),
    Path("/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/probe_773592324"),
    Path("/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/probe_773592328"),
    Path("/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/probe_773592330"),
]
_SAMP_FREQ_NEUROPIXELS = 30
_GAIN_TO_UV_NEUROPIXELS = 0.195

class _SiMouse:
    def __getitem__(self, idx):
        global SAMP_FREQ, GAIN_TO_UV
        SAMP_FREQ = _SAMP_FREQ_NEUROPIXELS
        GAIN_TO_UV = _GAIN_TO_UV_NEUROPIXELS
        
        return _SI_MOUSE_PATHS[idx]
SI_MOUSE = _SiMouse()
SI_MODELS = [
    "/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/dl_models/240318/a/240318_161415_981130",
    "/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/dl_models/240318/b/240318_163253_679441",
    "/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/dl_models/240318/c/240318_165245_967091",
    "/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/dl_models/240318/d/240318_172719_805804",
    "/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/dl_models/240318/e/240318_174428_896437",
    "/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/dl_models/240318/f/240318_180745_727120",
]
# endregion

# region MEA organoids
_ORGANOID_PATHS = [
    Path("/data/MEAprojects/DLSpikeSorter/data/2950"),
    Path("/data/MEAprojects/DLSpikeSorter/data/2953"),
    Path("/data/MEAprojects/DLSpikeSorter/data/2954"),
    Path("/data/MEAprojects/DLSpikeSorter/data/2957"),
    Path("/data/MEAprojects/DLSpikeSorter/data/5116"),
    Path("/data/MEAprojects/DLSpikeSorter/data/5118"),
]
_SAMP_FREQ_MEA = 20
_GAIN_TO_UV_MEA = 6.29425 

class _Organoid():
    def __getitem__(self, idx):
        global SAMP_FREQ, GAIN_TO_UV
        SAMP_FREQ = _SAMP_FREQ_MEA
        GAIN_TO_UV = _GAIN_TO_UV_MEA
        
        return _ORGANOID_PATHS[idx]
ORGANOID = _Organoid()
ORGANOID_MODELS = [
    "/data/MEAprojects/DLSpikeSorter/models/v0_4_4/2950/230101_133131_959516",
    "/data/MEAprojects/DLSpikeSorter/models/v0_4_4/2953/230101_133514_582221",
    "/data/MEAprojects/DLSpikeSorter/models/v0_4_4/2954/230101_134042_729459",
    "/data/MEAprojects/DLSpikeSorter/models/v0_4_4/2957/230101_134408_403069",
    "/data/MEAprojects/DLSpikeSorter/models/v0_4_4/5116/230101_134927_487762",
    "/data/MEAprojects/DLSpikeSorter/models/v0_4_4/5118/230101_135307_305876",
    "/data/MEAprojects/DLSpikeSorter/models/v0_4_4/5118/230101_135307_305876",
]

# endregion
    


