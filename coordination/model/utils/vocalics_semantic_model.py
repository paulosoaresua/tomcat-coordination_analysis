from __future__ import annotations


class VocalicsSemanticsModelParameters:

    def __init__(self):
        self.sd_uc = None
        self.sd_c = None
        self.sd_vocalics = None
        self.sd_obs_vocalics = None

    def reset(self):
        self.sd_uc = None
        self.sd_c = None
        self.sd_vocalics = None
        self.sd_obs_vocalics = None
