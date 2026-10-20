class SDE():
    """Diffusion parameterisation as in Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", 2022."""

    def __init__(self, type, sde_hp):
        self.type = type
        self.sde_hp = sde_hp
