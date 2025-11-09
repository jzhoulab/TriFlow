from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()

    # Interpolant configuration
    config.interpolant = ConfigDict()
    config.interpolant.min_t = 1e-2
    
    # Amino acid types settings within interpolant
    config.interpolant.aatypes = ConfigDict()
    config.interpolant.aatypes.corrupt = True
    # config.interpolant.aatypes.schedule = 'linear'
    # config.interpolant.aatypes.schedule_exp_rate = 10
    # config.interpolant.aatypes.temp = 0.1
    config.interpolant.aatypes.noise = 0
    config.interpolant.aatypes.do_purity = False
    config.interpolant.aatypes.train_extra_mask = 0.0
    config.interpolant.aatypes.interpolant_type = 'masking'

    # Sampling settings within interpolant
    config.interpolant.sampling = ConfigDict()
    config.interpolant.sampling.num_timesteps = 10
    config.interpolant.sampling.do_sde = False
    


    return config




