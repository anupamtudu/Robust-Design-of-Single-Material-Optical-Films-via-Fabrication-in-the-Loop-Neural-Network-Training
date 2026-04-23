from chapter_3f2 import *

# 1. Define the config that was used to create the model
cfg_to_load = DesignConfig(num_layers=100)

# 2. Instantiate the model class *first*
model = OnlineOptimizer(
    seed_size=cfg_to_load.num_layers, 
    num_layers=cfg_to_load.num_layers,
    n_min=cfg_to_load.n_min, 
    n_max=cfg_to_load.n_max, 
    d_min_nm=cfg_to_load.d_min_nm, 
    d_max_nm=cfg_to_load.d_max_nm
).to(DEVICE)

# 3. Define the path to your saved file
model_path = os.path.join(OUTPUT_DIR, "model_100L_linear.pth")

# 4. Load the state dictionary into the model
model.load_state_dict(torch.load(model_path))

# 5. Set the model to evaluation mode
model.eval()