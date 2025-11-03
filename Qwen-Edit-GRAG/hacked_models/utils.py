import math
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Literal
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path
from datetime import datetime
from base64 import b64encode
from PIL import Image
import random
import io
from io import BytesIO

def seed_everything(seed: int = 42, deterministic: bool = False):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"✅ Random seed set to {seed}, deterministic={deterministic}")