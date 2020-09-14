from utils import *

device = torch.device('cuda')
warnings.simplefilter("ignore")

""" Load Files """

path = 'Dataset/Latent'

for f in glob.glob(os.path.join(path, '*.npy')):
    print(f)