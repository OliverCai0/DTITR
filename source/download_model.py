from argparse import ArgumentParser
from huggingface_hub import hf_hub_download

parser = ArgumentParser()
parser.add_argument(
    '-f',
    type=str,
    help='Name of the Model'
)
args = parser.parse_args()

model_path = hf_hub_download(
    repo_id="DLSAutumn2023/DTITR_Recreation",
    filename="NOFCNN100EPOCHS",
    local_dir='.'
)