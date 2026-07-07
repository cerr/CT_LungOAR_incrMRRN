import sys
from model_wrapper import run_inference_nii

if __name__ == '__main__':
    run_inference_nii.main(sys.argv[1], sys.argv[2])
