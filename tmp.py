import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-parts-zip", default="photos_no_part.zip", required=False, help="path to no parts zip file -- zip containing images without any parts"
)
parser.add_argument(
    "--parts-zip", default="photos.zip", required=False, help="path to no parts images zip"
)
parser.add_argument(
    "--threshold", type=float, default=0.5, required=False, help="treshold detection"
)
print(parser.parse_args().__dict__)