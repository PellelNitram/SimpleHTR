import argparse
import json
from typing import Tuple, List

import cv2
import editdistance
from path import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor


class FilePaths:
    """Filenames and paths to data."""
    #fn_char_list = '../model/charList.txt'
    #fn_summary = '../model/summary.json'
    #fn_corpus = '../data/corpus.txt'
    fn_char_list = '/home/martin/data/xournalpp_ml/simplehtr/0_trained_model/word_model/charList.txt'
    fn_summary = '/home/martin/data/xournalpp_ml/simplehtr/0_trained_model/word_model/summary.json'
    fn_corpus = '../data/corpus.txt'


def get_img_height() -> int:
    """Fixed height for NN."""
    return 32


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())


def infer(model: Model, fn_img: Path, outfile: Path) -> None:
    """Recognizes text in image provided by file path."""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)

    with open(outfile, 'w') as f:
        json.dump({
            'recognised': recognized[0],
            'probability': float( probability[0] ),
        }, f, indent=4)

def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/word.png')
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')
    parser.add_argument('--model-dir', help='Path to model folder.', type=Path, required=True)
    parser.add_argument('--outfile', help='Path to store prediction to.', type=Path, required=True)

    return parser.parse_args()


def main():
    """Main function."""

    # parse arguments and set CTC decoder
    args = parse_args()
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    # infer text on test image
    model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump, model_dir=args.model_dir)
    infer(model, args.img_file, args.outfile)


if __name__ == '__main__':
    main()
