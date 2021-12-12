from dataclasses import dataclass
from time import sleep
from typing import List, Optional
from flask import current_app
from more_itertools import sliced
import numpy as np
import pandas as pd
from pathy import os
from spacy import sys
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

COSINE_FOLDER = "cosine"

# def cosine_similarity(a, b):
#     """
#     Calculate the cosine similarity between two vectors.
#     """
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@dataclass
class PartialMatrixCreationParams:
    partials_folder: str
    merged_filename: str
    keep_partials: bool = True
    merge_partials: bool = True
    step_size: int = 100
    start_from: int = 0

def _calc_partials(tfIdfMatrix: np.ndarray, params: PartialMatrixCreationParams):
    lineCount = tfIdfMatrix.shape[0]

    # Create all folders and subfolders to store partial similarity matrices
    try:
        os.makedirs(params.partials_folder, exist_ok=False)
    except FileExistsError:
        print(f"ERROR: Folder '{params.partials_folder}' already exists, please remove it before running this script", file=sys.stderr)
        exit(1)

    progressBar = tqdm(
        range(params.start_from, lineCount, params.step_size),
        desc="Creating cosine similarity matrix",
        unit="lines",
        unit_scale=True, # Show the progress in lines
        unit_divisor=params.step_size # Show the progress in lines
    )

    for i in progressBar:
        # Slice of lines to process (copies all columns on each line)
        sliced_matrix = tfIdfMatrix[i:i+params.step_size]

        # Calculate cosine similarity between each line and the rest of the lines
        partial_sim_matrix = cosine_similarity(sliced_matrix, tfIdfMatrix)

        # Save partial similarity matrix
        partial_sim_matrix_filename = f"{params.partials_folder}/partial_{i}_{i+params.step_size-1}.csv"
        np.savetxt(partial_sim_matrix_filename, partial_sim_matrix, delimiter=",")

def create_cosine_similarity_matrix(headers: List[str], tfIdfMatrix: np.ndarray, params: PartialMatrixCreationParams) -> None:
    assert type(tfIdfMatrix) == np.ndarray, f"Matrix must be a numpy array, got {type(tfIdfMatrix)}"

    # Calculate cosine similarity between each line and the rest of the lines (in slices, to avoid memory issues)
    _calc_partials(tfIdfMatrix, params)

    # Merge partial similarity matrices
    if params.merge_partials:
        merge_partial_similarity_matrices(headers, params.partials_folder, params.merged_filename)
    
    # Remove partial similarity matrices
    if not params.keep_partials:
        os.removedirs(params.partials_folder)

# Merges partial similarity matrices into a single file, just merges csv files withou overwriting and without loading them all in memory
def merge_partial_similarity_matrices(headers: List[str], partials_folder: str, merged_filename: str) -> None:
    # Get all partial similarity matrices
    partial_sim_matrices_paths: List[str] = [ f'{partials_folder}/{filename}' for filename in os.listdir(partials_folder) if filename.endswith(".csv") ]
    partial_sim_matrices_paths.sort()

    csv_header = ','.join(headers)

    with open(f"{merged_filename}", "w") as merged_file:
        merged_file.writelines([csv_header+"\n"])

        progressbar = tqdm(partial_sim_matrices_paths, desc="Merging partial similarity matrices", unit="files", unit_scale=True)
        for partial_sim_matrix in progressbar:
            with open(partial_sim_matrix, "r") as partial_sim_matrix_file:
                # Read partial similarity matrix
                partial_sim_matrix_lines = partial_sim_matrix_file.readlines()

                # Write partial similarity matrix to merged file
                merged_file.writelines(partial_sim_matrix_lines)

def create_weighted_cosine_similarity_matrix(headers: List[str], matrix: np.ndarray, colWeights: np.ndarray, params: PartialMatrixCreationParams) -> str:
    # Applying weights to u and v instead of in the cosine function (it was too slow)
    # https://www.tutorialguruji.com/python/how-does-the-parameter-weights-work-in-scipy-spatial-distance-cosine/
    # https://stats.stackexchange.com/questions/384419/weighted-cosine-similarity/448904#448904

    sqrt_colWeights = np.sqrt(colWeights)
    weighted_matrix = matrix * sqrt_colWeights
    return create_cosine_similarity_matrix(headers, weighted_matrix, params)

if __name__ == "__main__":
    # test()
    pass