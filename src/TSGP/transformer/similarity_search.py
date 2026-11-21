"""
Similarity search for creating training pairs from archived individuals.
Uses FAISS for efficient nearest neighbor search based on semantic vectors.
"""

import os
import pickle
import time
import glob
from typing import List, Tuple

import numpy as np
import faiss
from tqdm import tqdm

from TSGP.transformer.utils import generate_vocab, create_vectorization, transform_deap_ind_to_keras_string


class SimilaritySearcher:
    """
    Searches for similar individuals based on their output vectors using FAISS.
    Creates training pairs for transformer pretraining.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the similarity searcher.
        
        Args:
            config: Configuration dictionary containing:
                - model['n_neighbors']: Number of neighbors to find
                - model['max_length']: Maximum sequence length
                - model['similarity_threshold']: Maximum distance for similar pairs
        """
        self.config = config
        self.model_config = config['model']
        
        self.n_neighbors = self.model_config['n_neighbors']
        
        self.max_length = self.model_config['max_length']
        self.similarity_threshold = self.model_config['similarity_threshold']
        
        functions = self.model_config['functions']
        variables = self.model_config['variables']
        ephemerals = self.model_config['ephemerals']
        
        vocab = generate_vocab(functions, variables, ephemerals)

        # Ragged=True to exclude padding tokens
        self.encoder_vectorization = create_vectorization(
            vocab_data=vocab,
            flags=False,
            ragged= True
        )
        
        print(f"Similarity Searcher initialized:")
        print(f"  FAISS: exact flat L2 index (no training)")
        print(f"  Neighbors per query: {self.n_neighbors}")
        print(f"  Length threshold: {self.max_length}")
        print(f"  Similarity threshold: {self.similarity_threshold}")

    @staticmethod
    def _print_stats(label: str, values) -> None:
        """Print mean, median, P5 and P95 statistics for a sequence of values."""
        if not values:
            print(f"  {label}: (no data)")
            return
        arr = np.asarray(values, dtype=float)
        print(
            f"  {label}: n={len(arr):,}, "
            f"mean={np.mean(arr):.3f}, "
            f"median={np.median(arr):.3f}, "
            f"P5={np.percentile(arr, 5):.3f}, "
            f"P95={np.percentile(arr, 95):.3f}"
        )

    def load_archives(self, file_path: str) -> Tuple[List[str], List]:
        """
        Load individuals from archive files.
        
        Args:
            file_path: Path to the archive pickle file
            
        Returns:
            Tuple of (individuals_list, output_vectors_list)
        """
        individuals_set = set()
        individuals_list = []
        output_vectors = []
        
        counter_duplicate = 0
        counter_extreme = 0
        total_loaded = 0
    
        archive_name = os.path.basename(file_path)
        print(f"  Loading {archive_name}...")
        
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        for ind in loaded_data:
            total_loaded += 1
            
            ind_string = str(ind[0])
            if ind_string in individuals_set:
                counter_duplicate += 1
                continue
            
            output_vec = ind[1]
            if (
                np.abs(output_vec).max() > 1e6 or 
                np.any(np.isnan(output_vec)) or 
                np.any(np.isinf(output_vec))
            ):
                counter_extreme += 1
                continue
            
            individuals_set.add(ind_string)
            individuals_list.append(ind_string)
            output_vectors.append(output_vec)

        print(f"\nLoading summary:")
        print(f"  Total individuals loaded: {total_loaded}")
        print(f"  Unique individuals kept: {len(individuals_list)}")
        print(f"  Duplicates skipped: {counter_duplicate}")
        print(f"  Extreme values skipped: {counter_extreme}")
        
        return individuals_list, output_vectors
    
    def filter_by_length(self, individuals_list: List[str], output_vectors: List) -> Tuple[List[str], np.ndarray, List[int]]:
        """
        Filter individuals by sequence length and unknown tokens.
        
        Args:
            individuals_list: List of individual strings
            output_vectors: List of output vectors (not numpy array yet)
            
        Returns:
            Tuple of (filtered_individuals, filtered_vectors, expression_lengths)
        """
        keras_string_list = [transform_deap_ind_to_keras_string(ind) for ind in individuals_list]
        print("  Tokenizing individuals...")
        tokenized_individuals = self.encoder_vectorization(keras_string_list).numpy()

        del keras_string_list

        print("  Creating filter mask...")
        keep_mask = np.ones(len(tokenized_individuals), dtype=bool)
        expression_lengths = []
        all_lengths = []

        for i, tokens in enumerate(tqdm(tokenized_individuals, desc="Filtering")):
            token_len = len(tokens)
            all_lengths.append(token_len)
            # Check length threshold and unknown tokens (token index 1)
            if token_len > self.max_length or 1 in tokens:
                keep_mask[i] = False
            else:
                expression_lengths.append(token_len)

        del tokenized_individuals

        removed_count = np.sum(~keep_mask)
        print(f"\nLength stats BEFORE filtering:")
        self._print_stats("Token lengths", all_lengths)
        print(f"Length stats AFTER  filtering (max_length={self.max_length}):")
        self._print_stats("Token lengths", expression_lengths)
        print(f"  Removed: {removed_count}, Kept: {np.sum(keep_mask)}")
        
        print("  Applying filter mask...")
        individuals_list = [ind for ind, keep in zip(individuals_list, keep_mask) if keep]
        vectors_array = np.array(output_vectors, dtype=np.float32)[keep_mask]
        
        del output_vectors
        
        print(f"\nFiltering by length ({self.max_length}):")
        print(f"  Removed: {removed_count}")
        print(f"  Kept: {len(individuals_list)}")
        
        return individuals_list, vectors_array, expression_lengths
    
    def build_faiss_index(self, vectors_array: np.ndarray) -> faiss.Index:
        """
        Build FAISS flat index for exact similarity search (no training).
        
        Args:
            vectors_array: Array of vectors to index
            
        Returns:
            FAISS flat L2 index
        """
        d = vectors_array.shape[1]  # Dimensionality
        
        print(f"\nBuilding FAISS index:")
        print(f"  Vector dimensionality: {d}")
        print(f"  Number of vectors: {len(vectors_array)}")
        print(f"  Min value: {np.min(vectors_array):.4f}")
        print(f"  Max value: {np.max(vectors_array):.4f}")
        
        # Flat L2 index: exact brute-force search, no training needed
        index = faiss.IndexFlatL2(d)
        
        faiss.omp_set_num_threads(os.cpu_count() or 1)
        
        print("  Adding vectors to index...")
        index.add(vectors_array)
        
        return index
    
    def search_similar_pairs(self, index: faiss.Index, vectors_array: np.ndarray, individuals_list: List[str]) -> List[Tuple[str, str, float]]:
        """
        Search for similar pairs using FAISS index.
        
        Args:
            index: Trained FAISS index
            vectors_array: Raw vectors used for FAISS search
            individuals_list: List of individual strings
            
        Returns:
            List of (individual1, individual2, distance) tuples
        """
        print(f"\nSearching within threshold {self.similarity_threshold} (range search)...")
        
        search_start = time.time()
        # range_search takes a squared L2 radius and returns every neighbour within it.
        radius = self.similarity_threshold ** 2
        lims, D_flat, I_flat = index.range_search(vectors_array, radius)
        lims = np.asarray(lims)
        D_flat = np.asarray(D_flat)
        I_flat = np.asarray(I_flat)
        search_time = time.time() - search_start
        
        print(f"  Search completed in {search_time:.2f} seconds")
        
        similar_pairs = set()
        no_neighbor_counter = 0
        all_raw_distances = []

        print("  Processing distance matrix...")
        for i in tqdm(range(len(vectors_array)), desc="Creating pairs"):
            start, end = int(lims[i]), int(lims[i + 1])
            neighbor_count = 0

            slice_dists = D_flat[start:end]
            slice_inds = I_flat[start:end]

            for idx in np.argsort(slice_dists):
                neighbor_idx = int(slice_inds[idx])
                dist_sq = float(slice_dists[idx])

                # Skip self and identical vectors
                if neighbor_idx == i or dist_sq == 0:
                    continue

                distance = float(np.sqrt(dist_sq))
                all_raw_distances.append(distance)

                pair = (individuals_list[i], individuals_list[neighbor_idx], distance)
                similar_pairs.add(pair)
                neighbor_count += 1

                if neighbor_count >= self.n_neighbors:
                    break

            if neighbor_count < self.n_neighbors:
                no_neighbor_counter += 1
        
        print(f"\nDistance stats (threshold={self.similarity_threshold}):")
        self._print_stats("Exact distances", all_raw_distances)
        print(f"\n  Individuals with insufficient neighbors: {no_neighbor_counter}")

        return list(similar_pairs)
    
    def run(self, archive_directory: str) -> None:
        """
        Run the complete similarity search pipeline over one archive directory.

        Each archive file in the directory is processed on its own and written to
        its own ``similar_pairs_<n>.pkl`` beside it, so an interrupted run resumes
        by skipping the outputs that already exist.

        Args:
            archive_directory: Directory containing behavior_archive_*.pkl files
        """
        print("=" * 60)
        print("SIMILARITY SEARCH FOR TRANSFORMER PRETRAINING")
        print("=" * 60)
        
        print("\n[1/5] Loading archives...")

        archive_paths = sorted(glob.glob(os.path.join(archive_directory, "*archive*.pkl")))
        
        if not archive_paths:
            raise RuntimeError(f"Warning: No archive files found in {archive_directory}!")
        
        print(f"\nProcessing {len(archive_paths)} archive(s) from {archive_directory}")
        for i, file_path in enumerate(archive_paths):
            archive_num = file_path.split('_')[-1].replace('.pkl', '')
        
            output_path = f"{archive_directory}/similar_pairs_{archive_num}.pkl"

            if os.path.exists(output_path):
                print(f"  Skipping {os.path.basename(file_path)}: {os.path.basename(output_path)} already exists")
                continue

            try:
                individuals_list, vectors_array = self.load_archives(file_path)
            except Exception as error:
                print(f"  Skipping {os.path.basename(file_path)}: cannot be loaded ({error})")
                continue

            print("\n[2/5] Filtering by length...")
            individuals_list, vectors_array, expression_lengths = self.filter_by_length(
                individuals_list, vectors_array
            )
            print(f"  Output vectors shape: {vectors_array.shape}")

            print("\n[3/5] Building FAISS flat index...")
            index = self.build_faiss_index(vectors_array)
            
            print("\n[4/5] Searching for similar pairs...")
            similar_pairs = self.search_similar_pairs(index, vectors_array, individuals_list)

            print("\n[5/5] Saving results...")
            
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                pickle.dump(similar_pairs, f)
            print(f"  Saved {len(similar_pairs)} pairs to: {output_path}")
            
            distances = [pair[2] for pair in similar_pairs]

            print(f"\nDistance statistics:")
            print(f"  Mean: {np.mean(distances):.4f}")
            print(f"  Min: {np.min(distances):.4f}")
            print(f"  Max: {np.max(distances):.4f}")
            print(f"  Median: {np.median(distances):.4f}")
        
        print("\n" + "=" * 60)
        print("SIMILARITY SEARCH COMPLETED SUCCESSFULLY")
