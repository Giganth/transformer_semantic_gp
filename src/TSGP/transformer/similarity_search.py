# src/TSGP/transformer/similarity_search.py
"""
Similarity search for creating training pairs from archived individuals.
Uses FAISS for efficient nearest neighbor search based on semantic vectors.
"""

import os
import pickle
import time
import glob
import collections
from typing import List, Tuple

import numpy as np
import faiss
from tqdm import tqdm
import matplotlib.pyplot as plt

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
                - model['nlist']: Number of clusters for FAISS index
                - model['nprobe']: Number of clusters to search
                - model['n_neighbors']: Number of neighbors to find
                - model['max_length']: Maximum sequence length
                - model['similarity_threshold']: Maximum distance for similar pairs
        """
        self.config = config
        self.model_config = config['model']
        
        # FAISS parameters
        self.nlist = self.model_config['nlist']
        self.nprobe = self.model_config['nprobe']
        self.n_neighbors = self.model_config['n_neighbors']
        
        # Filtering parameters
        self.max_length = self.model_config['max_length']
        self.similarity_threshold = self.model_config['similarity_threshold']
        
        # Create vectorization
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
        print(f"  FAISS nlist: {self.nlist}, nprobe: {self.nprobe}")
        print(f"  Neighbors per query: {self.n_neighbors}")
        print(f"  Length threshold: {self.max_length}")
        print(f"  Similarity threshold: {self.similarity_threshold}")
    
    def load_archives(self, archive_directory: str) -> Tuple[List[str], np.ndarray]:
        """
        Load individuals from archive files.
        
        Args:
            archive_directory: Directory containing archive pickle files
            
        Returns:
            Tuple of (individuals_list, output_vectors_array)
        """
        individuals_set = set()
        individuals_list = []
        output_vectors = []
        
        counter_duplicate = 0
        counter_extreme = 0
        total_loaded = 0
        
        archive_paths = sorted(glob.glob(os.path.join(archive_directory, "*archive*.pkl")))
        
        if not archive_paths:
            raise RuntimeError(f"No archive files found in {archive_directory}!")
        
        print(f"\nProcessing {len(archive_paths)} archive(s) from {archive_directory}")
        
        for file_path in archive_paths:
            archive_name = os.path.basename(file_path)
            print(f"  Loading {archive_name}...")
            
            with open(file_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            for ind in loaded_data:
                total_loaded += 1
                
                ind_string = str(ind[0])
                # Skip duplicates
                if ind_string in individuals_set:
                    counter_duplicate += 1
                    continue
                
                # Check for extreme values, NaN, or Inf
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
        
        vectors_array = np.array(output_vectors, dtype=np.float32)
        print(f"  Output vectors shape: {vectors_array.shape}")
        
        return individuals_list, vectors_array
    
    def filter_by_length(self, individuals_list: List[str], output_vectors: List) -> Tuple[List[str], np.ndarray, List[int]]:
        """
        Filter individuals by sequence length and unknown tokens.
        
        Args:
            individuals_list: List of individual strings
            output_vectors: List of output vectors (not numpy array yet)
            
        Returns:
            Tuple of (filtered_individuals, filtered_vectors, expression_lengths)
        """
        # Tokenize individuals
        keras_string_list = [transform_deap_ind_to_keras_string(ind) for ind in individuals_list]
        print("  Tokenizing individuals...")
        tokenized_individuals = self.encoder_vectorization(keras_string_list).numpy()

        # Free memory
        del keras_string_list

        # Create boolean mask for items to keep (memory efficient)
        print("  Creating filter mask...")
        keep_mask = np.ones(len(tokenized_individuals), dtype=bool)
        expression_lengths = []
        
        for i, tokens in enumerate(tqdm(tokenized_individuals, desc="Filtering")):
            # Check length threshold and unknown tokens (token index 1)
            if len(tokens) > self.max_length or 1 in tokens:
                keep_mask[i] = False
            else:
                expression_lengths.append(len(tokens))
        
        # Free memory
        del tokenized_individuals
        
        removed_count = np.sum(~keep_mask)
        print(f"  Removed: {removed_count}, Kept: {np.sum(keep_mask)}")
        
        # Filter using boolean indexing (memory efficient - happens in-place-ish)
        print("  Applying filter mask...")
        individuals_list = [ind for i, ind in enumerate(tqdm(individuals_list, desc="Filtering individuals")) if keep_mask[i]]
        output_vectors = [vec for i, vec in enumerate(tqdm(output_vectors, desc="Filtering vectors")) if keep_mask[i]]
        
    
        # Convert to numpy array once after filtering
        print(f"  Converting to numpy array...")
        vectors_array = np.array(output_vectors, dtype=np.float32)
        
        # Free memory
        del output_vectors
        
        print(f"\nFiltering by length ({self.max_length}):")
        print(f"  Removed: {removed_count}")
        print(f"  Kept: {len(individuals_list)}")
        
        return individuals_list, vectors_array, expression_lengths

    
    def build_faiss_index(self, vectors_array: np.ndarray) -> faiss.Index:
        """
        Build FAISS index for similarity search.
        
        Args:
            vectors_array: Array of vectors to index
            
        Returns:
            Trained FAISS index
        """
        d = vectors_array.shape[1]  # Dimensionality
        
        print(f"\nBuilding FAISS index:")
        print(f"  Vector dimensionality: {d}")
        print(f"  Number of vectors: {len(vectors_array)}")
        print(f"  Min value: {np.min(vectors_array):.4f}")
        print(f"  Max value: {np.max(vectors_array):.4f}")
        
        # Create IVF index with L2 (Euclidean) distance
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, self.nlist)
        
        print("  Training index...")
        index.train(vectors_array)
        index.nprobe = self.nprobe
        
        print("  Adding vectors to index...")
        index.add(vectors_array)
        
        # Set number of threads
        faiss.omp_set_num_threads(os.cpu_count() or 1)
        
        return index
    
    def search_similar_pairs(self, index: faiss.Index, vectors_array: np.ndarray, individuals_list: List[str]) -> List[Tuple[str, str, float]]:
        """
        Search for similar pairs using FAISS index.
        
        Args:
            index: Trained FAISS index
            vectors_array: Array of vectors
            individuals_list: List of individual strings
            
        Returns:
            List of (individual1, individual2, distance) tuples
        """
        print(f"\nSearching for {self.n_neighbors} nearest neighbors...")
        
        search_start = time.time()
        D, I = index.search(vectors_array, self.n_neighbors + 1)  # +1 because first is itself
        search_time = time.time() - search_start
        
        print(f"  Search completed in {search_time:.2f} seconds")
        
        # Process results to create training pairs
        similar_pairs = set()
        no_neighbor_counter = 0
        
        print("  Processing distance matrix...")
        for i in tqdm(range(len(I)), desc="Creating pairs"):
            neighbor_set = set()
            
            for element, neighbor_idx in enumerate(I[i]):
                # Skip self and zero distances
                if neighbor_idx == i or D[i][element] == 0:
                    continue
                
                distance = D[i][element]
                
                # Apply similarity threshold
                if distance > self.similarity_threshold:
                    continue
                
                # Recompute exact distance
                distance = float(np.linalg.norm(vectors_array[neighbor_idx] - vectors_array[i]))
                
                if distance > 0:  # Ensure non-zero distance
                    pair = (individuals_list[i], individuals_list[neighbor_idx], distance)
                    neighbor_set.add(pair)
                
                # Limit neighbors per individual
                if len(neighbor_set) >= self.n_neighbors:
                    break
            
            if len(neighbor_set) < self.n_neighbors:
                no_neighbor_counter += 1
            
            similar_pairs.update(neighbor_set)
        
        print(f"\n  Individuals with insufficient neighbors: {no_neighbor_counter}")
        
        return list(similar_pairs)
    
    def plot_statistics(self, expression_lengths: List[int], distances: List[float], output_dir: str):
        """
        Plot statistics about expressions and distances.
        
        Args:
            expression_lengths: List of expression lengths
            distances: List of distances between pairs
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot expression lengths histogram
        plt.figure(figsize=(8, 5))
        counts = collections.Counter(expression_lengths)
        plt.bar(counts.keys(), counts.values())
        plt.title('Expression Length Distribution')
        plt.xlabel('Expression Length')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.6)
        length_hist_path = os.path.join(output_dir, "expression_lengths_histogram.png")
        plt.savefig(length_hist_path)
        plt.close()
        print(f"  Saved: {length_hist_path}")
        
        # Plot distance histogram
        plt.figure(figsize=(8, 5))
        plt.hist(distances, bins=50, edgecolor='black')
        plt.title('Semantic Distance Distribution')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.6)
        dist_hist_path = os.path.join(output_dir, "distance_histogram.png")
        plt.savefig(dist_hist_path)
        plt.close()
        print(f"  Saved: {dist_hist_path}")
    
    def run(self, archive_directory: str, output_path: str) -> List[Tuple[str, str, float]]:
        """
        Run the complete similarity search pipeline.
        
        Args:
            archive_directories: List of directories containing archives
            output_path: Path to save the similar pairs pickle file
            
        Returns:
            List of similar pairs
        """
        print("=" * 60)
        print("SIMILARITY SEARCH FOR TRANSFORMER PRETRAINING")
        print("=" * 60)
        
        # Step 1: Load archives
        print("\n[1/5] Loading archives...")
        individuals_list, vectors_array = self.load_archives(archive_directory)
        
        # Step 2: Filter by length
        print("\n[2/5] Filtering by length...")
        individuals_list, vectors_array, expression_lengths = self.filter_by_length(
            individuals_list, vectors_array
        )
        
        # Step 3: Build FAISS index
        print("\n[3/5] Building FAISS index...")
        index = self.build_faiss_index(vectors_array)
        
        # Step 4: Search for similar pairs
        print("\n[4/5] Searching for similar pairs...")
        similar_pairs = self.search_similar_pairs(index, vectors_array, individuals_list)
        
        # Step 5: Save results and plot statistics
        print("\n[5/5] Saving results...")
        
        # Save pairs
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(similar_pairs, f)
        print(f"  Saved {len(similar_pairs)} pairs to: {output_path}")
        
        # Print statistics
        distances = [pair[2] for pair in similar_pairs]
        print(f"\nDistance statistics:")
        print(f"  Mean: {np.mean(distances):.4f}")
        print(f"  Min: {np.min(distances):.4f}")
        print(f"  Max: {np.max(distances):.4f}")
        print(f"  Median: {np.median(distances):.4f}")
        
        # Plot statistics
        self.plot_statistics(expression_lengths, distances, output_dir)
        
        print("\n" + "=" * 60)
        print("SIMILARITY SEARCH COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return similar_pairs
