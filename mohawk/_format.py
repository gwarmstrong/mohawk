__author__ = "Daniel McDonald"

import numpy as np
import skbio


def locus_generator(start, stop, length, n, f):
    """Generate index positions

    Parameters
    ----------
    start : int
        A starting or minimum index
    stop : int
        The largest possible index
    length : int
        The length of a substring to extract
    n : int
        The number of indices to generate
    f : function
        The function to use to generate values. The expected function signature
        is f(int, int, int) -> np.array. This signature is compatible with
        np.random.randint (e.g., np.random.randint(0, 10, 3) for three indices
        within the interval [0, 3).

    Returns
    -------
    np.array
        Index positions such that each index + length is assured to be bounded
        by the interval [start, stop]
    """
    return f(start, stop - length, n)


def extract_subsequences(sequence, positions, length):
    """Extract subsequences of a given length

    Paramters
    ---------
    sequence : np.array
        A 1D vector to extract subsequences from
    positions : np.array
        A 1D vector of integers such that each integer is in the interval
        [0, len(sequence) - length)
    length : int
        The length of a subsequence

    Returns
    -------
    np.array
        A 2D array such that each row is a subsequence of `sequence`.
    """
    if positions.min() < 0 or (positions.max() - length) >= len(sequence):
        raise ValueError("A requested position is out of bounds")

    position_stops = positions + length
    indices = np.vstack([positions, position_stops]).T

    return np.vstack([sequence[i:j] for i, j in indices])


def encode_sequence(sequence):
    """Convert a DNA sequence into an integer encoded sequence

    Parameters
    ----------
    sequence : skbio.DNA
        A DNA sequence

    Returns
    -------
    np.array
        A vector of the sequence in an integer encoding
    """
    as_int = sequence.values.view(np.int8)
    empty = np.empty(as_int.size)
    empty[as_int == 65] = 0  # ord('A')
    empty[as_int == 84] = 1  # ord('T')
    empty[as_int == 71] = 2  # ord('G')
    empty[as_int == 67] = 3  # ord('C')
    return empty


def join_contigs(sequences):
    """Construct a single DNA sequence with the break points

    Parameters
    ----------
    sequence : iterable of skbio.DNA objects
        A collection of DNA sequences to join

    Returns
    -------
    skbio.DNA
        The concatenated sequence
    np.array
        The boundaries of each individual sequence. The values of this array
        are such that s[index[n]:index[n+1]] yields one of the sequences that
        had been concatenated, where "s" is the full concatenated sequence,
        index is this vector, and n is an offset into the index vector
    """
    sequences = list(sequences)  # unroll if a generator

    full_sequence = skbio.DNA.concat(sequences)

    lengths = np.empty(len(sequences) + 1, dtype=int)
    lengths[0] = 0
    lengths[1:] = np.array(list(map(len, sequences))).cumsum()

    return full_sequence, lengths


def sample_from_contig_set(sequences, depth, length, randfunc):
    """Sample reads from a set of sequences

    The input set of sequences may come from, for example, a genome. The genome
    may be represented by many contigs. This code will randomly sample with
    replacement from the contigs such that the number of reads sampled is
    proportional to the size of the contig. For example, let's say we had two
    contigs, one that was 800nt long, and one that was 200nt long. If we wanted
    to obtain 100 reads, we would expect that most of the reads (80%) would
    derive from the longer contig.

    This method is not perfect: the endpoints of the contigs will be under
    sampled. It's not immediately obvious whether this is a overall problem
    or how to approach it.

    Parameters
    ----------
    sequences : iterable of skbio.DNA
        Contigs that we will sample from
    depth : int
        The total number of reads to produce from these contigs
    length : int
        The length of the reads to produce
    randfunc : function
        The function to use to generate values. The expected function signature
        is f(int, int, int) -> np.array. This signature is compatible with
        np.random.randint (e.g., np.random.randint(0, 10, 3) for three indices
        within the interval [0, 3).

    Returns
    -------
    np.array
        A 2D matrix of integer values that encode the sampled reads. Each row
        is a "read".
    """
    sequence, breakpoints = join_contigs(sequences)
    sequence_encoded = encode_sequence(sequence)

    sequence_lengths = np.diff(breakpoints)
    probabilities = sequence_lengths / sequence_lengths.sum()
    n_per_contig = np.random.multinomial(depth, probabilities)

    contig_info = zip(breakpoints[:-1], breakpoints[1:], n_per_contig)
    loci = np.hstack([locus_generator(start, stop, length, n, randfunc)
                      for start, stop, n in contig_info])

    return extract_subsequences(sequence_encoded, loci, length)
