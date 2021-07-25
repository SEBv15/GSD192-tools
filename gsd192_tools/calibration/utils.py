import numpy as np
from typing import Tuple

def rebin(source_counts: np.ndarray, source_energies: np.ndarray, target_energies: np.ndarray) -> np.ndarray:
  """
  Re-bin the counts of one strip to match another strip's energy bins while keeping total counts constant. 
  This is needed so the strips can easily be summed.

  Parameters:
    source_counts (np.ndarray[float]): The counts to be rebinned
    source_energies (np.ndarray[float]): The energy locations/bins of the counts
    target_energies (np.ndarray[float]): The target energy bins

  Returns:
    (np.ndarray[float]): The rebinned input counts
  """
  y = np.zeros(target_energies.shape[0], dtype=float)

  # Minimum position in the output where the counts could be distributed to
  min_idx = 0

  # Go through all elements in our data and distribute their counts to the target bins
  for i in range(source_energies.shape[0]):
      # Calculate the bin width of the current bin we are trying to distribute
      this_step = source_energies[i + 1] - source_energies[i] if i + 1 < source_energies.shape[0] else source_energies[i] - source_energies[i - 1]

      # Advance the minimum position to the minimum target bin that overlaps the current source bin
      while min_idx + 1 < target_energies.shape[0] and target_energies[min_idx + 1] < source_energies[i]:
          min_idx += 1

      # Go through all the target bins that overlap the source bin and distribute the counts by overlap area
      j = min_idx
      while j < target_energies.shape[0] and target_energies[j] < source_energies[i] + this_step:
          step = target_energies[j + 1] - target_energies[j] if j + 1 < target_energies.shape[0] else target_energies[j] - target_energies[j - 1]
          overlap = max(0, min(target_energies[j] + step, source_energies[i] + this_step) - max(target_energies[j], source_energies[i]))
          y[j] += source_counts[i] * overlap / this_step
          j += 1

  return y

def sum_strips(energies: np.ndarray, counts: np.ndarray, base_strip: int) -> Tuple[np.ndarray, np.ndarray]:
  """
  Sum the strips by rebinning all strips to have the same energy bins as the base strip and adding them up.

  Parameters:
    energies (np.ndarray[float]): The energy value of every channel for every strip (192 x 4096)
    counts (np.ndarray[float]): The counts from every strip (192 x 4096)
    base_strip (int): The index of the strip to use as refernce

  Returns:
    (Tuple[np.ndarray[float], np.ndarray[float]]): The energy value of every bin and their summed counts
  """
  summed = np.copy(counts[base_strip])
  bins = np.copy(energies[base_strip])

  for i in range(counts.shape[0]):
    if i != base_strip:
      summed += rebin(counts[i], energies[i], bins)

  return (bins, summed)

