import re
from operator import add

input_path = "./problem1_inputs/"

# Read the file. spark.read.text creates one DataFrame row per input line.
df = spark.read.text(input_path)


def normalize(row):
    """
    Convert one Spark Row into a normalized list of words.

    Example:
        Row(value="I had seen little.")
        -> ["i", "had", "seen", "little"]
    """
    line = row["value"]

    # Convert to lowercase.
    line = line.lower()

    # Remove every character other than a-z and spaces.
    line = re.sub(r"[^a-z ]", "", line)

    # split() handles repeated spaces and removes empty strings.
    return line.split()


def generate_pairs(row):
    """
    Generate ordered word-pair contributions for one line.

    Each output has the form:

        ((word1, word2), 1)

    word1 is the focal word.
    word2 is a word appearing on the same line.
    """
    words = normalize(row)

    if not words:
        return []

    # Process each focal word once per line while preserving first-seen order.
    unique_focal_words = list(dict.fromkeys(words))

    pair_emissions = []

    for word1 in unique_focal_words:
        self_pair_emitted = False

        # Iterate over the original list, not the distinct list.
        # This preserves repeated occurrences of target words.
        for word2 in words:

            if word1 == word2:
                # Emit a self-pair once, even if word1 appears multiple
                # times in the line.
                if not self_pair_emitted:
                    pair_emissions.append(((word1, word2), 1))
                    self_pair_emitted = True
            else:
                # Repeated occurrences of other words are preserved.
                pair_emissions.append(((word1, word2), 1))

    return pair_emissions


# Convert the DataFrame to an RDD, generate all pair contributions,
# and aggregate identical pairs across all lines.
word_pair_counts = (
    df.rdd
      .flatMap(generate_pairs)
      .reduceByKey(add)
)

# Inspect the results.
results = word_pair_counts.collect()

for pair, count in sorted(results):
    print(pair, count)