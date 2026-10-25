import re
import sys
from operator import add

from pyspark.sql import SparkSession


def normalize(row):
    """
    Normalize one input row and return a list of words.

    Example:
        Row(value="1 The Adventure of the Blue Carbuncle';")
        ->
        ["the", "adventure", "of", "the", "blue", "carbuncle"]
    """
    line = row["value"]

    # Lowercase the line.
    line = line.lower()

    # Remove every character other than a-z and spaces.
    line = re.sub(r"[^a-z ]", "", line)

    # Split into words.
    return line.split()


def generate_pairs(row):
    """
    Generate ordered word-pair contributions for one input line.

    Each emitted record has this form:

        ((word1, word2), 1)

    word1 is the focal word.
    word2 is another word from the same line.

    For self-pairs, the first occurrence is skipped. Therefore, if a word
    occurs n times in a line, its self-pair contributes n - 1.
    """
    words = normalize(row)

    if not words:
        return []

    # Each distinct word becomes a focal word once per line.
    focal_words = list(dict.fromkeys(words))

    pair_emissions = []

    for word1 in focal_words:
        skipped_first_self_pair = False

        # Use the original word list so repeated words are preserved.
        for word2 in words:
            if word1 == word2 and not skipped_first_self_pair:
                # Skip only the first self-pair for this focal word.
                skipped_first_self_pair = True
                continue

            pair_emissions.append(((word1, word2), 1))

    return pair_emissions


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: spark-submit hw3_p1.py "
            "<input_file> <output_file>"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Create the SparkSession.
    spark = (
        SparkSession.builder
        .appName("WordPairOccurrences")
        .master("local[*]")
        .getOrCreate()
    )

    try:
        # spark.read.text creates one DataFrame row per input line.
        df = spark.read.text(input_file)

        # Generate pair contributions and aggregate identical keys.
        word_pair_counts = (
            df.rdd
            .flatMap(generate_pairs)
            .reduceByKey(add)
        )

        # Sort by pair so the output is deterministic.
        results = dict(
            word_pair_counts
            .sortByKey()
            .collect()
        )

        # Write the dictionary representation to the requested output file.
        with open(output_file, "w", encoding="utf-8") as output:
            output.write(str(results))

        print(results)

    finally:
        # Shut down the SparkSession cleanly.
        spark.stop()


if __name__ == "__main__":
    main()
