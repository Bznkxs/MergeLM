import json
import numpy as np
import random
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
def read_retrieval_head(retrieval_head_file="./Mistral-7B-32k-frozen_150.json", cutoff=0.1, *args, **kwargs):
    """
    Read the retrieval heads from the file.
    The retrieval heads are sorted by their retrieval scores in descending order.
    :param retrieval_head_file: the file containing the retrieval heads
    :param cutoff: the cutoff score. The default behavior is that the function returns the heads with scores at least the cutoff.
    :param kwargs: exclude: a list of heads to exclude. These heads would not appear in the returned heads. The argument would be in the same format as the output of this function.
                      random: the number of heads to randomly sample. If provided, then cutoff is ignored. Heads will be randomly sampled. This is used as a baseline feature.
    :return: a list. Each element is a tuple of a head and its score. The head is a tuple of layer and head index. The score is the average score of the head.
    """
    with open(retrieval_head_file, "r") as f:
        head_list = json.load(f)
    head_score_list = [([int(number) for number in item[0].split('-')], np.mean(item[1])) for item in head_list.items()]
    head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True)
    if kwargs.get("exclude") is not None:
        exclude = set(tuple(x[0]) for x in kwargs.get("exclude"))
        head_score_list = [item for item in head_score_list if tuple(item[0]) not in exclude]
    if kwargs.get("random") is not None:
        head_score_list = random.sample(head_score_list, kwargs.get("random"))
    else:
        if cutoff > 1:
            assert int(cutoff) == cutoff, "cutoff should be an integer if it is greater than 1."
            print(f"Keeping maximum {cutoff} heads.")
            return head_score_list[:int(cutoff)]

        for i, (head, score) in enumerate(head_score_list):
            if score < cutoff:
                print(f"{i} of {len(head_score_list)} heads ({i / len(head_score_list)}) have score at least {cutoff}")
                return head_score_list[:i]

    return head_score_list

def get_score_table(head_list):
    """
    Get the score table from the head list.
    :param head_list: a list of heads. Each head is a tuple of a layer and a head index. The score is the average score of the head.
    :return: a list of lists. Each list is a row in the table. The first element is the head, and the second element is the score.
    """
    score_table = [[0 for _ in range(32)] for _ in range(32)]
    for head, score in head_list:
        score_table[head[0]][head[1]] = score
    return np.array(score_table)

def draw_table(score_table, save_path=None):
    """
    score_table is a list of lists.
    Use matplotlib to draw a table. Use color to indicate the value of the score.
    """
    plt.figure()
    cmap = plt.get_cmap("PuOr")
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    plt.imshow(score_table, interpolation="none", norm=BoundaryNorm(np.arange(-1, 1, 0.01), cmap.N), cmap=cmap)
    plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path)


def compare_scores(file1, file2):
    """
    Compare the scores of two retrieval heads.
    :param file1: the file containing the first retrieval heads
    :param file2: the file containing the second retrieval heads
    """
    heads1 = read_retrieval_head(file1, 0)
    heads2 = read_retrieval_head(file2, 0)

    table1 = get_score_table(heads1)
    table2 = get_score_table(heads2)
    diff = table2 - table1

    draw_table(get_score_table(heads1), save_path=file1.replace(".json", ".png"))
    draw_table(get_score_table(heads2), save_path=file2.replace(".json", ".png"))
    draw_table(diff, save_path="diff_" + file1.replace(".json", "_") + file2.replace(".json", ".png"))


if __name__ == "__main__":
    compare_scores("Mistral-7B-32k-100.json", "Mistral-7B-32k-frozen_150.json")

