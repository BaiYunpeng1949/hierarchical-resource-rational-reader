from collections import Counter

def list_diff_preserve_order(a, b):
    """
    Return the elements in b that are not already in a, 
    preserving order and handling duplicates properly.

    Parameters:
        a (list): The original list.
        b (list): The new list containing a plus possibly more items.

    Returns:
        list: Items in b that are new compared to a.
    """
    a_counts = Counter(a)
    result = []

    for item in b:
        if a_counts[item] > 0:
            a_counts[item] -= 1
        else:
            result.append(item)

    return result


if __name__ == "__main__":
    a = [1, 2, 0.3]
    b = [1, 2, 0.3, 5, 0.6, 2, 2]
    c = list_diff_preserve_order(a, b)

    for word_index in c:
        print(word_index, type(word_index))