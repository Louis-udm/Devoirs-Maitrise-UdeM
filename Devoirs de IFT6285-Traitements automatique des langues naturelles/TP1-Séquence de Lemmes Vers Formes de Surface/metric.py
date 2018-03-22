def accuracy(references, translations):
    """
    Computes the average accuracy per sentence.
    :param references: The list of reference sentences as strings.
    :param translations: The list of translated sentences as strings.
    :return: The average accuracy per sentence.
    :raise ValueError: If ``references`` and ``translations`` do not have the same length.
    """
    if len(references) != len(translations):
        raise ValueError("Lists must have the same length.")

    accuracies = []
    for reference, translation in zip(references, translations):
        accuracies.append(sentence_accuracy(reference, translation))
    return sum(accuracies) / len(accuracies)


def sentence_accuracy(reference_sentence, translated_sentence):
    """
    Computes the ratio of correctly translated words in a reference sentence.
    If the translation contains more or less words than the reference, they are counted as ``false``.
    :param reference_sentence: The reference sentence as a string
    :param translated_sentence: The translation sentence as a string
    :return: the accuracy for a sentence.
    """
    correct = 0
    total = 0

    references = reference_sentence.split()
    translations = translated_sentence.split()
    for reference, translation in zip(references, translations):
        if reference == translation:
            correct += 1
        total += 1

    total += abs(len(references) - len(translations))
    return correct / total


def display_result(sentences_number, accuracy_score):
    print("Prediction (using {} sentences)".format(sentences_number))
    print("- Exactitude: {}".format(accuracy_score))
    print()
