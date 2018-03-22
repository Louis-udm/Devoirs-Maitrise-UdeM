import loader
import metric
import timeit
import time


def most_frequent_word(word_dict):
    most_freq_word = ""
    freq = 0
    for word in word_dict:
        if word_dict[word] > freq:
            most_freq_word = word
            freq = word_dict[word]
    return most_freq_word


def train(lemmatized_sentences, surface_sentences):
    lemma_to_surface = {}

    for lemmatized_sentence, surface_sentence in zip(lemmatized_sentences, surface_sentences):
        for lemma, word in zip(lemmatized_sentence.split(), surface_sentence.split()):
            if lemma not in lemma_to_surface:
                lemma_to_surface[lemma] = {}

            if word in lemma_to_surface[lemma]:
                lemma_to_surface[lemma][word] += 1
            else:
                lemma_to_surface[lemma][word] = 1

    for lemma in lemma_to_surface:
        lemma_to_surface[lemma] = most_frequent_word(lemma_to_surface[lemma])

    return lemma_to_surface


def predict(original_lemmatized_sentences, lemma_to_surface):
    translated_sentences = []
    for sentence in original_lemmatized_sentences:
        sentence_translation = ""
        for lemma in sentence.split():
            word = lemma

            if lemma in lemma_to_surface:
                word = lemma_to_surface[lemma]

            sentence_translation += " " + word

        translated_sentences.append(sentence_translation.lstrip())

    return translated_sentences


print("Started at {}\n".format(time.strftime("%d-%m-%Y %H:%M:%S")))
start_time = timeit.default_timer()
lemmatized_sentences, surface_sentences = loader.load("data/train")
lemma_to_surface = train(lemmatized_sentences, surface_sentences)
elapsed_time = timeit.default_timer() - start_time

print("Training (using {} sentences)".format(len(lemmatized_sentences)))
print("- Elapsed time {0:.2f}s".format(elapsed_time))
print()

original_lemmatized_sentences, target_sentences = loader.load("data/test")

translated_sentences = predict(original_lemmatized_sentences, lemma_to_surface)

accuracy = metric.accuracy(target_sentences, translated_sentences)
metric.display_result(len(target_sentences), accuracy)

print(translated_sentences[6])
print(target_sentences[6])
