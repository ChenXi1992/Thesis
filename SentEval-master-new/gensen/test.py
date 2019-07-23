from gensen import GenSenSingle
from scipy.spatial import distance


# Sentences need to be lowercased.
sentences = [
    'hello world .',
    'the quick brown fox jumped over the lazy dog .',
    'this is a sentence .',
    'the white cat is sleeping on the ground .'
]
vocab = [
    'the', 'quick', 'brown', 'fox', 'jumped', 'over', 'lazy', 'dog',
    'hello', 'world', '.', 'this', 'is', 'a', 'sentence', '<s>',
    '</s>', '<pad>', '<unk>'
]

gensen = GenSenSingle(
    model_folder='./data/models',
    filename_prefix='nli_large_bothskip_parse',
    pretrained_emb='./data/embedding/glove.840B.300d.h5'
)
_, reps_h_t = gensen.get_representation(
    sentences, pool='last', return_numpy=True
)

for i in range(3):
    for j in range(i, 4):
        c = 1.0 - distance.cosine(reps_h_t[i], reps_h_t[j])
        print(i, j, c)

