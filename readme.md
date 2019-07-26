Thesis code: Assessing the influence of word embedding and model architecture on sentence embedding. 

Evaluate the sentence embedding which obtained from GloVe, fasttext, ELMo, BERT, GPT2(lite), Gensen on random initialized architecture: Max/Avg overtime pooling, roi pooling, CNN, Bi- LSTM and Transformer.


Takeaway: 
Be cautious when choosing the baseline for sentence embedding. Concatenation of average and max pooling should be used if contextual word embedding(ELMo,BERT,GPT etc) is being used. As for traditional word embedding(GloVe, fasttext, Gensen etc), projecting the word embedding to high dimensions in a randomly initialized Bi-LSTM model and using it as a baseline.

The repository that (partially)used in this experiment: 

SentEval: https://github.com/facebookresearch/SentEval

Gensen: https://github.com/Maluuba/gensen

bert-as-service: https://github.com/hanxiao/bert-as-service

pytorch_pretrained_bert: https://github.com/huggingface/pytorch-transformers

rand-lstm: https://github.com/facebookresearch/randsent

Transformer: https://github.com/tensorflow/models/tree/master/official/transformer

