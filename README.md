# Bias-detection-in-NLP
The aim of this work was to detect gender bias in hiring procedures and was conducted in partnership with an industry partner.
Word embeddings are the centre of this approach since they are the fundamental building blocks of many NLP applications.
Our partner provided us with numerous CVs and cover letters of applicants so that we can train our own embeddings. Moreover, the publicly available 
GloVe word embeddings were also utilised for comparison and algorithm performance reasons.

An approach to certify and detect gender bias in the vector space of the embeddings is provided and this approach is based on unsupervised machine 
learning techniques and algebraic operations. Additionally, beginning from a 'black list' that contains gender-implying words two classifiers were used in order to generalize
this list and provide us with a set of words that need to be extracted from the applicants supporting documents during the hiring procedure.

The supporting documents used to build our own word embeddings belong to our industry partner and constitute sensitive information, thus we cannot make tham public.
