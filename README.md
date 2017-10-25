# Sequence-to-sequence KOR>ENG transliterator (WIP!)

## purpose

network to transliterate Korean (hangul) words into romanized English forms.

this is a network test for a future related task.

## data

corpus is crawled from the Korean Watchtower releases ( *raw data not included, sorry. encoded sentences are included for network training, as they cannot be used to reconstruct the original text* )

data is tokenized with `konlpy.tag.Twitter` and target data is generated with `hangul-romanize` from https://pypi.python.org/pypi/hangul-romanize/0.0.1

a total of approximately 12K unique words are found in this data source.

the input is a sequence of Korean graphemes (separated with code from github user & friend Kcrong), output is a series of English letters.

## network

the network is a simple recurrent sequence-to-sequence network in `keras`, with optional attention layer.

code for attention is based on this gist and modified to return distribution over each timestep: 

https://gist.github.com/nigeljyng/37552fb4869a5e81338f82b338a304d3

*NB: currently attention is not included as i believe there may be bugs in my modifications*

## current results

results are sub-par, considering that i have used this simple implementation for other tasks to much better success and that this task is relatively straightforward (convert each grapheme to one or more English letters, without modelling any linguistic patterns such as assimilation). results on 500 test-set examples from each network can be seen in the /result directory. it may be the case that the dataset is too small.