import nltk
import sys
from typing import Optional

class CorpusStatistics:
    """
    A class to analyze a corpus in plain text or a list of segments.
    """

    def __init__(self, path: Optional[str]=None, expression: str=r"[\w+\/\\\@\-\.\#\:]+\b", corpus: Optional[list]=None, name: Optional[str]=None):
        self.corpus = []
        if path is not None:
            self.load_corpus(path)
        elif corpus is not None:
            self.load_corpus_from_list(corpus)
        if name is not None:
            self.name = name
        else:
            self.name = "corpus"
        self.tokenizer = nltk.tokenize.RegexpTokenizer(expression)

    def set_Regexp(self, expression) -> None:
        """
        Set the regular expression used to tokenize the corpus.
        """
        self.tokenizer = nltk.tokenize.RegexpTokenizer(expression)

    def load_corpus(self, path) -> None:
        """
        Load a corpus from a file where each line is a segment.
        Ensures each line is treated as a single segment, stripping any trailing whitespace.
        """
        with open(path, "r", encoding="utf-8") as f:
            self.corpus = []
            for line in f:
                segment = line.rstrip('\r\n')
                if segment:  # Optionally skip empty lines
                    self.add_segment(segment)

    def load_corpus_from_list(self, corpus:list) -> None:
        """
        Load a corpus from a list.
        """
        self.corpus = []
        for seg in corpus:
            self.corpus.append(seg)
    
    def add_segment(self, segment:str) -> None:
        """
        Add a segment to the corpus.
        """
        self.corpus.append(segment)

    def segments(self) -> list:
        """
        Return the segments of the corpus.
        """
        return self.corpus

    def seg_count(self) -> int:
        """
        Count the number of segments in the corpus.
        """
        return len(self.corpus)
    
    def words(self) -> list:
        """
        Return the words of the corpus.
        """
        words = []
        for sent in self.corpus:
            for word in self.tokenizer.tokenize(sent):
                words.append(word)
        return words
    
    def words_count(self) -> int:
        """
        Count the number of words in the corpus.
        """
        return len(self.words())
    
    def vocab(self) -> list:
        """
        Return the vocabulary of the corpus.
        """
        vocab = []
        for sent in self.corpus:
            for word in self.tokenizer.tokenize(sent):
                if word not in vocab:
                    vocab.append(word)
        return vocab
    
    def vocab_size(self) -> int:
        """
        Return the size of the vocabulary of the corpus.
        """
        return len(self.vocab())
    
    def avg_seg_len(self) -> float:
        """
        Return the average length of the segments in the corpus.
        """
        return (self.words_count() / self.seg_count()) if self.seg_count() > 0 else 0
    
    def min_seg(self) -> str:
        """
        Return the shortest segment in number of words in the corpus.
        """
        min_len = sys.maxsize
        min_seg = ""
        for sent in self.corpus:
            if len(self.tokenizer.tokenize(sent)) < min_len:
                min_len = len(self.tokenizer.tokenize(sent))
                min_seg = sent
        return min_seg
    
    def max_seg(self) -> str:
        """
        Return the longest segment in number of words in the corpus.
        """
        max_len = 0
        max_seg = ""
        for sent in self.corpus:
            if len(self.tokenizer.tokenize(sent)) > max_len:
                max_len = len(self.tokenizer.tokenize(sent))
                max_seg = sent
        return max_seg
    
    def min_seg_len(self) -> int:
        """
        Return the minimum word length of the segments in the corpus.
        """
        return len(self.tokenizer.tokenize(self.min_seg()))
    
    def max_seg_len(self) -> int:
        """
        Return the maximum word length of the segments in the corpus.
        """
        return len(self.tokenizer.tokenize(self.max_seg()))

    def save(self, file:str) -> None:
        """
        Save the corpus to a file.
        """
        with open(file, "w") as f:
            for sent in self.corpus:
                f.write(sent.replace("\n", " ") + "\n")

    def stats(self, file:str=None, save:bool=False) -> None:
        """
        Print or save in a file all the corpus analysis.
        """
        if save:
            if file is None:
                file = self.name + "_stats.txt"
            with open(file, 'w') as f:
                f.write("Segments\tWords\tVocab\tSeg. Len.\tMin. Seg. Len.\tMax. Seg. Len.\tMin. Seg.\tMax. Seg.\n")
                stats=("%d\t%d\t%d\t%.2f\t%d\t%d" % (self.seg_count(), self.words_count(), self.vocab_size(), self.avg_seg_len(), self.min_seg_len(), self.max_seg_len())).replace(".", ",")
                f.write(stats + "\t" + self.min_seg() + "\t" + self.max_seg() + "\n")
        else:
            print("Name: %s" % self.name)
            print("Segments: %d" % self.seg_count())
            print("Words: %d" % self.words_count())
            print("Vocab: %d" % self.vocab_size())
            print("Seg. Len.: %f" % self.avg_seg_len())
            print("Min. Seg. Len.: %d" % self.min_seg_len())
            print("Max. Seg. Len.: %d" % self.max_seg_len())
            print("Min. Seg.: %s" % self.min_seg())
            print("Max. Seg.: %s" % self.max_seg())
            print()

    
    def to_json(self) -> dict:
        """
        Return a JSON object with the corpus analysis.
        """
        return {
            "segments": self.seg_count(),
            "words": self.words_count(),
            "vocab": self.vocab_size(),
            "seg_len": self.avg_seg_len(),
            "min_seg": self.min_seg(),
            "max_seg": self.max_seg(),
            "min_seg_len": self.min_seg_len(),
            "max_seg_len": self.max_seg_len()
        }
