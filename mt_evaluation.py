import json
import random
import time
import os
import torch
from typing import Optional
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF
from sacrebleu.metrics.ter import TER
from corpus_statistics import CorpusStatistics
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
import transformers
import logging

# Suppress transformers warnings
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()

class MTEvaluation:
    """
    Class to evaluate MT 
    """

    def __init__(self, lng_src: str, lng_tgt: str, engines: dict, source: Optional[str]=None, references: Optional[str]=None, dataset: str="dataset"):
        """

        :param lng_src: Source language
        :param lng_tgt: Target language
        :param engines: Engines to test (dict with name as key and ID as value)
        :param source: Source file
        :param references: Reference file
        :param dataset: Dataset name
        :param bleu_tok: BLEU tokenizer
        """
        self.lng_src = lng_src
        self.lng_tgt = lng_tgt
        self.engines = engines
        if source is not None:
            self.src = CorpusStatistics(path=source, name="source")
        if references is not None:
            self.ref = CorpusStatistics(path=references, name="reference")
        self.dataset = dataset
        self.mt = {}
        self.time = {}
        for engine in engines.keys():
            self.mt[engine] = CorpusStatistics(name=engine)
            self.time[engine] = 0
        self.bleu = BLEU(effective_order=True, trg_lang=self.lng_tgt)
        self.chrf = CHRF(word_order=2)
        self.ter = TER()
        self.errors = {}
        self.lang_mapping = {
            "en": "English",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "ro": "Romanian",
        }

    
    def set_source(self, source:str) -> None:
        """
        Set the source file
        """
        self.src = CorpusStatistics(path=source, name="source")

    def set_source_from_list(self, source:list) -> None:
        """
        Set the source file from a list
        """
        self.src = CorpusStatistics(name="source")
        self.src.load_corpus_from_list(source)
    
    def set_references(self, references:str) -> None:
        """
        Set the references file
        """
        self.ref = CorpusStatistics(path=references, name="reference")

    def set_references_from_list(self, references:list) -> None:
        """
        Set the references file from a list
        """
        self.ref = CorpusStatistics(name="reference")
        self.ref.load_corpus_from_list(references)
    
    def set_dataset(self, dataset:str) -> None:
        """
        Set the dataset name
        """
        self.dataset = dataset
    
    def set_random(self, n:int) -> None:
        """
        Set a random number of sentences for the source and reference corpuses
        """
        random.seed(42)
        indices = random.sample(range(0, self.src.seg_count()), n)
        src = [self.src.segments()[i] for i in indices]
        self.src.load_corpus_from_list(src)
        ref = [self.ref.segments()[i] for i in indices]
        self.ref.load_corpus_from_list(ref)
        for engine in self.engines.keys():
            if len(self.mt[engine].segments()) > 0:
                mt = [self.mt[engine].segments()[i] for i in indices]
                self.mt[engine].load_corpus_from_list(mt)

    def set_sample(self, n:int) -> None:
        """
        Set a sample of sentences for the source and reference corpuses
        """
        src = self.src.segments()[:n]
        self.src.load_corpus_from_list(src)
        ref = self.ref.segments()[:n]
        self.ref.load_corpus_from_list(ref)
        for engine in self.engines.keys():
            if len(self.mt[engine].segments()) > 0:
                mt = self.mt[engine].segments()[:n]
                self.mt[engine].load_corpus_from_list(mt)

    def get_bleu(self, engines: Optional[list] = None) -> dict:
        """
        Get the BLEU for a specific engine or a list of engines
        """
        e = []
        bleu = {}
        if engines is not None:
            e = engines
        else:
            e = self.engines.keys()
        for engine in e:
            bleu[engine] = self.bleu.corpus_score(self.ref.segments(), [self.mt[engine].segments()])
        return bleu
    def get_chrf(self, engines: Optional[list] = None) -> dict:
        """
        Get the CHRF for a specific engine or a list of engines
        """
        e = []
        chrf = {}
        if engines is not None:
            e = engines
        else:
            e = self.engines.keys()
        for engine in e:
            chrf[engine] = self.chrf.corpus_score(self.ref.segments(), [self.mt[engine].segments()])
        return chrf
        return chrf
    
    def get_ter(self, engines: Optional[list] = None) -> dict:
        """
        Get the TER for a specific engine or a list of engines
        """
        e = []
        ter = {}
        if engines is not None:
            e = engines
        else:
            e = self.engines.keys()
        for engine in e:
            ter[engine] = self.ter.corpus_score(self.ref.segments(), [self.mt[engine].segments()])
        return ter
    
    def get_references(self) -> list:
        """
        Get the reference
        """
        return self.ref.segments()
    
    def get_source(self) -> list:
        """
        Get the source
        """
        return self.src.segments()

    def get_mt(self, engines: Optional[list] = None) -> dict:
        """full_report
        Get the MT for a specific engine or a list of engines
        """
        e = []
        mt = {}
        if engines is not None:
            e = engines
        else:
            e = self.engines.keys()
        for engine in e:
            mt[engine] = self.mt[engine].segments()
        return mt
    
    def get_time(self, engines: Optional[list] = None) -> dict:
        """
        Get the translation time for a specific engine or a list of engines
        """
        e = []
        t = {}
        if engines is not None:
            e = engines
        else:
            e = self.engines.keys()
        for engine in e:
            t[engine] = self.time[engine]
        return t
    
    def translate(self, engines: Optional[list] = None, save: bool = False, folder: Optional[str] = None, to_json: bool = False):
        """
        Translate the source file with the specified engines (LLMs) and save the results in a JSON or text file or return a json object.
        Engines should be a dict with engine names as keys and HuggingFace model names or callable pipelines as values.
        """

        e = []
        if engines is not None:
            e = engines
        else:
            e = self.engines.keys()
        if not save and to_json:
            data = {}
            for engine in e:
                data[engine] = {}
        if save and folder:
            if not os.path.exists(folder):
                os.makedirs(folder)

        print("Translating from '" + self.lng_src + "' to '" + self.lng_tgt + "' with engines: " + str(e))
        print()
        
        for engine in e:
            print("Translating with '" + engine + "'")

            # Special cases
            if engine == "LLaMA":
                # LLaMA is a special case
                model_info = self.engines[engine]
                # Use chat-style pipeline
                translator = transformers.pipeline(
                    "text-generation",
                    model=model_info,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map="auto",
                )

                self.mt[engine] = CorpusStatistics(name="mt_" + engine)
                segments = self.src.segments()
                self.time[engine] = 0
                try:
                    for seg in tqdm(segments, desc=f"Translating ({engine})"):
                        messages = [
                            {
                                "role": "system",
                                "content": f"You are a translation assistant. Translate the following sentence into {self.lang_mapping[self.lng_tgt]}. Respond **only** with the translation, no labels, no other languages.",
                            },
                            {
                                "role": "user",
                                "content": f"{self.lang_mapping[self.lng_src]}: {seg}\n{self.lang_mapping[self.lng_tgt]}:",
                            },
                        ]
                        start = time.time()
                        output = translator(messages, max_new_tokens=50)
                        end = time.time()
                        self.time[engine] += end - start
                        # Extract the translation from the output
                        translation = output[0]["generated_text"][-1]["content"]
                        translation = str(translation).strip()
                        self.mt[engine].add_segment(translation)
                except Exception as ex:
                    if self.mt[engine].seg_count() == 0:
                        self.errors[engine] = "Empty translation"
                    else:
                        self.errors[engine] = "Error"
                    print(ex)
            elif engine == "M2M100":
                # M2M100 is a special case, we need to use the M2M100Tokenizer and M2M100ForConditionalGeneration
                model_info = self.engines[engine]
                tokenizer = M2M100Tokenizer.from_pretrained(model_info, use_fast=True)
                model = M2M100ForConditionalGeneration.from_pretrained(
                    model_info,
                    device_map="auto",
                    torch_dtype="auto"
                )
                assert torch.cuda.is_available(), "CUDA is not available."

                self.mt[engine] = CorpusStatistics(name="mt_" + engine)
                segments = self.src.segments()
                self.time[engine] = 0
                try:
                    for seg in tqdm(segments, desc=f"Translating ({engine})"):
                        encoded_src = tokenizer(seg, return_tensors="pt", padding=True, truncation=True).to(model.device)
                        start = time.time()
                        # Generate output
                        generated_tokens = model.generate(**encoded_src, max_new_tokens=30, forced_bos_token_id=tokenizer.get_lang_id(self.lng_tgt))
                        end = time.time()
                        self.time[engine] += end - start
                        # Remove the prompt from the output
                        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                        output = output.strip()
                        # HuggingFace pipeline returns the prompt with the generated text, so we need to extract the translation
                        self.mt[engine].add_segment(output)
                except Exception as ex:
                    if self.mt[engine].seg_count() == 0:
                        self.errors[engine] = "Empty translation"
                    else:
                        self.errors[engine] = "Error"
                    print(ex)
            else:
                # Get the HuggingFace pipeline or model name
                model_info = self.engines[engine]
                if callable(model_info):
                    translator = model_info
                else:
                    # Assume model_info is a model name string
                    tokenizer = AutoTokenizer.from_pretrained(model_info, use_fast=True)
                    model = AutoModelForCausalLM.from_pretrained(

                        model_info,
                        device_map="auto",
                        torch_dtype="auto"
                    )
                    assert torch.cuda.is_available(), "CUDA is not available."
                    # Create a translation pipeline
                    translator = pipeline("text-generation", model=model, tokenizer=tokenizer)

                self.mt[engine] = CorpusStatistics(name="mt_" + engine)
                segments = self.src.segments()
                self.time[engine] = 0
                try:
                    for seg in tqdm(segments, desc=f"Translating ({engine})"):
                        prompt = self.lang_mapping[self.lng_src] + ": " + seg + " " + self.lang_mapping[self.lng_tgt] + ":"
                        start = time.time()
                        # Generate output
                        output = translator(prompt, max_new_tokens=30, do_sample=False)[0]['generated_text']
                        end = time.time()
                        self.time[engine] += end - start
                        # Remove the prompt from the output
                        output = output[len(prompt):].strip()
                        # HuggingFace pipeline returns the prompt with the generated text, so we need to extract the translation
                        self.mt[engine].add_segment(output)
                except Exception as ex:
                    if self.mt[engine].seg_count() == 0:
                        self.errors[engine] = "Empty translation"
                    else:
                        self.errors[engine] = "Error"
                    print(ex)

            if self.mt[engine].seg_count() > 0 and self.mt[engine].seg_count() != self.src.seg_count():
                self.errors[engine] = "Translation length different from source length"
                print("Translation length different from source length")

            if save and not to_json:
                if folder is not None:
                    file = folder + "/" + self.dataset + "_" + engine + "_" + self.lng_src + "_" + self.lng_tgt + ".txt"
                else:
                    file = self.dataset + "_" + engine + "_" + self.lng_src + "_" + self.lng_tgt + ".txt"
                self.mt[engine].save(file)
            elif save and to_json:
                if folder is not None:
                    file = folder + "/" + self.dataset + "_" + engine + "_" + self.lng_src + "_" + self.lng_tgt + ".json"
                else:
                    file = self.dataset + "_" + engine + "_" + self.lng_src + "_" + self.lng_tgt + ".json"
                with open(file, 'w') as f:
                    for src_seg, mt_seg in zip(self.src.segments(), self.mt[engine].segments()):
                        f.write(json.dumps({"src": src_seg, "mt": mt_seg}) + "\n")
            elif to_json:
                id = 1
                for src_seg, mt_seg in zip(self.src.segments(), self.mt[engine].segments()):
                    data[engine][id] = {"src": src_seg, "mt": mt_seg}
                    id += 1
            print("Translation with '%s' done in %.4f seconds" % (engine, self.time[engine]))
        if not save and to_json:
            return data

    def corpus_evaluate(self, engines: Optional[list] = None, save: bool = False, folder: Optional[str] = None, to_json: bool = True, print_results: bool = False):
        """
        Evaluate the translation for a specific engine or a list of engines and save the results in a file or return a json object
        """
        e = []
        if engines is not None:
            e = engines
        else:
            e = self.engines
        if not save and to_json:
            data = {}
            for engine in e:
                data[engine] = {}
        if save and folder:
            if not os.path.exists(folder):
                os.makedirs(folder)
        for engine in e:
            if self.mt[engine].seg_count() != 0 and self.ref.seg_count() != 0:
                b = self.bleu.corpus_score(self.mt[engine].segments(), [self.ref.segments()])
                c = self.chrf.corpus_score(self.mt[engine].segments(), [self.ref.segments()])
                t = self.ter.corpus_score(self.mt[engine].segments(), [self.ref.segments()])
                if save and not to_json:
                    if folder is not None:
                        file = folder + "/" + self.dataset + "_scores_" + engine + "_" + self.lng_src + "_" + self.lng_tgt + ".txt"
                    else:
                        file = self.dataset + "_scores_" + engine + "_" + self.lng_src + "_" + self.lng_tgt + ".txt"
                    with open(file, 'w') as f:
                        f.write("BLEU\tCHRF\tTER\n")
                        f.write(str(b.score) + "\t" + str(c.score) + "\t" + str(t.score))
                elif save and to_json:
                    if folder is not None:
                        file = folder + "/" + self.dataset + "_scores_" + engine + "_" + self.lng_src + "_" + self.lng_tgt + ".json"
                    else:
                        file = self.dataset + "_scores_" + engine + "_" + self.lng_src + "_" + self.lng_tgt + ".json"
                    with open(file, 'w') as f:
                        f.write(json.dumps({"bleu": b.score, "chrf": c.score, "ter": t.score}))
                elif to_json:
                    data[engine] = {"bleu": b.score, "chrf": c.score, "ter": t.score}
                    
                if print_results:
                    print("Engine: " + engine)
                    print("BLEU: ", b.score)
                    print("CHRF: ", c.score)
                    print("TER: ", t.score)
                    print()
            elif self.mt[engine].seg_count() == 0:
                print("No segments to evaluate for engine " + engine + ".")
            elif self.ref.seg_count() == 0:
                print("No reference segments to evaluate for engine " + engine + ".")
        if not save and to_json:
            return data

    def segment_evaluate(self, engines: Optional[list] = None, save: bool = False, folder: Optional[str] = None, to_json: bool = False):
        """
        Evaluate the translation per segment for a specific engine or a list of engines and save the results to a file or return a json object
        """
        e = []
        if engines is not None:
            e = engines
        else:
            e = self.engines
        if not save and to_json:
            data = {}
            for engine in e:
                data[engine] = {}
        if save and folder:
            if not os.path.exists(folder):
                os.makedirs(folder)
        for engine in e:
            if self.mt[engine].seg_count() != 0 and self.ref.seg_count() != 0:
                id = 1
                if save and not to_json:
                    if folder is not None:
                        file = folder + "/" + self.dataset + "_seg-scores_" + engine + "_" + self.lng_src + "_" + self.lng_tgt + ".txt"
                    else:
                        file = self.dataset + "_seg-scores_" + engine + "_" + self.lng_src + "_" + self.lng_tgt + ".txt"
                    with open(file, 'w') as f:
                        f.write("ID\tSource\tMT\tReference\tBLEU\tCHRF\tTER\tEngine\n")
                        for src_seg, mt_seg, ref_seg in zip(self.src.segments(), self.mt[engine].segments(), self.ref.segments()):
                            b = self.bleu.sentence_score(mt_seg, [ref_seg])
                            c = self.chrf.sentence_score(mt_seg, [ref_seg])
                            t = self.ter.sentence_score(mt_seg, [ref_seg])
                            stats = ("%.2f\t%.2f\t%.2f\t%d" % (b.score, c.score, t.score, self.engines[engine])).replace(".", ",")
                            f.write(str(id) + "\t" + src_seg + "\t" + mt_seg + "\t" + stats + "\n")
                            id += 1
                elif save and to_json:
                    if folder is not None:
                        file = folder + "/" + self.dataset + "_seg-scores_" + engine + "_" + self.lng_src + "_" + self.lng_tgt + ".json"
                    else:
                        file = self.dataset + "_seg-scores_" + engine + "_" + self.lng_src + "_" + self.lng_tgt + ".json"
                    with open(file, 'w') as f:
                        for src_seg, mt_seg, ref_seg in zip(self.src.segments(), self.mt[engine].segments(), self.ref.segments()):
                            b = self.bleu.sentence_score(mt_seg, [ref_seg])
                            c = self.chrf.sentence_score(mt_seg, [ref_seg])
                            t = self.ter.sentence_score(mt_seg, [ref_seg])
                            f.write(json.dumps({"src": src_seg, "mt": mt_seg, "ref": ref_seg, "bleu": b.score, "chrf": c.score, "ter": t.score, "engine": self.engines[engine]}) + "\n")
                            id += 1
                elif to_json:
                    for src_seg, mt_seg, ref_seg in zip(self.src.segments(), self.mt[engine].segments(), self.ref.segments()):
                        b = self.bleu.sentence_score(mt_seg, [ref_seg])
                        c = self.chrf.sentence_score(mt_seg, [ref_seg])
                        t = self.ter.sentence_score(mt_seg, [ref_seg])
                        data[engine][id] = {"src": src_seg, "mt": mt_seg, "ref": ref_seg, "bleu": b.score, "chrf": c.score, "ter": t.score}
                        id += 1
                else:
                    print("ID\tSource\tMT\tReference\tBLEU\tCHRF\tTER\tEngine\n")
                    for src_seg, mt_seg, ref_seg in zip(self.src.segments(), self.mt[engine].segments(), self.ref.segments()):
                        b = self.bleu.sentence_score(mt_seg, [ref_seg])
                        c = self.chrf.sentence_score(mt_seg, [ref_seg])
                        t = self.ter.sentence_score(mt_seg, [ref_seg])
                        print(str(id) + "\t" + src_seg + "\t" + mt_seg + "\t" + ref_seg + "\t" + str(b.score) + "\t" + str(c.score) + "\t" + str(t.score) + "\t" + str(self.engines[engine]) + "\n")
                        id += 1
            elif self.mt[engine].seg_count() == 0:
                print("No segments to evaluate for engine " + engine + ".")
            elif self.ref.seg_count() == 0:
                print("No reference segments to evaluate for engine " + engine + ".")
        if to_json and not save:
            return data

    def full_report(self, engines: Optional[list] = None, save:bool=False, folder: Optional[str] = None, to_json:bool=False):
        """
        Evaluate translation for a specific engine or a list of engines in one file or return a json object
        """
        e = []
        if engines is not None:
            e = engines
        else:
            e = self.engines
        if save and folder:
            if not os.path.exists(folder):
                os.makedirs(folder)
        if not save and not to_json:
            print("Specify if you want to save the results to a text or JSON file or return a json object instead.")
        elif save and not to_json:
            if folder is not None:
                file = folder + "/" + self.dataset + "_full_report_" + self.lng_src + "_" + self.lng_tgt + ".txt"
            else:
                file = self.dataset + "_full_report_" + self.lng_src + "_" + self.lng_tgt + ".txt"
            with open(file, 'w') as f:
                f.write("Source corpus statistics\n")
                f.write("Number of Segments\t%d\n" % self.src.seg_count())
                f.write("Number of Words\t%d\n" % self.src.words_count())
                f.write("Vocabulary\t%d\n" % self.src.vocab_size())
                avg = ("%.2f" % self.src.avg_seg_len())
                f.write("Average seg. length (words)\t" + avg.replace(".", ",") + "\n")
                f.write("Min seg. length (words-characters)\t%d - %d\t%s\n" % (self.src.min_seg_len(),len(self.src.min_seg()), self.src.min_seg()))
                f.write("Max seg. length (words-characters)\t%d - %d\t%s\n" % (self.src.max_seg_len(),len(self.src.max_seg()), self.src.max_seg()))
                f.write("\n")
                f.write("Automatic evaluation\n")
                f.write("Engine\tTime\tBLEU\tCHRF\tTER\n")
                for engine in e:
                    if self.mt[engine].seg_count() != 0 and self.ref.seg_count() != 0:
                        b = self.bleu.corpus_score(self.mt[engine].segments(), [self.ref.segments()])
                        c = self.chrf.corpus_score(self.mt[engine].segments(), [self.ref.segments()])
                        t = self.ter.corpus_score(self.mt[engine].segments(), [self.ref.segments()])
                        stats = ("\t%.4f\t%.2f\t%.2f\t%.2f" % (self.time[engine], b.score, c.score, t.score)).replace(".", ",")
                        f.write(engine + stats + "\n")
                    elif self.mt[engine].seg_count() == 0:
                        print("No segments to evaluate for engine " + engine + ".")
                    elif self.ref.seg_count() == 0:
                        print("No reference segments to evaluate for engine " + engine + ".")
                f.write("\n")
                f.write("\tSegment analysis\n")
                f.write("\tID\tSource\tMT\tReference\tBLEU\tCHRF\tTER\tEngine\n")
                for engine in e:
                    if self.mt[engine].seg_count() != 0 and self.ref.seg_count() != 0:
                        id = 1
                        for src_seg, mt_seg, ref_seg in zip(self.src.segments(), self.mt[engine].segments(), self.ref.segments()):
                            b = self.bleu.sentence_score(mt_seg, [ref_seg])
                            c = self.chrf.sentence_score(mt_seg, [ref_seg])
                            t = self.ter.sentence_score(mt_seg, [ref_seg])
                            stats = ("%.2f\t%.2f\t%.2f\t" % (b.score, c.score, t.score)).replace(".", ",")
                            f.write("\t" + str(id) + "\t" + src_seg + "\t" + mt_seg + "\t" + ref_seg + "\t" + stats + engine + "\n")
                            id += 1
                    elif self.mt[engine].seg_count() == 0:
                        print("No segments to evaluate for engine " + engine + ".")
                    elif self.ref.seg_count() == 0:
                        print("No reference segments to evaluate for engine " + engine + ".")
        else:
            data = {}
            data["dataset_segments"] = self.src.seg_count()
            data["dataset_words"] = self.src.words_count()
            data["dataset_vocab"] = self.src.vocab_size()
            data["dataset_avg_words"] = self.src.avg_seg_len()
            data["dataset_min_seg_words"] = self.src.min_seg_len()
            data["dataset_min_seg_chars"] = len(self.src.min_seg())
            data["dataset_min_seg"] = self.src.min_seg()
            data["dataset_max_seg_words"] = self.src.max_seg_len()
            data["dataset_max_seg_chars"] = len(self.src.max_seg())
            data["dataset_max_seg"] = self.src.max_seg()
            data["stats"] = {}
            for engine in e:
                if engine in self.errors:
                    data["stats"][engine] = {}
                    data["stats"][engine]["bleu"] = 0
                    data["stats"][engine]["chrf"] = 0
                    data["stats"][engine]["ter"] = 0
                    data["stats"][engine]["time"] = self.time[engine]
                    data["stats"][engine]["error"] = True
                    data["stats"][engine]["error_msg"] = self.errors[engine]
                elif self.mt[engine].seg_count() != 0 and self.ref.seg_count() != 0:
                    data["stats"][engine] = {}
                    b = self.bleu.corpus_score(self.mt[engine].segments(), [self.ref.segments()])
                    c = self.chrf.corpus_score(self.mt[engine].segments(), [self.ref.segments()])
                    t = self.ter.corpus_score(self.mt[engine].segments(), [self.ref.segments()])
                    data["stats"][engine]["bleu"] = b.score
                    data["stats"][engine]["chrf"] = c.score
                    data["stats"][engine]["ter"] = t.score
                    data["stats"][engine]["time"] = self.time[engine]
                    data["stats"][engine]["error"] = False
                elif self.mt[engine].seg_count() == 0:
                    data["stats"][engine] = {}
                    data["stats"][engine]["bleu"] = 0
                    data["stats"][engine]["chrf"] = 0
                    data["stats"][engine]["ter"] = 0
                    data["stats"][engine]["time"] = self.time[engine]
                    data["stats"][engine]["error"] = True
                    data["stats"][engine]["error_msg"] = "Empty translation."
                elif self.ref.seg_count() == 0:
                    data["stats"][engine] = {}
                    data["stats"][engine]["bleu"] = 0
                    data["stats"][engine]["chrf"] = 0
                    data["stats"][engine]["ter"] = 0
                    data["stats"][engine]["time"] = self.time[engine]
                    data["stats"][engine]["error"] = True
                    data["stats"][engine]["error_msg"] = "Empty reference."
            data["seg_analysis"] = {}
            for engine in e:
                if self.mt[engine].seg_count() != 0 and self.ref.seg_count() != 0:
                    data["seg_analysis"][engine] = {}
                    id = 1
                    for src_seg, mt_seg, ref_seg in zip(self.src.segments(), self.mt[engine].segments(), self.ref.segments()):
                        b = self.bleu.sentence_score(mt_seg, [ref_seg])
                        c = self.chrf.sentence_score(mt_seg, [ref_seg])
                        t = self.ter.sentence_score(mt_seg, [ref_seg])
                        data["seg_analysis"][engine][id] = {"src": src_seg, "mt": mt_seg, "ref": ref_seg, "bleu": b.score, "chrf": c.score, "ter": t.score}
                        id += 1
                elif self.mt[engine].seg_count() == 0:
                    print("No segments to evaluate for engine " + engine + ".")
                elif self.ref.seg_count() == 0:
                    print("No reference segments to evaluate for engine " + engine + ".")
            if save:
                if folder is not None:
                    file = folder + "/" + self.dataset + "_full_report_" + self.lng_src + "_" + self.lng_tgt + ".json"
                else:
                    file = self.dataset + "_full_report_" + self.lng_src + "_" + self.lng_tgt + ".json"
                with open(file, "w") as f:
                    json.dump(data, f, indent=4)
            else:
                return data
