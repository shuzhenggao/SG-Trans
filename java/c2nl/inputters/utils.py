import logging
import random
import string
import numpy as np
from scipy import sparse
from collections import Counter
from tqdm import tqdm

from c2nl.objects import Code, Summary
from c2nl.inputters.vocabulary import Vocabulary, UnicodeCharsVocabulary
from c2nl.inputters.constants import BOS_WORD, EOS_WORD, PAD_WORD, \
    UNK_WORD, TOKEN_TYPE_MAP, AST_TYPE_MAP, DATA_LANG_MAP, LANG_ID_MAP
from c2nl.utils.misc import count_file_lines

logger = logging.getLogger(__name__)


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def generate_random_string(N=8):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N))


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def process_examples(lang_id,
                     source,
                     source_sub,
                     source_tag,
                     target,
                     source_ky,
                     source_t5,
                     source_it,
                     source_is,
                     source_df_sp,
                     source_cf_sp,
                     max_src_len,
                     max_tgt_len,
                     code_tag_type,
                     uncase=False,
                     test_split=True):
    code_tokens = source.split()
    code_type = []
    code_subtoken = source_sub.split()
    code_keyword = source_ky.split()
    code_type5 = source_t5.split()
    code_intoken = source_it.split()
    code_instatement = source_is.split()
    if source_tag is not None:
        code_type = source_tag.split()
        if len(code_tokens) != len(code_type):
            return None

    code_tokens = code_tokens[:max_src_len]
    code_type = code_type[:max_src_len]
    code_subtoken = code_subtoken[:max_src_len]
    code_keyword = code_keyword[:max_src_len]
    code_type5 = code_type5[:max_src_len]
    code_intoken = code_intoken[:max_src_len]
    code_instatement = code_instatement[:max_src_len]
    if len(code_tokens) == 0:
        return None

    TAG_TYPE_MAP = TOKEN_TYPE_MAP if \
        code_tag_type == 'subtoken' else AST_TYPE_MAP
    code = Code()
    code.text = source
    code.language = lang_id
    code.tokens = code_tokens
    code.type = [TAG_TYPE_MAP.get(ct, 1) for ct in code_type]
    code.keyword = code_keyword
    code.type5 = code_type5
    code.intoken = code_intoken
    code.instatement = code_instatement
    code.subtoken = code_subtoken
    code.dataflow = source_df_sp
    code.controlflow = source_cf_sp

    if code_tag_type != 'subtoken':
        code.mask = [1 if ct == 'N' else 0 for ct in code_type]

    if target is not None:
        summ = target.lower() if uncase else target
        summ_tokens = summ.split()
        if not test_split:
            summ_tokens = summ_tokens[:max_tgt_len]
        if len(summ_tokens) == 0:
            return None
        summary = Summary()
        summary.text = ' '.join(summ_tokens)
        summary.tokens = summ_tokens
        summary.prepend_token(BOS_WORD)
        summary.append_token(EOS_WORD)
    else:
        summary = None

    example = dict()
    example['code'] = code
    example['summary'] = summary
    return example


def load_data(args, filenames, max_examples=-1, dataset_name='java',
              test_split=False):
    """Load examples from preprocessed file. One example per line, JSON encoded."""

    with open(filenames['src']) as f:
        sources = [line.strip() for line in
                   tqdm(f, total=count_file_lines(filenames['src']))]
    
    if filenames['src_sub'] is not None:
        with open(filenames['src_sub']) as f:
            source_sub = [line.strip() for line in
                           tqdm(f, total=count_file_lines(filenames['src_sub']))]
    else:
        source_sub = [None] * len(sources)

    if filenames['tgt'] is not None:
        with open(filenames['tgt']) as f:
            targets = [line.strip() for line in
                       tqdm(f, total=count_file_lines(filenames['tgt']))]
    else:
        targets = [None] * len(sources)

    if filenames['src_tag'] is not None:
        with open(filenames['src_tag']) as f:
            source_tags = [line.strip() for line in
                           tqdm(f, total=count_file_lines(filenames['src_tag']))]
    else:
        source_tags = [None] * len(sources)

    if filenames['src_keyword'] is not None:
        with open(filenames['src_keyword']) as f:
            source_ky = [line.strip() for line in
                           tqdm(f, total=count_file_lines(filenames['src_keyword']))]
    else:
        source_ky = [None] * len(sources)

    if filenames['src_type5'] is not None:
        with open(filenames['src_type5']) as f:
            source_t5 = [line.strip() for line in
                           tqdm(f, total=count_file_lines(filenames['src_type5']))]
    else:
        source_t5 = [None] * len(sources)

    if filenames['src_dataflow'] is not None:
        source_dataflow_sp= np.load(filenames['src_dataflow'], allow_pickle=True) # 读取
        #allmatrix=allmatrix_sp.toarray()
    else:
        source_dataflow_sp = [None] * len(sources)
    
    if filenames['src_controlflow'] is not None:
        source_controlflow_sp= np.load(filenames['src_controlflow'], allow_pickle=True)  # 读取
        #allmatrix=allmatrix_sp.toarray()
    else:
        source_controlflow_sp = [None] * len(sources)

    if filenames['src_intoken'] is not None:
        with open(filenames['src_intoken']) as f:
            source_it = [line.strip() for line in
                           tqdm(f, total=count_file_lines(filenames['src_intoken']))]
    else:
        source_it = [None] * len(sources)

    if filenames['src_instatement'] is not None:
        with open(filenames['src_instatement']) as f:
            source_is = [line.strip() for line in
                           tqdm(f, total=count_file_lines(filenames['src_instatement']))]
    else:
        source_is = [None] * len(sources)

    assert len(sources) == len(source_sub) == len(source_tags) == len(targets) == len(source_ky)  == len(source_t5) == len(source_it) == \
           len(source_is) == source_dataflow_sp.shape[0] == source_controlflow_sp.shape[0]

    examples = []
    for src, src_sub, src_tag, tgt, src_ky, src_t5, src_it, src_is, src_df_sp, src_cf_sp in\
        tqdm(zip(sources,source_sub,source_tags,targets,source_ky,source_t5,source_it,source_is,source_dataflow_sp,source_controlflow_sp),
                                  total=len(sources)):
        if dataset_name in ['java', 'python']:
            _ex = process_examples(LANG_ID_MAP[DATA_LANG_MAP[dataset_name]],
                                   src,
                                   src_sub,
                                   src_tag,
                                   tgt,
                                   src_ky,
                                   src_t5,
                                   src_it,
                                   src_is,
                                   src_df_sp,
                                   src_cf_sp,
                                   args.max_src_len,
                                   args.max_tgt_len,
                                   args.code_tag_type,
                                   uncase=args.uncase,
                                   test_split=test_split)
            if _ex is not None:
                examples.append(_ex)

        if max_examples != -1 and len(examples) > max_examples:
            break

    return examples


# ------------------------------------------------------------------------------
# Dictionary building
# ------------------------------------------------------------------------------


def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file) as f:
        for line in tqdm(f, total=count_file_lines(embedding_file)):
            w = Vocabulary.normalize(line.rstrip().split(' ')[0])
            words.add(w)

    words.update([BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD])
    return words


def load_words(args, examples, fields, dict_size=None):
    """Iterate and index all the words in examples (documents + questions)."""

    def _insert(iterable):
        words = []
        for w in iterable:
            w = Vocabulary.normalize(w)
            words.append(w)
        word_count.update(words)

    word_count = Counter()
    for ex in tqdm(examples):
        for field in fields:
            _insert(ex[field].tokens)

    # -2 to reserve spots for PAD and UNK token
    dict_size = dict_size - 2 if dict_size and dict_size > 2 else dict_size
    most_common = word_count.most_common(dict_size)
    words = set(word for word, _ in most_common)
    return words


def build_word_dict(args, examples, fields, dict_size=None,
                    no_special_token=False):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Vocabulary(no_special_token)
    for w in load_words(args, examples, fields, dict_size):
        word_dict.add(w)
    return word_dict


def build_word_and_char_dict(args, examples, fields, dict_size=None,
                             no_special_token=False):
    """Return a dictionary from question and document words in
    provided examples.
    """
    words = load_words(args, examples, fields, dict_size)
    dictioanry = UnicodeCharsVocabulary(words,
                                        args.max_characters_per_token,
                                        no_special_token)
    return dictioanry


def top_summary_words(args, examples, word_dict):
    """Count and return the most common question words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['summary'].tokens:
            w = Vocabulary.normalize(w)
            if w in word_dict:
                word_count.update([w])
    return word_count.most_common(args.tune_partial)
