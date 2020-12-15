# Research literature notes 🤓

[![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/anebz/resources/graphs/commit-activity)
[![Ask me anything!](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://www.twitter.com/aberasategi)
[![GitHub issues](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://GitHub.com/anebz/resources/issues/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

Notes from papers I'm reading, ordered by topic and chronologically.

* [NLP](#nlp)
    - [Embeddings](#embeddings)
    - [Architectures](#architectures)
    - [Frameworoks](#frameworks)
    - [Datasets](#datasets)
    - [Named entity recognition NER](#ner)
    - [Sarcasm detection](#sarcasm-detection)
    - [Text summarization](#text-summarization)
    - [Machine translation](#machine-translation)
* [Reinforcement learning](#reinforcement-learning)
* [Computer vision](#computer-vision)
* [Machine learning](#machine-learning)
* [Audio](#audio)
* [Linguistics](#linguistics)
* [Social sciences](#social-sciences)
* [Humanities](#humanities)
* [Economics](#economics)
* [Physics](#physics)
* [Neuroscience](#neuroscience)
* [Algorithms](#algorithms)

## NLP

1. What’s Going On in Neural Constituency Parsers? An Analysis, Gaddy et al., 2018 [[Paper](https://arxiv.org/abs/1804.07853)] [[Notes](2018/1804.07853.md)] [\#nlp](#nlp)
2. Two Methods for Domain Adaptation of Bilingual Tasks: Delightfully Simple and Broadly Applicable, Hangya et al., 2018 [[Paper](https://www.aclweb.org/anthology/P18-1075)] [[Notes](2018/1807.1075.md)] [\#nlp](#nlp)
3. What do you learn from context? Probing for sentence structure in contextualized word representations, Tenney et al., 2019 [[Paper](https://openreview.net/forum?id=SJzSgnRcKX)] [[Notes](2019/1905.md)] [\#nlp](#nlp)
4. BPE-Dropout: simple and effective subword regularization, Provilkov et al., 2019 [[Paper](https://arxiv.org/abs/1910.13267)] [[Notes](2019/1910.13267.md)] [\#nlp](#nlp)
5. Evaluating NLP models via contrast sets, Gardner et al., 2020 [[Paper](https://arxiv.org/abs/2004.02709)] [[Notes](2020/2004.02709.md)] [\#nlp](#nlp)
6. Byte Pair Encoding is Suboptimal for Language Model Pretraining, Bostrom et al., 2020 [[Paper](https://arxiv.org/abs/2004.03720)] [[Notes](2020/2004.03720.md)] [\#nlp](#nlp)
7. Translation artifacts in cross-lingual transfer learning, Artetxe et al., 2020 [[Paper](https://arxiv.org/abs/2004.04721)] [[Notes](2020/2004.04721.md)] [\#nlp](#nlp)
8. Weight poisoning attacks on pre-trained models, Kurita et al., 2020 [[Paper](https://arxiv.org/abs/2004.06660)] [[Notes](2020/2004.06660.md)] [\#nlp](#nlp)
9. SimAlign: High Quality Word Alignments without Parallel Training Data using Static and Contextualized Embeddings, Sabet et al., 2020 [[Paper](https://arxiv.org/abs/2004.08728)] [[Notes](2020/2004.08728.md)] [\#nlp](#nlp)
10. Experience Grounds Language, Bisk et al., 2020 [[Paper](https://arxiv.org/abs/2004.10151)] [[Notes](2020/2004.10151.md)] [\#nlp](#nlp) [\#linguistics](#linguistics)
11. Beyond accuracy: behavioral testing of NLP models with CheckList, Ribeiro et al., 2020 [[Paper](https://arxiv.org/abs/2005.04118)] [[Notes](2020/2005.04118.md)] [\#nlp](#nlp)
12. The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes, Kiela et al., 2020 [[Paper](https://arxiv.org/abs/2005.04790)] [[Notes](2020/2005.04790.md)] [\#nlp](#nlp)
13. The Unstoppable Rise of Computational Linguistics in Deep Learning, Henderson, 2020 [[Paper](https://arxiv.org/abs/2005.06420)] [[Notes](2020/2005.06420.md)] [\#nlp](#nlp) [\#linguistics](#linguistics)
14. Language (Technology) is Power: A Critical Survey of "Bias" in NLP, Blodgett et al., 2020 [[Paper](https://arxiv.org/abs/2005.14050)] [[Notes](2020/2005.14050.md)] [\#nlp](#nlp)
15. Representation Learning for Information Extraction from Form-like Documents, Majumder et al., 2020 [[Paper](https://www.aclweb.org/anthology/2020.acl-main.580/)] [[Notes](2020/2007.00580.md)] [\#nlp](#nlp)
16. Learning to tag OOV tokens by integrating contextual representation and background knowledge, He et al., 2020 [[Paper](https://www.aclweb.org/anthology/2020.acl-main.58/)] [[Notes](2020/2007.58.md)] [\#nlp](#nlp)
17. It's not just size that matters, small language models are also few-shot learners, Schick and Schütze, 2020 [[Paper](https://arxiv.org/abs/2009.07118)] [[Notes](2020/2009.07118.md)] [\#nlp](#nlp)
18. Did you read the next episode? Using textual cues for predicting podcast popularity, Joshi et al., 2020 [[Paper](https://drive.google.com/file/d/1fPwzroOnWXRD91jYB9RybaueIB3W4P9T/view)] [[Notes](2020/2010.18.md)] [\#nlp](#nlp)
19. A Survey on Recent Approaches for Natural Language Processing in Low-Resource Scenarios, Hedderich et al., 2020 [[Paper](https://arxiv.org/abs/2010.12309)] [[Notes](2020/2010.12309.md)] [\#nlp](#nlp)
20. Adapting Coreference Resolution to Twitter Conversations, Aktas et al., 2020 [[Paper](https://www.aclweb.org/anthology/2020.findings-emnlp.222/)] [[Notes](2020/2011.222.md)] [\#nlp](#nlp)
21. Learning from others' mistakes: avoiding dataset biases without modeilng them, Sanh et al., 2020 [[Paper](https://arxiv.org/abs/2012.01300)] [[Notes](2020/2012.01300.md)] [\#nlp](#nlp)

### Embeddings

1. Semi-supervised sequence tagging with bidirectional language models, Peters et al., 2017 [[Paper](https://arxiv.org/abs/1705.00108)] [[Notes](-2017/1705.00108.md)] [\#nlp](#nlp) [\#embeddings](#embeddings)
2. Mimicking Word Embeddings using Subword RNNs, Pinter et al., 2017 [[Paper](https://www.aclweb.org/anthology/D17-1010/)] [[Notes](-2017/1709.1010.md)] [\#nlp](#nlp) [\#embeddings](#embeddings)
3. Deep contextualized word representations, Peters et al., 2018 [[Paper](https://arxiv.org/abs/1802.05365)] [[Notes](2018/1802.05365.md)] [\#nlp](#nlp) [\#embeddings](#embeddings)
4. Linguistic Knowledge and Transferability of Contextual Representations, Liu et al., 2019 [[Paper](https://arxiv.org/abs/1903.08855)] [[Notes](2019/1903.08855.md)] [\#nlp](#nlp) [\#embeddings](#embeddings)
5. Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates, Kudo, 2018 [[Paper](https://www.aclweb.org/anthology/P18-1007/)] [[Notes](2018/1807.1007.md)] [\#nlp](#nlp) [\#embeddings](#embeddings)
6. Dissecting contextual word embeddings: architecture and representation, Peters et al., 2018 [[Paper](https://arxiv.org/abs/1808.08949)] [[Notes](2018/1808.08949.md)] [\#nlp](#nlp) [\#embeddings](#embeddings)
7. BERT: Pre-training of deep bidirectional transformers for language understanding, Devlin et al., 2018 [[Paper](https://arxiv.org/abs/1810.04805)] [[Notes](2018/1810.04805.md)] [\#nlp](#nlp) [\#embeddings](#embeddings)
8. Learning Semantic Representations for Novel Words: Leveraging Both Form and Context, Schick et al., 2018 [[Paper](https://arxiv.org/abs/1811.03866)] [[Notes](2018/1811.03866.md)] [\#nlp](#nlp) [\#embeddings](#embeddings)
9. Wikipedia2Vec: An Efficient Toolkit for Learning and Visualizing the Embeddings of Words and Entities from Wikipedia, Yamada et al., 2018 [[Paper](https://arxiv.org/abs/1812.06280)] [[Notes](2018/1812.06280.md)] [\#nlp](#nlp) [\#embeddings](#embeddings)
10. Rare Words: A Major Problem for Contextualized Embeddings and How to Fix it by Attentive Mimicking, Schick et al., 2019 [[Paper](https://www.aclweb.org/anthology/N19-1048/)] [[Notes](2019/1906.1048.md)] [\#nlp](#nlp) [\#embeddings](#embeddings)
11. Attentive Mimicking: Better Word Embeddings by Attending to Informative Contexts, Schick et al., 2019 [[Paper](https://arxiv.org/abs/1904.06707)] [[Notes](2019/1904.06707.md)] [\#nlp](#nlp) [\#embeddings](#embeddings)
12. BERTRAM: Improved Word Embeddings Have Big Impact on Contextualized Model Performance, Schick et al., 2019 [[Paper](https://arxiv.org/abs/1910.07181)] [[Notes](2019/1910.07181.md)] [\#nlp](#nlp) [\#embeddings](#embeddings)
13. BERT is Not a Knowledge Base (Yet): Factual Knowledge vs. Name-Based Reasoning in Unsupervised QA, Poerner et al., 2019 [[Paper](https://arxiv.org/abs/1911.03681)] [[Notes](2019/1911.03681.md)] [\#nlp](#nlp) [\#embeddings](#embeddings)

### Architectures

1. Conditional Random Fields: probabilistic models for segmenting and labeling sequence data, Lafferty et al, 2001 [[Paper](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)] [[Notes](-2017/0106.md)] [\#nlp](#nlp) [\#architectures](#architectures)
2. Bidirectional LSTM-CRF Models for sequence tagging, Huang et al., 2015 [[Paper](https://arxiv.org/abs/1508.01991)] [[Notes](-2017/1508.01991.md)] [\#nlp](#nlp) [\#architectures](#architectures)
3. Neural Architectures for Named Entity Recognition, Lample et al., 2016 [[Paper](https://www.aclweb.org/anthology/N16-1030)] [[Notes](-2017/1606.1030.md)] [\#nlp](#nlp) [\#architectures](#architectures) [\#NER](#ner)
4. Named Entity Recognition with Bidirectional LSTM-CNNs, Chiu et al., 2016 [[Paper](https://www.aclweb.org/anthology/16-1026)] [[Notes](-2017/1607.1026.md)] [\#nlp](#nlp) [\#architectures](#architectures)
5. Attention is all you need, Vaswani et al., 2018 [[Paper](https://arxiv.org/abs/1706.03762)] [[Notes](-2017/1706.03762.md)] [\#nlp](#nlp) [\#architectures](#architectures)
6. Reasoning with Sarcasm by Reading In-between, Tay et al., 2018 [[Paper](https://www.aclweb.org/anthology/P18-1093/)] [[Notes](2018/1807.1093.md)] [\#sarcasm-detection](#sarcasm-detection) [\#architectures](#architectures)
7. XLNet: generalized autoregressive pretraining for language understanding, Yang et al., 2019 [[Paper](https://arxiv.org/abs/1906.08237)] [[Notes](2019/1906.08237.md)] [\#nlp](#nlp) [\#architectures](#architectures)
8. R-Transformer: Recurrent Neural Network Enhanced Transformer, Wang et al., 2019 [[Paper](https://arxiv.org/abs/1907.05572)] [[Notes](2019/1907.05572.md)] [\#nlp](#nlp) [\#architectures](#architectures)
9. Generalization through Memorization: Nearest Neighbor Language Models, Khandelwal et al., 2019 [[Paper](https://arxiv.org/abs/1911.001723)] [[Notes](2019/1911.001723.md)] [\#nlp](#nlp) [\#architectures](#architectures)
10. Single Headed Attention RNN: Stop Thinking With Your Head, Merity, 2019 [[Paper](https://arxiv.org/abs/1911.11423)] [[Notes](2019/1911.11423.md)] [\#nlp](#nlp) [\#architectures](#architectures)
11. A Transformer-based approach to Irony and Sarcasm detection, Potamias et al., 2019 [[Paper](https://arxiv.org/abs/1911.10401)] [[Notes](2019/1911.10401.md)] [\#sarcasm-detection](#sarcasm-detection) [\#architecture](#architecture)
12. ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training, Qi et al., 2020 [[Paper](https://arxiv.org/abs/2001.04063)] [[Notes](2020/2001.04063.md)] [\#nlp](#nlp) [\#architectures](#architectures)
13. Pre-trained Models for Natural Language Processing: A Survey, Qiu et al., 2020 [[Paper](https://arxiv.org/abs/2003.08271)] [[Notes](2020/2003.08271.md)] [\#nlp](#nlp) [\#architectures](#architectures)
13. SqueezeBERT: What can computer vision teach NLP about efficient neural networks?, Iandola et al., 2020 [[Paper](https://arxiv.org/abs/2006.11316)] [[Notes](2020/2006.11316.md)] [\#nlp](#nlp) [\#architectures](#architectures) [\#computer-vision](#computer-vision)
14. A comparison of LSTM and BERT for small corpus, Ezen-Can, 2020 [[Paper](https://arxiv.org/abs/2009.05451)] [[Notes](2020/2009.05451.md)] [\#nlp](#nlp) [\#architectures](#architectures)

### Frameworks

1. Flair: an easy-to-use framework for stat-of-the-art NLP [[Paper](https://www.aclweb.org/anthology/N19-4010)] [[Notes](2019/1906.4010.md)] [\#nlp](#nlp) [\#frameworks](#frameworks)
2. HuggingFace's Transformers: State-of-the-art Natural Language Processing, Wolf et al., 2019 [[Paper](https://arxiv.org/abs/1910.03771)] [[Notes](2019/1910.03771.md)] [\#nlp](#nlp) [\#frameworks](#frameworks)
3. Selective Brain Damage: Measuring the Disparate Impact of Model Pruning, Hooker et al., 2019 [[Paper](https://arxiv.org/abs/1911.05248)] [[Notes](2019/1911.05248.md)] [\#frameworks](#frameworks)
4. Why should we add early exits to neural networks?, Scardapane et al., 2020 [[Paper](https://arxiv.org/abs/2004.12814)] [[Notes](2020/2004.12814.md)] [\#frameworks](#frameworks)

### Datasets

1. Introduction to the CoNLL-2003 shared task: language-independent named entity recognition, Sang et al., 2003 [[Paper](https://dl.acm.org/citation.cfm?id=1119195)] [[Notes](-2017/0306.md)] [\#nlp](#nlp) [\#datasets](#datasets)
2. Datasheets for datasets, Gebru et al., 2018 [[Paper](https://arxiv.org/abs/1803.09010)] [[Notes](2018/1803.09010.md)] [\#nlp](#nlp) [\#datasets](#datasets)
3. SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference, Zellers et al., 2018 [[Paper](https://arxiv.org/abs/1808.05326)] [[Notes](2018/1808.05326.md)] [\#nlp](#nlp) [\#datasets](#datasets)
4. A Named Entity Recognition Shootout for German, Riedl and Padó, 2018 [[Paper](https://www.aclweb.org/anthology/P18-2020)] [[Notes](2018/1807.2020.md)] [\#nlp](#nlp) [\#NER](#ner) [\#datasets](#datasets)
5. Probing Neural Network Comprehension of Natural Language Arguments, Nivel and Kao, 2019 [[Paper](https://arxiv.org/abs/1907.07355)] [[Notes](2019/1907.07355.md)] [\#nlp](#nlp) [\#datasets](#datasets)
6. Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference., McCoy et al., 2019 [[Paper](https://arxiv.org/abs/1902.01007)] [[Notes](2019/1902.01007.md)] [\#nlp](#nlp) [\#linguistics](#linguistics) [\#datasets](#datasets)
7. UR-FUNNY: A Multimodal Language Dataset for Understanding Humor, Hasan et al., 2019 [[Paper](https://arxiv.org/abs/1904.06618)] [[Notes](2019/1904.06618.md)] [\#sarcasm-detection](#sarcasm-detection) [\#datasets](#datasets)
8. HellaSwag: Can a Machine Really Finish Your Sentence?, Zellers et al., 2019 [[Paper](https://arxiv.org/abs/1905.07830)] [[Notes](2019/1905.07830.md)] [\#nlp](#nlp) [\#datasets](#datasets)
9. Sentiment analysis is not solved! Assessing and probing sentiment classification, Barnes et al., 2019 [[Paper](https://arxiv.org/abs/1906.05887)] [[Notes](2019/1906.05887.md)] [\#nlp](#nlp) [\#datasets](#datasets)
10. Multi-Modal Sarcasm Detection in Twitter with Hierarchical Fusion Model, Cai et al., 2019 [[Paper](https://www.aclweb.org/anthology/P19-1239/)] [[Notes](2019/1907.1239.md)] [\#sarcasm-detection](#sarcasm-detection) [\#datasets](#datasets)
11. Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper), Castro et al., 2019 [[Paper](https://www.aclweb.org/anthology/P19-1455/)] [[Notes](2019/1907.1455.md)] [\#sarcasm-detection](#sarcasm-detection) [\#datasets](#datasets)
12. iSarcasm: A Dataset of Intended Sarcasm, Oprea et al., 2019 [[Paper](https://arxiv.org/abs/1911.03123)] [[Notes](2019/1911.03123.md)] [\#datasets](#datasets) [\#sarcasm-detection](#sarcasm-detection)
13. Lessons from archives: strategies for collecting sociocultural data in machine learning, Seo Jo and Gebru, 2019 [[Paper](https://arxiv.org/abs/1912.10389)] [[Notes](2019/1912.10389.md)] [\#nlp](#nlp) [\#datasets](#datasets)
14. BERTweet: A pre-trained language model for English Tweets, Nguyen et al., 2020 [[Paper](https://arxiv.org/abs/2005.10200)] [[Notes](2020/2005.10200.md)] [\#nlp](#nlp) [\#datasets](#datasets)
15. GAIA: a fine-grained multimedia knowlege extraction system, Li et al., 2020 [[Paper](https://www.aclweb.org/anthology/2020.acl-demos.11/), [[Notes](2020/2007.11.md)] [\#nlp](#nlp) [\#datasets](#datasets)
16. It's morphin' time! Combating linguistic discrimination with inflectional perturbations, Tan et al., 2020 [[Paper](https://www.aclweb.org/anthology/2020.acl-main.263), [[Notes](2020/2007.00263.md)] [\#nlp](#nlp) [\#datasets](#datasets)
17. Reactive Supervision: A New method for Collecting Sarcasm Data, Shmueli et al, 2020 [[Paper](https://arxiv.org/abs/2009.13080)] [[Notes](2020/2009.13080.md)] [\#datasets](#datasets) [\#sarcasm-detection](#sarcasm-detection)


### NER

1. Introduction to the CoNLL-2003 shared task: language-independent named entity recognition, Sang et al., 2003 [[Paper](https://dl.acm.org/citation.cfm?id=1119195)] [[Notes](-2017/0306.md)] [\#nlp](#nlp) [\#datasets](#datasets) [\#NER](#ner)
2. Neural Architectures for Named Entity Recognition, Lample et al., 2016 [[Paper](https://www.aclweb.org/anthology/N16-1030)] [[Notes](-2017/1606.1030.md)] [\#nlp](#nlp) [\#architectures](#architectures) [\#NER](#ner)
3. Named Entity Recognition with Bidirectional LSTM-CNNs, Chiu et al., 2016 [[Paper](https://www.aclweb.org/anthology/Q16-1026)] [[Notes](-2017/1607.1026.md)] [\#nlp](#nlp) [\#architectures](#architectures) [\#NER](#ner)
4. Towards Robust Named Entity Recognition for Historic German, Schweter et al., 2019 [[Paper](https://arxiv.org/abs/1906.07592)] [[Notes](2019/1906.07592.md)] [\#nlp](#nlp) [\#NER](#ner)
5. A Named Entity Recognition Shootout for German, Riedl and Padó, 2018 [[Paper](https://www.aclweb.org/anthology/P18-2020)] [[Notes](2018/1807.2020.md)] [\#nlp](#nlp) [\#NER](#ner) [\#datasets](#datasets)

### Sarcasm detection

[summary](sarcasm_detection.md)

1. Sarcasm Detection on Twitter: A Behavioral Modeling Approach, Rajadesingan et al., 2015 [[Paper](https://dl.acm.org/citation.cfm?id=2685316)] [[Notes](-2017/1502.md)] [\#sarcasm-detection](#sarcasm-detection)
2. Contextualized Sarcasm Detection on Twitter, Bamman and Smith, 2015 [[Paper](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM15/paper/viewPaper/10538)] [[Notes](-2017/1504.md)] [\#sarcasm-detection](#sarcasm-detection)
3. Harnessing Context Incongruity for Sarcasm Detection, Joshi et al., 2015 [[Paper](https://www.aclweb.org/anthology/P15-2124/)] [[Notes](-2017/1507.2124.md)] [\#linguistics](#linguistics) [\#sarcasm-detection](#sarcasm-detection)
4. Automatic Sarcasm Detection: A Survey, Joshi et al., 2017 [[Paper](https://dl.acm.org/citation.cfm?id=3124420)] [[Notes](-2017/1602.03426.md)] [\#sarcasm-detection](#sarcasm-detection)
5. Detecting Sarcasm is Extremely Easy ;-), Parde and Nielsen, 2018 [[Paper](https://www.aclweb.org/anthology/W18-1303/)] [[Notes](2018/1806.1303.md)] [\#sarcasm-detection](#sarcasm-detection)
6. CASCADE: Contextual Sarcasm Detection in Online Discussion Forums, Hazarika et al., 2018 [[Paper](https://arxiv.org/abs/1805.06413)] [[Notes](2018/1805.06413.md)] [\#sarcasm-detection](#sarcasm-detection)
7. Reasoning with Sarcasm by Reading In-between, Tay et al., 2018 [[Paper](https://www.aclweb.org/anthology/P18-1093/)] [[Notes](2018/1807.1093.md)] [\#sarcasm-detection](#sarcasm-detection) [\#architectures](#architectures)
8. Tweet Irony Detection with Densely Connected LSTM and Multi-task Learning, Wu et al., 2018 [[Paper](https://www.aclweb.org/anthology/S18-1006/)] [[Notes](2018/1806.1006.md)] [\#sarcasm-detection](#sarcasm-detection)
9. UR-FUNNY: A Multimodal Language Dataset for Understanding Humor, Hasan et al., 2019 [[Paper](https://arxiv.org/abs/1904.06618)] [[Notes](2019/1904.06618.md)] [\#sarcasm-detection](#sarcasm-detection) [\#datasets](#datasets)
10. Exploring Author Context for Detecting Intended vs Perceived Sarcasm, Oprea and Magdy, 2019 [[Paper](https://www.aclweb.org/anthology/P19-1275/)] [[Notes](2019/1907.1275.md)] [\#sarcasm-detection](#sarcasm-detection)
11. Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper), Castro et al., 2019 [[Paper](https://www.aclweb.org/anthology/P19-1455/)] [[Notes](2019/1907.1455.md)] [\#sarcasm-detection](#sarcasm-detection) [\#datasets](#datasets)
12. Multi-Modal Sarcasm Detection in Twitter with Hierarchical Fusion Model, Cai et al., 2019 [[Paper](https://www.aclweb.org/anthology/P19-1239/)] [[Notes](2019/1907.1239.md)] [\#sarcasm-detection](#sarcasm-detection) [\#datasets](#datasets)
13. A2Text-Net: A Novel Deep Neural Network for Sarcasm Detection, Liu et al., 2019 [[Paper](https://www.researchgate.net/profile/Liyuan_Liu23/publication/337425314_A2Text-Net_A_Novel_Deep_Neural_Network_for_Sarcasm_Detection/links/5dd6bd1d458515dc2f41db91/A2Text-Net-A-Novel-Deep-Neural-Network-for-Sarcasm-Detection.pdf)] [[Notes](2019/1912.33742.md)] [\#sarcasm-detection](#sarcasm-detection)
14. Sarcasm detection in tweets, Rajagopalan et al., 2019 [[Paper](https://jadhosn.github.io/projects/CSE575_FinalReport-SarcasmDetection.pdf)] [[Notes](2019/1911.575.md)] [\#sarcasm-detection](#sarcasm-detection)
15. A Transformer-based approach to Irony and Sarcasm detection, Potamias et al., 2019 [[Paper](https://arxiv.org/abs/1911.10401)] [[Notes](2019/1911.10401.md)] [\#sarcasm-detection](#sarcasm-detection) [\#architecture](#architecture)
16. Deep and dense sarcasm detection, Pelser et al., 2019 [[Paper](https://arxiv.org/abs/1911.07474)] [[Notes](2019/1911.07474.md)] [\#sarcasm-detection](#sarcasm-detection)
17. iSarcasm: A Dataset of Intended Sarcasm, Oprea et al., 2019 [[Paper](https://arxiv.org/abs/1911.03123)] [[Notes](2019/1911.03123.md)] [\#datasets](#datasets) [\#sarcasm-detection](#sarcasm-detection)
18. Reactive Supervision: A New method for Collecting Sarcasm Data, Shmueli et al, 2020 [[Paper](https://arxiv.org/abs/2009.13080)] [[Notes](2020/2009.13080.md)] [\#datasets](#datasets) [\#sarcasm-detection](#sarcasm-detection)


### Text summarization

1. Evaluating the Factual Consistency of Abstractive Text Summarization, Kryscinski et al., 2019 [[Paper](https://arxiv.org/abs/1910.12840)] [[Notes](2019/1910.12840.md)] [\#nlp](#nlp) [\#text-summarization](#text-summarization)
2. A survey on text simplification, Sikka and Mago, 2020 [[Paper](https://arxiv.org/abs/2008.08612)] [[Notes](2020/2008.08612.md)] [\#nlp](#nlp) [\#text-summarization](#text-summarization)

### Machine translation

1. Unsupervised Tokenization for Machine Translation, Chung and Gildea, 2009 [[Paper](https://www.aclweb.org/anthology/D09-1075/)] [[Notes](-2017/091075.md)] [\#nlp](#nlp) [\#machine-translation](#machine-translation)
2. Neural Machine Translation of Rare Words with Subword Units, Sennrich et al., 2015 [[Paper](https://arxiv.org/abs/1508.07909)] [[Notes](-2017/1508.07909.md)] [\#nlp](#nlp) [\#machine-translation](#machine-translation)
3. Unsupervised neural machine translation, Artetxe et al., 2017 [[Paper](https://arxiv.org/abs/1710.11041)] [[Notes](2017/1710.11041.md)] [\#nlp](#nlp) [\#machine-translation](#machine-translation)
4. How Much Does Tokenization Affect Neural Machine Translation? Domingo et al., 2018 [[Paper](https://arxiv.org/abs/1812.08621)] [[Notes](2018/1812.08621.md)] [\#nlp](#nlp) [\#machine-translation](#machine-translation)
5. Reusing a Pretrained Language Model on Languages with Limited Corpora for Unsupervised NMT, Chronopoulou et al., 2020 [[Paper](https://www.aclweb.org/anthology/2020.emnlp-main.214/)] [[Notes](2020/2011.214.md)] [\#nlp](#nlp) [\#machine-translation](#machine-translation)

---

## Reinforcement learning

1. Theory of Minds: Understanding Behavior in Groups Through Inverse Planning, Shum et al., 2019 [[Paper](https://arxiv.org/abs/1901.06085)] [[Notes](2019/1901.06085.md)] [\#reinforcement-learning](#reinforcement-learning) [\#social-sciences](#social-sciences)
2. The Hanabi Challenge: A New Frontier for AI Research, Bard et al., 2019 [[Paper](https://arxiv.org/abs/1902.00506)] [[Notes](2019/1902.00506.md)] [\#reinforcement-learning](#reinforcement-learning)
3. Mastering Atari, Go, Chess and Shogi by Planning with a learned model, Schrittwieser et al., 2019 [[Paper](https://arxiv.org/abs/1911.08265)] [[Notes](2019/1911.08265.md)] [\#reinforcement-learning](#reinforcement-learning)
4. Language as a cognitive tool to imagine goals in curiosity-driven exploration, Colas et al., 2020 [[Paper](https://arxiv.org/abs/2002.09253)] [[Notes](2020/2002.09253.md)] [\#reinforcement-learning](#reinforcement-learning)
5. Planning to Explore via Self-Supervised World Models, Sekar et al., 2020 [[Paper](https://arxiv.org/abs/2005.05960)] [[Notes](2020/2005.05960.md)] [\#reinforcement-learning](#reinforcement-learning)

---

## Computer vision

1. Cubic Stylization, Derek Liu and Jacobson, 2019 [[Paper](https://arxiv.org/abs/1910.02926)] [[Notes](2019/1910.02926.md)] [\#computer-vision](#computer-vision)
2. SqueezeBERT: What can computer vision teach NLP about efficient neural networks?, Iandola et al., 2020 [[Paper](https://arxiv.org/abs/2006.11316)] [[Notes](2020/2006.11316.md)] [\#nlp](#nlp) [\#computer-vision](#computer-vision)

---

## Machine learning

1. Gender shades: intersectional accuracy disparities in commercial gender classification, Buolamwini and Gebru, 2018 [[Paper](http://proceedings.mlr.press/v81/buolamwini18a.html)] [[Notes](2018/1802.md)] [\#machine-learning](#machine-learning)
3. Interpretable Machine Learning - A Brief History, State-of-the-Art and Challenges, Molnar et al., 2020 [[Paper](https://arxiv.org/abs/2010.09337)] [[Notes](2020/2010.09337.md)] [\#machine-learning](#machine-learning)

## Audio

1. End-to-End Adversarial Text-to-Speech, Donahue et al., 2020 [[Paper](https://arxiv.org/abs/2006.03575)] [[Notes](2020/2006.03575.md)] [\#audio](#audio)
2. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations, Baevski et al., 2020 [[Paper](https://arxiv.org/abs/2006.11477)] [[Notes](2020/2006.11477.md)] [\#audio](#audio)

---

## Linguistics

1. Moving beyond the plateau: from lower-intermediate to upper-intermediate, Richards, 2015 [[Paper](https://www.cambridge.org/elt/blog/2015/08/26/moving-beyond-plateau-lower-upper-intermediate/)] [[Notes](-2017/1508.md)] [\#linguistics](#linguistics)
2. Harnessing Context Incongruity for Sarcasm Detection, Joshi et al., 2015 [[Paper](https://www.aclweb.org/anthology/P15-2124/)] [[Notes](-2017/1507.2124.md)] [\#linguistics](#linguistics) [\#sarcasm-detection](#sarcasm-detection)
3. A Trainable Spaced Repetition Model for Language Learning, Settles and Meeder, 2016 [[Paper](https://www.aclweb.org/anthology/P16-1174/)] [[Notes](-2017/1608.1174.md)] [\#linguistics](#linguistics)
4. Targeted synctactic evaluation of language models, Marvin and Linzen, 2018 [[Paper](https://arxiv.org/abs/1808.09031)] [[Notes](2018/1808.09031.md)] [\#nlp](#nlp) [\#linguistics](#linguistics)
5. Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference., McCoy et al., 2019 [[Paper](https://arxiv.org/abs/1902.01007)] [[Notes](2019/1902.01007.md)] [\#nlp](#nlp) [\#linguistics](#linguistics) [\#datasets](#datasets)
6. Language Models as Knowledge Bases?, Petroni et al., 2019 [[Paper](https://arxiv.org/abs/1909.01066)] [[Notes](2019/1909.01066.md)] [\#nlp](#nlp) [\#linguistics](#linguistics)
7. Different languages, similar encoding efficiency: Comparable information rates across the human communicative niche, Coupé et al., 2019 [[Paper](https://advances.sciencemag.org/content/5/9/eaaw2594)] [[Notes](2019/190904.md)] [\#linguistics](#linguistics) [\#social-sciences](#social-sciences)
8. My English sounds better than yours: Second language learners perceive their own accent as better than that of their peers, Mittlerer et al., 2020 [[Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0227643)] [[Notes](2020/2001.0227643.md)] [\#linguistics](#linguistics)
9. Experience Grounds Language, Bisk et al., 2020 [[Paper](https://arxiv.org/abs/2004.10151)] [[Notes](2020/2004.10151.md)] [\#nlp](#nlp) [\#linguistics](#linguistics)
10. The Unstoppable Rise of Computational Linguistics in Deep Learning, Henderson, 2020 [[Paper](https://arxiv.org/abs/2005.06420)] [[Notes](2020/2005.06420.md)] [\#nlp](#nlp) [\#linguistics](#linguistics)
11. Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data, Bender et al., 2020 [[Paper](https://www.aclweb.org/anthology/2020.acl-main.463/)] [[Notes](2020/2005.00463.md)] [\#nlp](#nlp) [\#linguistics](#linguistics)

---

## Social sciences

1. Antisocial Behavior in Online Discussion Communities, Cheng et al., 2015 [[Paper](https://arxiv.org/abs/1504.00680)] [[Notes](-2017/1504.00680.md)] [\#social-sciences](#social-sciences)
2. How much does education improve intelligence? A meta-analysis, Ritchie et al., 2017 [[Paper](https://journals.sagepub.com/doi/abs/10.1177/0956797618774253)] [[Notes](-2017/1711.md)] [\#social-sciences](#social-sciences)
3. Theory of Minds: Understanding Behavior in Groups Through Inverse Planning, Shum et al., 2019 [[Paper](https://arxiv.org/abs/1901.06085)] [[Notes](2019/1901.06085.md)] [\#reinforcement-learning](#reinforcement-learning) [\#social-sciences](#social-sciences)
4. Fake news game confers psychological resistance against online misinformation, Roozenbeek and van der Linden, 2019 [[Paper](https://www.nature.com/articles/s41599-019-0279-9)] [[Notes](2019/1908.md)] [\#social-sciences](#social-sciences) [\#humanities](#humanities)
5. Different languages, similar encoding efficiency: Comparable information rates across the human communicative niche, Coupé et al., 2019 [[Paper](https://advances.sciencemag.org/content/5/9/eaaw2594)] [[Notes](2019/190904.md)] [\#linguistics](#linguistics) [\#social-sciences](#social-sciences)
6. Kids these days: Why the youth of today seem lacking, Protzko and Schooler, 2019 [[Paper](https://advances.sciencemag.org/content/5/10/eaav5916)] [[Notes](2019/1910.5916.md)] [\#social-sciences](#social-sciences)

---

## Humanities

1. Fake news game confers psychological resistance against online misinformation, Roozenbeek and van der Linden, 2019 [[Paper](https://www.nature.com/articles/s41599-019-0279-9)] [[Notes](2019/1908.md)] [\#social-sciences](#social-sciences) [\#humanities](#humanities)

---

## Economics

1. Why do people stay poor? Balboni et al., 2020 [[Paper](https://economics.mit.edu/faculty/cbalboni/research)] [[Notes](2020/2003.20.md)] [\#economics](#economics)

---

## Physics

1. First-order transition in a model of prestige bias, Skinner, 2019 [[Paper](https://arxiv.org/abs/1910.05813)] [[Notes](2019/1910.05813.md)] [\#physics](#physics)

---

## Neuroscience

1. A deep learning framework for neuroscience, Richard et al., 2019 [[Paper](https://www.nature.com/articles/s41593-019-0520-2)] [[Notes](2019/1911.41503.md)] [\#neuroscience](#neuroscience)

---

## Algorithms

1. Replace or Retrieve Keywords In Documents At Scale, Singh, 2017 [[Paper](https://arxiv.org/abs/1711.00046)] [[Notes](-2017/1711.00046.md)] [\#algorithms](#algorithms)
