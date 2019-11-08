# Sarcasm detection summary 2009-2019

Breakthroughs:

* Speech based (-2009)
* Text, rule-based (2009-2015)
* Deep learning (2015-2018)
* Multi-modal (2018-)

## Research

Datasets or models.

Data usually from twitter, Ptacek and Riloff. Reddit, forums, more longer text, now multi-modal, video, audio, text transcript.

## 1. Rule-based

History of sarcasm detection:

* 2006: sarcasm recognition in speech using average pitch, pitch slope and laughter cues
* 2009: simple linguistic features like interjection, changed names
* 2010: syntactic and pattern-based features
* 2011: study on the role of unigrams and emoticons
* 2013: sarcasm detection based on hashtags
* 2013: rule-based system using phrases of positive verb phrases and negative situations from tweets.

Riloff et al., 2013, state that sarcasm is a contrast between positive sentiment word and a negative situation.

R. González-Ibáñez, S. Muresan, and N. Wacholder. Identifying sarcasm in twitter: A closer look. In ACL (Short Papers), pages 581–586. Citeseer, 2011

## 1502.md

Definition: Sarcasm, while similar to irony, differs in that it's usually viewed as being caustic and derisive. Some researchers even consider it to be aggressive humor and a form of verbal aggression. Oxford dictionary's definition: way of using words that are the opposite of what you mean in order to be unpleasant to somebody or to make fun of them.

SCUBA: Behavioral modeling framework tuned for detecting sarcasm from a user's past tweets. We use psychological and behavioral studies.

Data: 9.1k tweets dividing the tweets with #sarcasm and #not.
Features: Sarcasm can aim to do 5 things, we extract features for all:

1. Sarcasm as a contrast of sentiments
2. Sarcasm as a complex form of expression
3. Sarcasm as a means of conveying emotion
4. Sarcasm as a possible function of familiarity
5. Sarcasm as a form of written expression

Models: decision trees, logistic regression and SVM. 

Evaluation: accuracies range from 78 to 83%.

Conclusion: limited historical information can greatly help improve sarcasm detection.

## 1504.md

We include extra-linguistic information form the context of a tweet vs. purely linguistic: info about the author, relationship to audience, context.

Data: 9.7k tweets using #sarcasm and #sarcastic for positive sarcasm, for negative, same amount. 19.5k balanced set.

Model: binary logistic regression with 5-fold cross-validation

Features: tweet itself, author, audience, tweet it's responding to. bigrams, pronunciation, sentiment, pos, author's historical topics, historical sentiment, etc.

Evaluation: Tweet only yields 75.4%, adding the other features: 85.1%.

Conclusion: 

* there's an effect on the interaction of the author and audience in the recognition of sarcasm
* #sarcasm is not used between friends, but rather is a signal to the author's intent to an unknown audience

## 1507.md

We explore context incongruity (incompatibility), explicit and implicit, and two data sources: tweets and decision forum posts.

Definition: Sarcasm is defined as: a cutting, often ironic remark intended to express contempt or ridicule (source: the free dictionary). 

Past work: rule-based, statistical approaches, unigrams, hashtag-based sentiment, positive verb being followed by negative sentiment. None of these are based on linguistic theories so we use one: context incongruity.

Campbell and Katz (2012) state that sarcasm occurs along different dimensions, namely failed expectation, pragmatic insincerity, negative tension, presence of a victim and along stylistic components such as emotion words.

incongruity: state of being not in agreement. Ivanko and Pexman (2003) state that the sarcasm processing time depends on the degree of context incongruity between the statement and the context.

* Explicit incongruity: overtly expressed through sentiment words of both polarities: I love being ignored
* Implicit incongruity: phrases of implied sentiment: I love this paper so much that I made a doggy bag out of it

Data: 

* 5.2k tweets, 4.1k sarcastic, hashtag-based detection
* 2.2k tweets, 506 sarcastic, manually labeled
* Discussion forum posts, 1.5k posts, 752 sarcastic, manually annotated.

Features: augment tweet's text with lexical features, pragmatic (emoticons, laughter expressions, punctuation), implicit and explicit incongruity.

Model: LibSVM with RBF kernel with 5-fold cross-validation.

Evaluation: F1 score 88%, 5% better than baseline (Riloff et al., 2013)

Conclusion: errors found when incongruity was within text, incongruity due to numbers, dataset granularity, some sentences have non-sarcastic portions, irrelevant features, and politeness wasn't accounted for.

## 1602.03426.md

Survey of past work in automatic sarcasm detection. We observe 3 milestones:

* Semi-supervised pattern extraction to identify implicit sentiment
* hashtag-based supervision
* use of context beyond target text

Definition: Free dictionary: sarcasm is a form of verbal irony intended to express contempt or ridicule. It has an implied negative sentiment, but a positive surface sentiment.

History of sarcasm studies in linguistics (more in paper): failed expectation, pragmatic insincerity, negative tension and presence of a victim.

Most research has been done in English, and most have formulated the problem as a classification problem.

Datasets:

* short text: tweets, manually annotated or hashtag-based supervision.
* long text: reviews and discussion forum posts, most of them multiple labels one of them sarcasm.

Approaches:

* Rule-based
* Statistical: bag of words, lexicon, pragmatic features
* Mostly use SVM, logistic regression, Naive Bayes
* Deep learning: similarity between word embeddings as features, LSTM + DNN

Role of context in sarcasm detection only after 2015: author-specific context, conversation context, topical context (some topics more likely to evolve sarcasm).

Issues:

* Issues with data:
    - hashtag-based can provide large-scale datasets
    - but quality doubtful
* Issues with features
    - many studies deliberate if sentiment can be used as a feature
    - relationship between sentiment and sarcasm
    - culture-specific aspects of sarcasm detection
* Issues with dataset skews
    - most datasets are balanced but that's not the case in real world

## 1806.md

Error detection of domain-general sarcasm detection systems on twitter and amazon product review datasets, 4k tweets and 1k amazon product reviews.

Both datasets are imbalanced and twitter's is hashtag-based.

Errors: more false positives than false negatives.

False negatives: in most cases, sarcasm could be inferred using world knowledge. some didn't convey sarcasm once hashtag was removed.

Fake positives, most used excessive punctuation, mix of positive and negative sentiment.

Recommendation for future:

* World knowledge, frame-semantic resources
* Text normalization: disambiguate compound hashtags, spelling correction

## 1805.06413.md

New model: CASCADE: hybrid approach of both content and context-driven modeling for sarcasm detection in online social media discussions.

Previous work:

* Content-based: standard classification, lexical and pragmatic indicators
    -  Prosodic and spectral cues in spoken dialogue, positive predicates, interjections, emoticons, positive sentoment words in negative situations, etc.
* Context-based: importance of speaker and/or topical information associated to a text, historical posts of user, sentiment, etc.

Features: user embeddings encoding stylommetric and personality features (unsupervised method to get a fixed size vector for each user), and contextual information from the discourse of comments in the forums.

Model: CNN to extract syntactic features, location-invariant local patterns. Baseline models: BoW, CNN, CNN-SVM

Dataset: Reddit corpus, 1M+ examples, 3 variants:

* balanced
* imbalanced, 20-80
* political threads

CASCADE improves all baselines, CNN + user embeddings + contextual discourse features. Context information doesn't work sometimes because our model isn't sequential. User embeddings, misclassification for user with fewer historical posts.

## 1806.1006.md

Dense LSTM and multi-task learning.

Rule-based methods depend on lexicons to identify irony, and traditional ML methods like SVM need manual feature enginnering.

The embedding layer is used to convert the input tweets into a sequence of dense vectors. The POS tag features are one-hot encoded and concatenated with the word embedding vectors.

The last layer outputs the hidden representation of texts, which can be concatenated with the sentiment features and the sentence embedding features. 3 dense layers with ReLU activation are used to predict for 3 different tasks: determining the missing ironic hashtag, identifying ironic or non-ironic, and identifying the irony types (verbal irony, other types of verbal irony, and situational irony).

By using this multi-task learning method, our model can incorporate different information such as the irony hashtags.

## 1904.06618.md

UR-FUNNY: multimodal dataset for understanding humor, a bit off-topic.

Humor is produced through text, vision and acoustic cues. UR-FUNNY focuses on punchline detection. Understanding the unique dependencies across modalities and its impact on humor require knowledge from multimodal language.

Data: TED talks, 1.7k speakers, 400 topics, 8.2k punchlines. Text, acoustic and visual features.

Model: MFN memory fusion network, state-of-the-art model in multimodal language.

Conclusion: models using all modalities outperform models that use only 1-2 modalities. Text is the most important one. The human performance on the UR-FUNNY is 82.5%, and our best model achieves 64.4%, so it's good but there's still a gap.

## 1907.1275.md

Investigation of impact of using author context on textual sarcasm detection. Author context: embedded representation of historical posts. 

Data: twitter, manual labelling and hashtag-based. state-of-the-art performance on the second dataset but not on the first, indicating a difference between intended and perceived sarcasm.

1. Riloff dataset, 700 tweets manually labelled.
2. Ptacek dataset, 27k tweets, hashtag-based.

Definition: Sarcasm is a form of irony that occurs when there's a discrepancy between the literal meaning of an utterance and its intended meaning. This is used to express a form of dissociative attitude towards a previous proposition, often in the form of contempt or derogation.

We try to answer 2 questions:

1. is the user embedding predictive of the sarcastic nature of the tweet?
2. is the predictive power of the embedding on the sarcastic nature of the tweet the same if the tweet is labelled via manual labelling vs. distant supervision?

Previous work:

* local models: only info within the text. linguistic incongruity, positive verb used in negative context, etc
* contextual models: local and contextual info, info about the forum type, user context, historical tweets, personality features.

Baseline: SIARN (#TODO: read paper Tay et al. 2018)

Results: User embeddings show remarkable predictive power on the Ptacek dataset. On the Riloff dataset, however, user embeddings seem to be far less informative.

users seem to have a prior disposition to being either sarcastic or nonsarcastic, which can be deduced from historical behaviour.

lack of coherence between the presence of sarcasm tags and manual annotations in the Riloff dataset suggests that the 2 labelling methods capture distinct phenomena, considering the subjective nature of sarcasm.

state-of-the-art results for distant supervision, but not for manual labelling

## 1907.1455.md

Most work has been done on just text, we want multi-modal. We present multi-modal dataset MUStARD, from TV shows. Audio utterances with sarcasm labels, context of historical utterances in dialogue, and speaker identities. up to 12.9% increase in F1 score.

We show some exapmles where incongruity in sarcasm is evident across different modalities, thus stressing the role of multi-modal approaches.

We get data from YouTube, Friends, TBBT, and negative samples from Friends. 6.4k videos, 350 sarcastic, 6k non-sarcastic.

The vocal tonality of the speaker often indicates sarcasm, text that otherwise looks seemingly straightforward is noticed to contain sarcasm only when the associated voices are heard. 

BERT for text, pitch, intonation, tonal-specific details, and video features from ImageNet.

Overall, the combination of visual and textual signals significantly improves over the unimodal variants, with a relative error rate reduction of up to 12.9%.

In the cases where the bimodal textual and visual model predicts sarcasm but the unimodal textual model fails, the textual component doesn't reveal any explicit sarcasm.

## 1907.1239.md

multimodal sarcasm detection for tweets consisting of text and images. 

Features: text features, image features and image attributes.

Model: biLSTM to obtain representation of the tweet text, inputs are the non-linearly transformed vectors.

Previous work only manually concatenates textual and visual features. We propose a hierarchical fusion model, using 2 ResNets for image features and image attributes respectively.

Models based only on the image or attribute modality don't perform well, while both text models perform much better. With image and attribute modalities, our proposed model correctly detects sarcasm in these tweets.

In future work, we will incorporate other modality such as audio into the sarcasm detection task and also investigate to make use of common sense knowledge in our model.

## 1807.1093.md

Attention-based neural model that looks in-between instead of across words, to model contrast and incongruity. We use 6 benchmark datasets, twitter, reddit, and internet argument corpus.

state-of-the-art sarcasm detection systems use sequential models, with the input document parsed one word at a time. There's no explicit interaction between word pairs, so can't explicitly model contrast, incongruity or juxtaposition of situation, and long-range dependencies.

We want to combine the effectiveness of state-of-the-art recurrent models while harnessing the intuition of looking in-between. We propose a multi-dimensional intra-attention recurrent network, MIARN, that models intricate similarities between each word pair in the sentence.

our model is the first attention model for sarcasm detection. attention layer learns to pay attention based on a word’s largest contribution to all words in the sequence.

This is one-dimensional adaptation, but sometimes words have many meanings, so we represent the word pair with vectors. multi-dimensional adaptation of the intra-attention mechanism.

In the 2 twitter dataset, Ptacek and Riloff, we get different results. MIARM better in Ptacek, SIARM in Riloff. Our model shows more improvement in long text (debates dataset) than in short text, confirming that we capture long-range dependencies.