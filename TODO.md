The purpose of the One Nation program is to train a model to find similarities in conservative and democratic policies/ideals and highlight them.

One Nation is a machine learning project with the purpose of consuming a dataset of politically related reddit posts, to find the topics Democrats and Republicans agree on and highlight them for repost.

Project Goals

- [x] ~~_Consume reddit data_~~ [2024-03-04]
- [x] ~~_Test each NLP to decide which will be used_~~ [2024-03-08]
- [ ] Decide parameters I want to train my model on
- [ ] Analyze data to identify where left and right wing policies/ideals overlap
- [ ] Highlight those similarites and posts
- [ ] Find and link additional articles/content matching those key words
- [ ] **Stretch Goal** Build a web scraper for content related to the findings or feed the information to a LLM or AI image creator
- [ ] **Stretch Goal** Track the engagement of content

Tasks

- [x] ~~_Test NLTK & Vader_~~ [2024-03-05]
- [x] ~~_Test Flair NLP_~~ [2024-03-07]
- [x] ~~_Create a counter or some way to limit the amount of lines analyzed_~~ [2024-03-11]
- [ ] Fine tune training based on defined parameters

Test Description

- First NLP Tested - NLTK with Vader SIA analyzes text by giving it a overall rating based on 3 categories: negative, neutral and positive with a compounded score being an average of all 3. From the data I saw, I did not agree with the opinions of the analysis. There was a large portion of titles that didn't appear to be analyzed at all, each with a 1.00 neutral score leading to 0.000 compound score. There were no apparent patterns why this happened. All titles were perfectly readable sentences. In fact, it had no problem scoring some very broken english with multiple mmispelled words. Was highlighted as "The social media analyzer" but the results were underwhelming.

- Second NLP Tested - Flair sentiment analyzer is much more accurate than NLTK and combined with the simplicity of their functions, seems to be the best fit. Analyzer has the same core function as NLTK but, Flair takes the context of the sentence into account. While not fully accurate without fine tuning, it does appear to produce more accurate analysis of the polarity of a sentence.

Rubber Duck

I want the program to first look at the Politcal Lean column to determine a starting point of which party the post is affiliated with. Then I want to scan the Title to determine the polarity of the sentence.
