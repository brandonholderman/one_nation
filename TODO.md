The purpose of the One Nation program is to train a model to find similarities in conservative and democratic policies/ideals and highlight them.

Project Goals

- [x] ~~_Consume reddit data_~~ [2024-03-04]
- [ ] Test each NLP to decide which will be used
- [ ] Decide parameters I want to train my model on
- [ ] Analyze data to identify where left and right wing policies/ideals overlap
- [ ] Highlight those similarites and posts
- [ ] Find and link additional articles/content matching those key words
- [ ] **Stretch Goal** Build a web scraper for content related to the findings or feed the information to a LLM or AI image creator
- [ ] **Stretch Goal** Track the engagement of content

Tasks

- [x] ~~_Test NLTK & Vader_~~ [2024-03-05]
- Vader SIA analyzes text by giving it a overall rating based on 3 categories: negative, neutral and positive with a compounded score being an average of all 3.
- [ ] Test Flair NLP

Test Description

- First NLP Tested - Vader SIA analyzes text by giving it a overall rating based on 3 categories: negative, neutral and positive with a compounded score being an average of all 3. From the data I saw, I did not agree with the opinions of the analysis. There was a large portion of titles that didn't appear to be analyzed at all, each with a 1.00 neutral score leading to 0.000 compound score. There were no apparent patterns why this happened. All titles were perfectly readable sentences. In fact, it had no problem scoring some very broken english with multiple mmispelled words. Was highlighted as "The social media analyzer" but the results were underwhelming.
