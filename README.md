# NumFaith

A numeric-faithfulness stress test for RAG hallucination detectors.

NumFaith takes correct, source-grounded answers from public financial QA data,
programmatically breaks them in controlled ways (swap a number, shift a date,
flip a direction word) so every broken answer is auto-labelled as unfaithful,
then runs off-the-shelf faithfulness detectors over both the clean and broken
answers to measure whether the detectors catch the breaks — with special
attention to numeric and temporal errors.
