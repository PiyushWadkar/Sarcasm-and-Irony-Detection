# Dataset Documentation: SemEval-2018 Task 3 (Subtask B)

## Overview
This project uses the official benchmark dataset from **SemEval-2018 Task 3: "Irony Detection in English Tweets"** (Subtask B: Multi-class Irony Detection), created and curated by Cynthia Van Hee, Els Lefever, and Véronique Hoste (Ghent University).

- **Benchmark**: SemEval-2018 Task 3 (ACL Anthology: https://aclanthology.org/S18-1005/)
- **Repository**: [https://github.com/Cyvhee/SemEval2018-Task3](https://github.com/Cyvhee/SemEval2018-Task3)
- **License**: Public academic research dataset provided under standard SemEval Open Evaluation terms.

---

## Class Definitions and Ground-Truth Mapping

In SemEval-2018 Task 3 Subtask B, annotations classify tweets into four linguistically grounded categories:

| SemEval Class | Original Description | Project Target Class | Linguistic Rationale |
|:---:|:---|:---:|:---|
| **0** | Non-ironic | **Non-Sarcastic / Literal** | Text containing direct, literal statements without figurative reversal. |
| **1** | Verbal irony realized through a polarity contrast | **Sarcastic** | Polarity inversion ("I love when my flight is cancelled"), the textbook definition of sarcasm. |
| **2** | Verbal irony without a polarity contrast | **Ironical** | Situational incongruity or verbal irony where meaning is incongruous rather than pure polarity flip. |
| **3** | Other forms of irony | **Ironical** | Complex figurative irony, situational absurdity, or blended rhetorical figures. |

### Dataset Partitioning:
- **Training Set (`train`)**: 3,817 samples
  - Non-Sarcastic / Literal (0): 1,916
  - Sarcastic (1): 1,383
  - Ironical (2 & 3): 518
- **Official Gold Test Set (`test`)**: 784 samples
  - Non-Sarcastic / Literal (0): 473
  - Sarcastic (1): 164
  - Ironical (2 & 3): 147
- **Total Samples**: 4,601 verified samples

---

## Known Limitations and Ethical Considerations
1. **Domain Bias**: Sourced primarily from social commentary, tweets, and online discussions. Performance on formal essays or biomedical text may differ.
2. **Context Absence**: Sarcasm often hinges on real-world shared knowledge or facial expressions not captured in isolated text strings.
3. **Imbalance**: Pure situational irony (Class 2/3) is rarer than polarity-clash sarcasm (Class 1) and literal text (Class 0), reflecting natural linguistic distribution.
