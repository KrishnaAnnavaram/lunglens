# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **nodule** | One physical lung nodule: one pylidc annotation cluster | lesion, tumour, slice |
| **patch** | A 32 x 32 x 32 cube of HU values around a nodule centre, 1 mm voxels | crop, image, slice |
| **HU** | Hounsfield units, the CT intensity scale | pixel value, intensity |
| **reader** | One radiologist who rated a nodule in LIDC-IDRI | annotator, labeller |
| **rating** | The malignancy value of one reader, 1 to 5 | score, grade |
| **consensus** | The median of the ratings of one nodule | average, mean label |
| **label** | 1 (consensus 4-5), 0 (consensus 1-2). Consensus 3 has no label | target, class |
| **manifest** | The JSONL file with one record per nodule | index, CSV |
| **patient-level split** | A split where all nodules of a patient are on one side | group split |
| **calibration** | The step that makes the output a probability that matches observed rates | scaling |
| **estimate** | The calibrated malignancy probability of the model for one nodule | score, risk, prediction |
| **band** | `lower`, `intermediate`, `higher` or `not_assessed` | risk level, category |
| **report** | The patient text built from the fixed templates | summary, letter |
| **safety notice** | The fixed text that says the result is not a diagnosis | disclaimer |
| **language code** | A FLORES-200 code, for example `spa_Latn` | locale, language id |
| **translation cache** | The JSON file with each translated string | dictionary |
| **FAQ** | The fixed list of questions and reviewed answers | chatbot, knowledge base |
| **reader reference** | AUROC of one reader against the consensus of the other readers | human baseline |
| **model card** | The Markdown file that gives the use, data, results and limits of one model | spec sheet |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **extract** | Cut patches and records from LIDC-IDRI with pylidc |
| **resample** | Change the voxel size to 1 mm |
| **calibrate** | Fit the probability map on held-out patients of the training set |
| **evaluate** | Calculate the metrics on the test patients |
| **localize** | Translate a template and put the numbers back into it |
| **match** | Find the FAQ entry that is nearest to a question |
| **validate** | Check DICOM header fields, patches and records |
