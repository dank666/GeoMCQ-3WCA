# GeoMCQ-3WCA

Code, data and model outputs for the paper "Leveraging Large Language Model and Three-Way Concept Analysis for
Automated Geography Multiple-Choice Question Answering".

For each question, an LLM states the conditions that a correct answer must satisfy and judges every option against
them as satisfied, violated or undetermined, using the knowledge of a curriculum dictionary. The judgments form a
three-valued formal context. The answer is the extent of a three-way concept of that context, and questions that the
three-way regions leave open are decided in a second stage.

## Setup

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m pytest

## Reproducing the results

    sh experiments/reproduce.sh

The script recomputes every result of Section 6 and Appendix C and writes the tables to `results/tables/` and the
figures to `results/figures/` (the figures use LaTeX for their fonts). All model calls are read from the cache in
`results/cache/` and `kb/cache/`, so no API key is needed and the results are identical to those in the paper.

The processed questions can be rebuilt from the raw files with

    .venv/bin/python experiments/prepare_data.py

## Files

- `src/geoqa/twc.py`: the framework (three-valued context, three-way regions, decision, deferral, explanation);
  `prompts_tw.py`: its prompts; `dcr.py`: dynamic concept retrieval; `kb.py`: the curriculum dictionary;
  `grounding.py`: concept linking; `fca.py`: formal contexts, three-way concepts and merged concepts;
  `dictionary_matching.py`: the dictionary-only baseline; `prompts.py`: direct answering; `prompts_kb.py`: dictionary
  construction; `llm.py`: a cached client for OpenAI-compatible APIs; `data.py`: question schema and loaders.
- `experiments/`: `prepare_data.py` (datasets and the development/test split), `extract_curriculum.py` and
  `build_dictionary.py` (dictionary construction), `run_twc.py` (the framework), `run_direct.py` (direct answering),
  `run_dictionary_matching.py` (baseline), `run_ablations.sh`, `reproduce.sh`, `make_results.py` (summary and
  tables) and `plot_figures.py` (Figures 3 to 5).
- `kb/geo_dictionary.jsonl`: the curriculum dictionary (7,685 entries); `kb/seeds/curriculum_requirements.jsonl`: the
  141 content requirements of the *General High School Geography Curriculum Standard (2017 edition)* from which it was
  built.
- `data/raw/geomcq_public/`: the GeoMCQ-1200 dataset, one JSON file per domain for single-choice and for
  multiple-choice questions (1,200 questions in six domains); `data/raw/public/`: the geography questions of AGIEval
  and GAOKAO-Bench; `data/processed/`: GeoMCQ-1200 with options permuted to balance answer positions (and in the
  released order), GaoKao-Geo, and the development/test split.
- `results/twc/<split>_<model>_<variant>.jsonl`: one line per question with the conditions, the judgments, the regions,
  the answer, its explanation and the token counts; `full` is the complete framework, the other variants are the
  ablations. `results/direct/` and `results/dictionary_matching/` hold the baselines.

## Calling the models

Copy `.env.example` to `.env` and enter an API key. The models are served by SiliconFlow under the identifiers
`Qwen/Qwen2.5-7B-Instruct`, `THUDM/GLM-4-9B-0414` and `Qwen/Qwen3.5-122B-A10B` and are decoded greedily, for example

    .venv/bin/python experiments/run_twc.py --split test --model THUDM/GLM-4-9B-0414 --dcr THUDM/GLM-4-9B-0414 --tag full

New responses are added to the cache. The dictionary was built with DeepSeek-V4-Flash; `build_dictionary.py` rebuilds
it with new model calls, and `extract_curriculum.py` needs the PDF of the curriculum standard in `kb/sources/`.

## Licenses

The code, GeoMCQ-1200 and the curriculum dictionary, all compiled by the authors, are released under the MIT License.
The GaoKao-Geo questions come from AGIEval (https://github.com/ruixiangcui/AGIEval, MIT License) and GAOKAO-Bench
(https://github.com/OpenLMLab/GAOKAO-Bench, Apache License 2.0), which collected them from the Chinese national college
entrance examination.
