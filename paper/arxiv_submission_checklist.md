# arXiv Submission Checklist

## Recommended Category

Recommended primary category: `cs.LG`.

Reason: the paper is about empirical behavior of an LLM inference optimization method, with a
machine-learning methodology and evaluation framing.

Recommended cross-list: `cs.PF`.

Reason: the paper includes performance measurement and evaluation. `cs.DC` is a possible cross-list
only if the final abstract emphasizes serving-systems or distributed/parallel systems concerns more
strongly; the current draft is more directly a machine-learning/performance evaluation paper than a
distributed systems paper.

## Endorsement

For a first-time arXiv submitter, arXiv may require endorsement for `cs.LG`, `cs.PF`, or related
computer-science categories. Endorsement requirements are account- and category-dependent. Before
uploading, create or log into the arXiv account, associate any institutional email address if
available, and check whether arXiv asks for an endorsement code for the chosen category.

## License

Recommended license if broad reuse is acceptable: Creative Commons Attribution 4.0 (`CC BY 4.0`).
It allows redistribution and adaptation, including commercial use, with attribution.

More restrictive fallback: arXiv's non-exclusive license to distribute. Use this if you do not want
to grant broad reuse rights. Before selecting a license, confirm that the target workshop or future
publication venue does not impose conflicting requirements.

## Before Upload

- Replace `paper/neurips_2023.sty` with the official NeurIPS 2026 or target-workshop style file if
  one is available.
- Regenerate tables and figures with:

```bash
python paper/generate_assets.py
```

- Compile locally or in Overleaf/arXiv. This machine currently lacks a local LaTeX toolchain
  (`pdflatex`, `latexmk`, and `bibtex` were not found).
- Confirm that every number in `paper/main.tex` still matches generated files in `paper/tables/`.
- Confirm that `results/*.csv` files needed by `paper/generate_assets.py` are committed.
- Review the limitations section carefully before public submission.
