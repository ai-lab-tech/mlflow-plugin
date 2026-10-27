# Third-party notices

The root Apache-2.0 license applies to original project code. It does not replace
the licenses of included upstream sources, pretrained model assets or dependencies.
Original copyright and source notices remain in their files.

## Included example source code

The source revisions below come from the dataset's `invocation.json` and
`preparation.json` provenance. License texts were retrieved from those exact
revisions, not inferred from the framework name.

| Included material | Upstream license evidence | Preserved license text |
| --- | --- | --- |
| All four `sklearn_*` cases | [BSD-3-Clause](https://raw.githubusercontent.com/scikit-learn/scikit-learn/25dee604bae18205b01548348388baf7a1cdfe0e/COPYING) | [scikit-learn](licenses/third-party/scikit-learn.txt) |
| `pytorch_mnist_lstm`, `pytorch_dcgan` | [BSD-3-Clause](https://raw.githubusercontent.com/pytorch/examples/acc295dc7b90714f1bf47f06004fc19a7fe235c4/LICENSE) | [pytorch-examples](licenses/third-party/pytorch-examples.txt) |
| `pytorch_polynomial` | [BSD-3-Clause](https://raw.githubusercontent.com/pytorch/tutorials/c8ccbbf7fcb8cb71adda1b957bd6bab50e29c2e6/LICENSE) | [pytorch-tutorials](licenses/third-party/pytorch-tutorials.txt) |
| `pytorch_custom_network` | [MIT](https://raw.githubusercontent.com/jcjohnson/pytorch-examples/0003fcdc606a79951e53bf8f46f8574fa4739f23/LICENSE) | [jcjohnson-pytorch-examples](licenses/third-party/jcjohnson-pytorch-examples.txt) |
| HF classification, NER and question-answering training sources | [Apache-2.0](https://raw.githubusercontent.com/huggingface/transformers/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/LICENSE) | [transformers](licenses/third-party/transformers.txt) |
| HF generation training sources | [Apache-2.0](https://raw.githubusercontent.com/huggingface/trl/04fd1203af4fc1e629f58e0ac0d0c5bb95f82a45/LICENSE) | [trl](licenses/third-party/trl.txt) |

The sklearn/HF training source files and PyTorch original examples are retained as
provenance. Serving does not execute those training entry points. Locally created
custom-HF integration code is original project code; its model weights and tokenizer
come from the classification fixture and retain their upstream provenance.

## Pretrained models and derived fixtures

| Assets | Source revision / license declaration | Local changes |
| --- | --- | --- |
| BERT weights/tokenizer in HF classification, NER, question answering and custom cases | [Google BERT miniature model card](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/blob/30b0a37ccaaa32f332884b96992754e246e48c5f/README.md), Apache-2.0 | Classification/NER/QA fixtures were fine-tuned by the recorded original scripts. Custom HF reuses classification weights with local custom inference code. |
| SmolLM2 weights/tokenizer in `huggingface_generation` | [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md), Apache-2.0 | Two recorded SFT updates; see the case's invocation and base-model metadata. |

The [Apache-2.0 text](LICENSE) is included in the repository and both packages.
The BERT upstream card requests citation of Iulia Turc, Ming-Wei Chang, Kenton Lee
and Kristina Toutanova, *Well-Read Students Learn Better: On the Importance of
Pre-training Compact Models* (2019), [arXiv:1908.08962](https://arxiv.org/abs/1908.08962).
The SmolLM2 card links its [model paper](https://arxiv.org/abs/2502.02737).
These are integration fixtures, not claims of model quality or original pretrained weights.

## Remaining artifact redistribution review

The model cards document the base models' declared licenses. They do not by
themselves settle redistribution of all derived training artifacts. Before making
the full Git/LFS history public, record a maintainer review of the training-data
terms and derived-weight redistribution for the retained fixtures, especially:

- HF classification/custom: SST-2, as recorded in the classification invocation.
- HF NER: `tomaarsen/conll2003`.
- HF question answering: `rajpurkar/squad`.
- HF generation: `trl-lib/Capybara`.
- sklearn and PyTorch: the source examples' datasets, including downloaded data
  and synthetic inputs, as identified by the retained scripts and run evidence.

This review is **not complete**. No blanket Apache-2.0 claim is made for the test
dataset, and model files have not been removed or relicensed. If an artifact cannot
be redistributed, resolve both the current tree and previously committed Git LFS
history before publication. See [dataset provenance](examples/data/README.md).

## Installed dependencies and runtime images

MLflow, framework libraries and Triton container images retain their own licenses.
They are not bundled into the plugin wheels. Review the actual versions in package
metadata, each fixture's requirements and the selected runtime image when preparing
a distribution. This notice is an inventory of included materials, not a complete
license audit of every transitive dependency.
