# DRW
[EMNLP 2022] [Distillation-Resistant Watermarking (DRW) for Model Protection in NLP](https://arxiv.org/abs/2210.03312)

Example:

```bash
python student.py --task ner --wm 1 --wmidx 2 --hard 1 --tseed 22 --nseed 11 --sub 0.5 --starti 0 --endi 0.5 --epochs 20 --eps 0.2 --k 16 --device cuda:0 --batch-size 32
```

## Citation

Please cite our paper if you find DRW useful for your research:

```bibtex
@inproceedings{zhao-etal-2022-distillation,
    title = "Distillation-Resistant Watermarking for Model Protection in NLP",
    author = "Zhao, Xuandong  and Li, Lei  and Wang, Yu-Xiang",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    year = "2022"
}
```
