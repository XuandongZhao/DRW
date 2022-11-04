# DRW
[EMNLP 2022 Findings] Distillation-Resistant Watermarking (DRW) for Model Protection in NLP

Example:

`
python student.py --task ner --wm 1 --wmidx 2 --hard 1 --tseed 22 --nseed 11 --sub 0.5 --starti 0 --endi 0.5 --epochs 20 --eps 0.2 --k 16 --device cuda:0 --batch-size 32
`