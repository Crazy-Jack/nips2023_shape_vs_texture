# Texture Synthesis Program for Top-K neuron visualization

## Prepare model

Download and place the pretrained VGG model to `models/VGG19_normalized_avg_pool_pytorch` from the github.

## Texture synthesis with Top-K + non Top-K neurons

```
python3 synthesize.py -i ../few-shot-img-syn/data/jeep -o vis_jeep_all --topk 1. --lr 10 -n 1
```

## Texture synthesis with Top-K neurons only

```
python3 synthesize.py -i ../few-shot-img-syn/data/jeep -o vis_jeep_topk --topk 0.05 --lr 10 -n 1
```
## Texture synthesis with non Top-K neurons only

```
python3 synthesize.py -i ../few-shot-img-syn/data/jeep -o vis_jeep_non_topk --topk 0.05 --reverse_topk --lr 10 -n 1
```