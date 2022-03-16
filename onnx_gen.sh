#!/usr/bin/env bash

### MobileNetV1 ##
## Seed
#python3 export2onnx.py \
#    plain_mobilenetv1 \
#    --output-file onnx/mobilenetv1_seed.onnx
## 50%-H
#python3 export2onnx.py \
#    learned_mobilenetv1 \
#    --output-file onnx/mobilenetv1_50_h.onnx \
#    --learned-ch 8 8 16 16 32 32 32	32 64 64 64	64 128 128 128 128 128 128 128 128 128 128 4 4 254 254 26
## 50%-L
#python3 export2onnx.py \
#    learned_mobilenetv1 \
#    --output-file onnx/mobilenetv1_50_l.onnx \
#    --learned-ch 8 8 16 16 32 32 32	32 64 64 64	64 128 128 128 128 128 128 128 128 125 125 4 4 3 3 245
## 25%-H
#python3 export2onnx.py \
#    learned_mobilenetv1 \
#    --output-file onnx/mobilenetv1_25_h.onnx \
#    --learned-ch 8 8 16	16 32 32 32	32 64 64 64	64 128 128 115 115 113 113 4 4 3 3 6 6 6 6 29
## 25%-L
#python3 export2onnx.py \
#    learned_mobilenetv1 \
#    --output-file onnx/mobilenetv1_25_l.onnx \
#    --learned-ch 8 8 16	16 32 32 32	32 64 64 64	64 128 128 121 121 13 13 2 2 2 2 6 6 6 6 17
## 12.5%-H
#python3 export2onnx.py \
#    learned_mobilenetv1 \
#    --output-file onnx/mobilenetv1_12-5_h.onnx \
#    --learned-ch 8 8 16	16 32 32 32	32 64 64 64	64 123 123 2 2 2 2 126 126 2 2 1 1 256 256 6
## 12.5%-L
#python3 export2onnx.py \
#    learned_mobilenetv1 \
#    --output-file onnx/mobilenetv1_12-5_l.onnx \
#    --learned-ch 2 2 3 3 3 3 28	28 64 64 63	63 29 29 123 123 2 2 7 7 128 128 3 3 256 256 18

## ResNet8 ##
# Seed
python3 export2onnx.py \
    plain_resnet8 \
    --output-file onnx/resnet8_seed.onnx
# 75%-H
python3 export2onnx.py \
    learned_resnet8 \
    --output-file onnx/resnet8_75_h.onnx \
    --learned-ch 16 16 16 32 32 32 64 36 36
# 75%-L
python3 export2onnx.py \
    learned_resnet8 \
    --output-file onnx/resnet8_75_l.onnx \
    --learned-ch 7 2 5 6 32	32 64 64 64
# 50%-H
python3 export2onnx.py \
    learned_resnet8 \
    --output-file onnx/resnet8_50_h.onnx \
    --learned-ch 16 16 16 32 13	13 61 33 33
# 50%-L
python3 export2onnx.py \
    learned_resnet8 \
    --output-file onnx/resnet8_50_l.onnx \
    --learned-ch 6 3 5 6 9 9 62	63 63
# 25%-H
python3 export2onnx.py \
    learned_resnet8 \
    --output-file onnx/resnet8_25_h.onnx \
    --learned-ch 16 16 16 32 8 8 48 10 10
# 25%-L
python3 export2onnx.py \
    learned_resnet8 \
    --output-file onnx/resnet8_25_l.onnx \
    --learned-ch 4 1 3 4 6 6 63 28 28