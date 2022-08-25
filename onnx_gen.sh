#!/usr/bin/env bash

### MobileNetV1 ##
## Seed
#python3 export2onnx.py \
#    plain_mobilenetv1 \
#    --output-file onnx/mobilenetv1_seed.onnx
## 25%-H
python3 export2onnx.py \
    learned_mobilenetv1 \
    --output-file onnx/mobilenetv1_25_h.onnx \
    --learned-ch 8 16 32 32 64 64 128 115 112 2 2 128 5 30
# 25%-L
python3 export2onnx.py \
    learned_mobilenetv1 \
    --output-file onnx/mobilenetv1_25_l.onnx \
    --learned-ch 5 12 32 32 64 64 128 118 116 3 2 128 2 25
# 12.5%-H
python3 export2onnx.py \
    learned_mobilenetv1 \
    --output-file onnx/mobilenetv1_12-5_h.onnx \
    --learned-ch 8 16 32 32 64 63 11 105 3 128 4 3 255 30
# 12.5%-L
python3 export2onnx.py \
    learned_mobilenetv1 \
    --output-file onnx/mobilenetv1_12-5_l.onnx \
    --learned-ch 6 5 6 10 12 33 42 127 11 6 128 5 255 33
# 6.25%-H
python3 export2onnx.py \
    learned_mobilenetv1 \
    --output-file onnx/mobilenetv1_6-25_h.onnx \
    --learned-ch 8 15 28 25 14 26 14 103 3 2 9 1 250 14
# 6.25%-L
python3 export2onnx.py \
    learned_mobilenetv1 \
    --output-file onnx/mobilenetv1_6-25_l.onnx \
    --learned-ch 5 4 5 8 14 12 10 123 3 8 5 3 254 23 

## ResNet8 ##
# Seed
#python3 export2onnx.py \
#    plain_resnet8 \
#    --output-file onnx/resnet8_seed.onnx
## 75%-H
#python3 export2onnx.py \
#    learned_resnet8 \
#    --output-file onnx/resnet8_75_h.onnx \
#    --learned-ch 11 15 15 32 26 64 44
## 75%-L
#python3 export2onnx.py \
#    learned_resnet8 \
#    --output-file onnx/resnet8_75_l.onnx \
#    --learned-ch 7 4 9 8 27 64 64
## 50%-H
#python3 export2onnx.py \
#    learned_resnet8 \
#    --output-file onnx/resnet8_50_h.onnx \
#    --learned-ch 10 16 14 32 12 62 36
## 50%-L
#python3 export2onnx.py \
#    learned_resnet8 \
#    --output-file onnx/resnet8_50_l.onnx \
#    --learned-ch 7 4 8 7 13 64 49
## 25%-H
#python3 export2onnx.py \
#    learned_resnet8 \
#    --output-file onnx/resnet8_25_h.onnx \
#    --learned-ch 12	6 14 32	12 23 30
## 25%-L
#python3 export2onnx.py \
#    learned_resnet8 \
#    --output-file onnx/resnet8_25_l.onnx \
#    --learned-ch 7 3 7 7 10 50 27