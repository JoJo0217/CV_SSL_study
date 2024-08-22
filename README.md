# AI 학습 repository입니다.

대부분의 모델, 학습 논문은 [블로그](https://velog.io/@jojo0217/posts)를 참조하시면 볼 수 있습니다.

## 실행
실행은 pwd를 각 폴더로 이동한 다음 run.sh 파일을 가지고 실행하면 됩니다.   
파일의 옵션은 main.py에서 읽어보실 수 있습니다.
```shell
./run.sh
```

## 모델들 
models 폴더에 각종 모델들이 정리되어 있습니다.   
model.py 파일에 모델을 불러오기 위한 이름이 정의되어 있습니다.
- lenet5
- resnet기반 모델들 
  - resnet18
  - resnet18preact
  - resnet18bottleneck
  - resnet50bottleneck
  - resnext
  - convnext
- transformer기반 모델들
  - vit
- mixer 기반 모델들
  - mlpmixer
  - convmixer

## 학습
fine-tune은 finetune 폴더에 구현이 되어있고   
pretrain은 pretrain 폴더에 구현이 되어있습니다.

실행방법은 둘다 동일하게 ./run.sh입니다.

pretrain 구현 목록입니다.
- moco
- rotnet
- simclr
- byol
- simsiam

---
학습한 모델을 테스트하기 위해서는 evaluate.py 코드를 활용할 수 있습니다.
```shell
python3 evaluate.py --model [경로] --option
```
만약 pretrain으로 학습하였을 경우 --pretrain을 추가하시면 됩니다.