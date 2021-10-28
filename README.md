# Pytorch Classification Basic

- Model : timm 패키지
- Cross validation 기능
- 복수의 train input sizes 가능 (예시 : --train_img_sizes 192 256 320 380)
- multi processing 가능 (multi-gpu 불가능)
- scheduler : CosineAnnealingWarmUpRestarts (torch.optim.COSINEANNEALINGWARMRESTARTS 함수는 WarmUp 기능이 제대로 작동하지 않음.)
- tensorboard 및 txt 로그 기록
- 윈도우 기준 경로이므로 리눅스에서 사용시 수정 필요(dataset에서 __getitem__)
- 함수, 클래스 이름 변경 필요
