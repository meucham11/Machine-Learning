텐서플로우 공부하기!

기존 python 3.7을 쓰다보니 tensorflow 설치에 충돌이 많았다.

가상환경을 만들어 python 3.65로 다운그레이드 환경을 준비한 후 텐서를 사용하여

주피터에서 커널을 연결하여 사용하기로 하였다.

```
anaconda prompt에

conda create -n 원하는이름 python=3.7     --> 원래 python 3.7.4 사용중이었으므로.
conda activate 위에입력한이름        --> 가상환경 활성화
conda install python=3.6.5          --> 원하는 파이썬 버전 설치
pip install tensorflow     --> 원하는 버전을 깔아주자  3.6이라 2가 깔려도 잘될것이다.
conda deactivate           --> 현재 가상환경 종료. 원래 환경으로 돌아가 파이썬 3.7.4가 실행될 것이다.

```
