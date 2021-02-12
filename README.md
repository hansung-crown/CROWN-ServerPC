# CROWN-Server PC

> **개발 환경**

- **Windows 10**
- **NVIDIA Geforce GTX 1060 3GB**
- **CUDA 9.0**
- **cuDNN 7.0.5**
- **Tensorflow 1.7.0**
- **Python 3.6.5**
- **Visual Studio Code**


## How To Use

1. Setup

2. Pre-trained model 다운로드
   - Inception-ResNet-v1 model (<a href="https://drive.google.com/uc?export=view&id=1akOzzDLc221LFBqVe5k9TYiT5sSNkUuo">20180402-114759</a>)
   - 참고 repository: [davidsandberg](https://github.com/davidsandberg/facenet)
   
3. Gesture model 다운로드
   - <a href="https://drive.google.com/uc?export=view&id=1MYu1HBdOFomYHZZIUr_K997RQNfxLE55">gesture_model</a>
   - 참고 repository: [jrobchin](https://github.com/jrobchin/Computer-Vision-Basics-with-Python-Keras-and-OpenCV)
   
4. data, output, logs 디렉토리 만들기
   - data 디렉토리: 학습시킬 얼굴 사진들을 인물별로 저장
   - output 디렉토리: data 디렉토리의 사진들을 align하여 저장
   - logs 디렉토리: 이벤트 발생 시 이미지를 캡쳐하여 저장 (ex. Danger, Unknown그룹 인식)
   
5. 가상환경에서 CROWN.py 실행


## Setup-NVIDIA 그래픽 드라이버

- **그래픽 드라이버 검색 및 설치**

  - https://www.nvidia.co.kr/Download/index.aspx?lang=kr
  - 본인 PC의 GPU 사양에 맞는 드라이버 설치
  - 본 팀은 GeForce GTX 1060 그래픽 카드를 탑재한 Windows 10 운영체제 PC 사용

    <img src="https://user-images.githubusercontent.com/56067179/105938653-0846ff00-609b-11eb-8580-ec62ddc17ab3.PNG" width="500" />

    <img src="https://user-images.githubusercontent.com/56067179/105938846-62e05b00-609b-11eb-8345-1929fabf5266.PNG" width="500" />


- **그래픽 드라이버 설치 확인**

  - 명령 프롬프트

    ```
    $ nvidia-smi
    ```

    <img src="https://user-images.githubusercontent.com/56067179/105939214-16e1e600-609c-11eb-8744-74a3c0ab6ff2.png" width="500" />

  - 제어판

    <img src="https://user-images.githubusercontent.com/56067179/105939286-409b0d00-609c-11eb-97be-ac472bd6c1da.png" width="500" />


> **참고**

- Tensorflow-gpu, Python, cuDNN, CUDA 호환 버전 확인

  - https://www.tensorflow.org/install/source_windows?hl=ko
  - 본 팀은 Tensorflow-gpu : 1.7.0, Python : 3.5-3.6, cuDNN : 7, CUDA : 9 사용

    <img src="https://user-images.githubusercontent.com/56067179/105945073-e607ae00-60a7-11eb-9944-06172c58b30b.png" width="500" />


## Setup-CUDA

- **CUDA 9.0 설치**

  - https://developer.nvidia.com/cuda-toolkit-archive

    <img src="https://user-images.githubusercontent.com/56067179/105944014-cff8ee00-60a5-11eb-92e9-3c8feb2814cb.png" width="800" />

    <img src="https://user-images.githubusercontent.com/56067179/105944151-18181080-60a6-11eb-80ab-9e34c1547fb3.png" width="500" />



## Setup-cuDNN

- **cuDNN 7.0.5 설치**

  - https://developer.nvidia.com/rdp/cudnn-archive 
  - NVIDIA 회원가입 후 다운로드 가능

    <img src="https://user-images.githubusercontent.com/56067179/105944415-9d9bc080-60a6-11eb-8973-a5d58fc04640.png" width="800" />

- **cuDNN 폴더 복사**

  - 다운로드 받은 파일을 아래의 경로에 복사/붙여넣기
  - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0`

    <img src="https://user-images.githubusercontent.com/56067179/105944453-af7d6380-60a6-11eb-9c08-6422fef4468b.png" width="800" />

- **환경 변수 확인**

  - 시스템 환경 변수에 `CUDA_PATH`, `CUDA_PATH_V9_0`이 있는지 확인

    <img src="https://user-images.githubusercontent.com/56067179/105944448-ae4c3680-60a6-11eb-90c2-416cd121d9e8.png" width="400" />

  

## Setup-Python

- **Python 3.6.5 설치**

  - https://www.python.org/downloads/

    <img src="https://user-images.githubusercontent.com/56067179/105945122-0172b900-60a8-11eb-9cea-a331bf5d2ab2.png" width="800" />

    <img src="https://user-images.githubusercontent.com/56067179/105945118-00418c00-60a8-11eb-8d14-9ba3d215b32e.png" width="800" />



## Setup-Visual Studio Code

- **Visual Studio Code 설치**

  - https://code.visualstudio.com/download

    <img src="https://user-images.githubusercontent.com/56067179/105947918-611f9300-60ad-11eb-8d86-f9ed7ad4dcac.png" width="600" />

- **가상환경 생성**

  - Python의 venv 모듈을 사용하여 가상환경 생성 (Python 3.5 버전 이상 지원)

  - VSCode CMD

    ```
    $ python -m venv crownenv // python -m venv {가상환경 이름}
    ```

- **Tensorflow 설치**

  - Tensorflow와 Tenworflow-gpu 1.7.0 설치
  - VSCode CMD

    ```
    $ pip install tensorflow==1.7.0     // tensorflow 
    $ pip install tensorfow-gpu==1.7.0  // tensorflow-gpu
    ```

- **Tensorflow 버전 확인**

  - VSCode CMD

    ```
    $ python 
    $ import tensorflow as tf 
    $ tf.__version__
    ```

- **gsutil 설정**

  - 아래와 같이 입력 후 안내에 따라 필요한 정보 입력
  - VSCode CMD

     ```
     $ gsutil config
     ```

- **FCM용 Firebase Admin SDK 설치**

  - https://firebase.google.com/docs/cloud-messaging/server

  <img src="https://user-images.githubusercontent.com/56067179/106376510-5af52380-63d9-11eb-87dd-c54b9003832f.PNG" width="80%" />

  - https://firebase.google.com/docs/admin/setup#python

  <img src="https://user-images.githubusercontent.com/56067179/106376512-5c265080-63d9-11eb-832f-1eee90bc58b2.PNG" width="80%" />

  - **Firebase Admin SDK JSON 파일 저장**


> **참고**

- CROWN 실행을 위한 라이브러리

  ```
  $ pip install paho-mqtt==1.4.0              // paho 
  $ pip install scipy==1.1.0                  // scipy 
  $ pip install scikit-learn-0.21.3           // sklearn 
  $ pip install opencv-python==4.1.0.25       // cv2 
  $ pip install matplotlib==3.0.3             // matplotlib 
  $ pip install pyrebase==3.0.27              // pyrebase 
  $ pip install pyfcm==1.4.7                  // pyfcm 
  $ pip install keras==2.2.4                  // keras 
  $ pip install numpy==1.16.1                 // numpy 
  $ pip install pillow==6.1.0                 // pillow 
  $ pip install requests==2.11.1              // requests 
  $ pip install google-cloud-storage==1.19.0  // google-cloud-storage 
  $ pip install gsutil==4.42                  // gsutil
  ```

