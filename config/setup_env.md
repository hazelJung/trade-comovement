# 가상환경 설정 가이드

## 1. 가상환경 생성
```bash
python -m venv venv
```

## 2. 가상환경 활성화

### Windows (PowerShell)
```powershell
.\venv\Scripts\Activate.ps1
```

### Windows (CMD)
```cmd
venv\Scripts\activate.bat
```

### Linux/Mac
```bash
source venv/bin/activate
```

## 3. 패키지 설치
```bash
pip install -r requirements.txt
```

## 4. Jupyter Notebook 실행
```bash
jupyter notebook
```

또는

```bash
jupyter lab
```

## 5. 노트북 열기
브라우저에서 `train_model.ipynb` 파일을 열어 실행하세요.

## 6. 가상환경 비활성화
작업이 끝나면 다음 명령어로 가상환경을 비활성화할 수 있습니다:
```bash
deactivate
```

