# KMU Trade Comovement – 팀 Git 사용 가이드

제3회 국민대학교 AI빅데이터 분석 경진대회  
**“Trade Data-based Comovement Analysis and Forecasting Approach”**

이 문서는 **우리 팀 전용 Git 사용 설명서**입니다.  
추후 스크립트가 완료 된 후에, readme.md 수정 예정

---

- `main`  
  - 최종 제출용 브랜치 (깨끗한 상태 유지, 여기서 직접 작업 ❌)
- `dev`  
  - 팀 작업 결과를 모으는 브랜치
- `exp/이름`  
  - **개인 실험용 브랜치**
  - 각자 EDA + 공행성 + 모델링을 마음껏 하는 공간

> ✔ **원칙 요약**  
> - 작업은 항상 `exp/본인이름`에서 한다.  
> - 공유하고 싶은 코드만 나중에 `dev`에 합친다.  
> - `main`은 최종 제출 시점에만 건드린다.

---
# 📦 1️⃣ Repository clone (최초 1회만)

```bash
git clone https://github.com/hazelJung/trade-comovement.git
cd trade-comovement
```

---

# ⚙️ 2️⃣ Git 최초 설정 (최초 1회만)

```bash
git config --global user.name "본인 이름"
git config --global user.email "깃허브 이메일"
```

---

# 🧭 3️⃣ 본인 브랜치로 이동

```bash
git branch -a
git checkout exp/본인이름
```

---

# ✍️ 4️⃣ 기본 Git 사용법

```bash
git status # 상태확인
git add . #파일 업로드
git commit -m "메시지" #하나의 버전으로 저장
git push #업로드
git pull origin dev #최신 용 가져오기
git merge dev #최신내용 합치기
git branch -a #모든 브랜치 보기 (* 표시가 현재)
git checkout 브랜치명 #브랜치 이동
git checkout -b 새브랜치 #브랜치 생성 및 이동
```

---

# 🧠 5️⃣ 개인 작업 흐름

```bash
git checkout exp/본인이름
# 코드 수정
git add .
git commit -m "feat: 작업 내용"
git push
```

---

# 🔄 6️⃣ dev 최신 코드 반영

```bash
git checkout dev
git pull origin dev

git checkout exp/본인이름
git merge dev
```

---

# 🔀 7️⃣ exp → dev 반영(PR)

1) GitHub → Pull Request  
2) base: develop / compare: exp/본인이름  
3) 설명 작성 → Create PR  
4) 팀장 승인 후 merge

---

# 🚀 8️⃣ dev → main 반영 (팀장 전용)

```bash
git checkout develop
git pull origin develop

git checkout main
git pull origin main
git merge develop
git push origin main
```

---

# 🗑 9️⃣ 브랜치 삭제

로컬 삭제:
```bash
git branch -d 브랜치명
git branch -D 브랜치명
```

원격 삭제:
```bash
git push origin --delete 브랜치명
```

---
