## Git 최초 설정

Git을 처음 설치했다면 아래 명령어를 한 번만 입력하세요.

```bash
git config --global user.name "본인이름"
git config --global user.email "본인의깃허브이메일"
```

```
git config --global user.name
git config --global user.email
```
---
깃 저장소 로컬에 가져오기
```
git clone [깃허브주소]
cd [레포이름]
```
예시:
```
git clone https://github.com/organization/project.git
cd project
```
작업 후 업로드하기
```
git status
git add .
git commit -m "feat: 작업 내용"
git push
```
---
처음 push가 안 될 때

처음 업로드하는 경우 아래 명령어를 한 번 입력하세요.
```
git push -u origin main
```
이후부터는 아래처럼 git push만 입력하면 됩니다.
```
git push
```
기본 브랜치가 main 혹은 master일 가능성이 높습니다.
