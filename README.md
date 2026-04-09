# JH201421228

---
Git을 처음 설치했다면 아래 명령어를 한 번만 입력하세요.
``` bash
git config --global user.name "본인이름"  
git config --global user.email "본인이깃허브이메일"  
```


설정 확인:  
``` bash
git config --global user.name  
git config --global user.email
```

---  
``` bash
git clone [깃허브주소]
cd [레포이름]
git pull origin [기본브랜치명]
git checkout -b [새브랜치명]
```

# 작업 후
``` bash
git status
git add .
git commit -m "feat: 작업 내용"
git push -u origin [새브랜치명]

```
