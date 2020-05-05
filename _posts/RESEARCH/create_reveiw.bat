@echo off

for /l %%i in (1,1,9) do (
   IF not exist 2019-08-13-PAPER-REVIEW-000%%i.md (
         copy 2019-08-13-PAPER-REVIEW-0000.md 2019-08-13-PAPER-REVIEW-000%%i.md
         echo File 2019-08-13-PAPER-REVIEW-000%%i.md is created!
	git pull
	git add .
	git commit -m "create file"
	git push
	start firefox https://github.com/userdyk-github/userdyk-github.github.io/edit/master/_posts/RESEARCH/2019-08-13-PAPER-REVIEW-000%%i.md
      	start firefox https://github.com/userdyk-github/userdyk-github.github.io/edit/master/_posts/RESEARCH/2019-08-13-PAPER-REVIEW.md 
	goto :eof
   )
)

for /l %%j in (10,1,99) do (
   IF not exist 2019-08-13-PAPER-REVIEW-00%%j.md (
         copy 2019-08-13-PAPER-REVIEW-0000.md 2019-08-13-PAPER-REVIEW-00%%j.md
         echo File 2019-08-13-PAPER-REVIEW-00%%j.md is created!
	git pull
	git add .
	git commit -m "create file"
	git push
	start firefox https://github.com/userdyk-github/userdyk-github.github.io/edit/master/_posts/RESEARCH/2019-08-13-PAPER-REVIEW-000%%i.md
      	start firefox https://github.com/userdyk-github/userdyk-github.github.io/edit/master/_posts/RESEARCH/2019-08-13-PAPER-REVIEW.md 
         goto :eof
   )
)


for /l %%k in (100,1,999) do (
   IF not exist 2019-08-13-PAPER-REVIEW-0%%k.md (
         copy 2019-08-13-PAPER-REVIEW-0000.md 2019-08-13-PAPER-REVIEW-0%%k.md
         echo File 2019-08-13-PAPER-REVIEW-0%%k.md is created!
	git pull
	git add .
	git commit -m "create file"
	git push
	start firefox https://github.com/userdyk-github/userdyk-github.github.io/edit/master/_posts/RESEARCH/2019-08-13-PAPER-REVIEW-000%%i.md
      	start firefox https://github.com/userdyk-github/userdyk-github.github.io/edit/master/_posts/RESEARCH/2019-08-13-PAPER-REVIEW.md 
         goto :eof
   )
)

for /l %%l in (1000,1,9999) do (
   IF not exist 2019-08-13-PAPER-REVIEW-%%l.md (
         copy 2019-08-13-PAPER-REVIEW-0000.md 2019-08-13-PAPER-REVIEW-%%l.md
         echo File 2019-08-13-PAPER-REVIEW-%%l.md is created!
	git pull
	git add .
	git commit -m "create file"
	git push
	start firefox https://github.com/userdyk-github/userdyk-github.github.io/edit/master/_posts/RESEARCH/2019-08-13-PAPER-REVIEW-000%%i.md
      	start firefox https://github.com/userdyk-github/userdyk-github.github.io/edit/master/_posts/RESEARCH/2019-08-13-PAPER-REVIEW.md 
         goto :eof
   )
)

@echo success
  
