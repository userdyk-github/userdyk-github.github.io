---
layout : post
title : PL00, System programming
categories: [PL00]
comments : true
tags : [PL00]
---
[Back to the previous page](https://userdyk-github.github.io/Study.html) <br>
List of posts to read before reading this article
- <a href='https://userdyk-github.github.io/'>post1</a>
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

## Contents
{:.no_toc}

* ToC
{:toc}

<hr class="division1">

## **Introduction**

### ***Environmental variables***
/home/user/.bashrc
```bash
$ echo $PATH
```
```bash
$ sudo echo $PATH
```
```bash
$ whereis [command]
```
```bash
$ which [command]
```
```bash
$ sudo which [command]
```
```vim
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
```
<br><br><br>
<span class="frame3">Execute</span><br>
```bash
$ [command]           # $echo $PATH;whereis [command];which [command]  : through environmental variables
$ ./[command]         # pwd : on current directory 
```
<br><br><br>

---


### ***File on Linux***
<a href="https://linuxhandbook.com/linux-directory-structure/" target="_blank">Linux directory structure</a>

<span class="frame3">working directory</span><br>
```bash
$ pwd
```
```bash
$ cd ~     # move user directory
$ cd /     # move root directory
$ cd -     # move previous working directory
```
```bash
$ ls
$ ls -l
$ ls -a
$ ls -al
$ ls | grep [pattern]
$ ls -i
```
<details markdown="1">
<summary class='jb-small' style="color:blue">ls wildcard example</summary>
<hr class='division3'>
![image](https://user-images.githubusercontent.com/52376448/68069411-2bed9d80-fda3-11e9-873a-478c2625db18.png)

```bash
$ ls host*
```
```
host.conf  hostname  hosts  hosts.allow  hosts.deny
```
<br>
```bash
$ ls host?
```
```
hosts
```
<br>
```bash
$ ls host[stu]
```
```
hosts
```

<hr class='division3'>
</details>
```bash
$ rm *                    # remove all files
$ rm [file_name]
$ rm -rf [folder_name]
```

<br><br><br>

<span class="frame3">Search contents</span><br>
```bash
$ cat [file_name]
$ cat [file_name] | sort {-r}
$ cat [file_name] | sort {-r} | grep [pattern]
$ head [file_name]
$ head -n[number_of_lines] [file_name]
$ tail [file_name]
$ tail -n[number_of_lines] [file_name]
$ more [file_name]                            # spacebar : next page, enter : next line
$ less [file_name]                        
```

```bash
$ grep [pattern] [file_name]         # search conformable pattern(string) in file
$ grep [pattern] *                   # search conformable pattern(string) only in present directory

$ grep -i [pattern] [file_name]      # search conformable pattern(string) in file, regardless of capital and small letter
$ grep -v [pattern] [file_name]      # search unconformable pattern(string) in file
$ grep -n [pattern] [file_name]      # search conformable pattern(string) in file, numbering line on results for searching
$ grep -l [pattern] [file_name]      # search conformable pattern(string) in file, displaying only file name including pattern

$ grep -c [pattern] [file_name]      # search conformable pattern(string) in file, displaying the number of consistant pattern line on a file 
$ grep -c [pattern] *                # search conformable pattern(string) in file, displaying the number of consistant pattern line on present directory 
$ grep -c [pattern] [folder_name]    # search conformable pattern(string) in file, displaying the number of consistant pattern line per file on a folder

$ grep -r [pattern] *                # search conformable pattern(string) in present directory including sub-directory
$ grep -E "[pattern_1]|[pattern_2]|[pattern_3]" [file_name]        # search pattern 1 or pattern 2 or pattern 3 in file
```
<details markdown="1">
<summary class='jb-small' style="color:blue">grep example</summary>
<hr class='division3'>
<a href="https://recipes4dev.tistory.com/157" target="_blank">URL</a>
<hr class='division3'>
</details>
<br><br><br>

<span class="frame3">Search file</span><br>
```bash
$ find / -name [file_name]                        # from root directory
$ find / -size [file_size]                        # from root directory
$ find / -name [file_name] -size [file_size]      # from root directory
$ find . -name [file_name]                        # from current directory
$ find . -size [file_size]                        # from current directory
$ find . -name [file_name] -size [file_size]      # from current directory
```
```bash
$ cmp [file_name] [file_name]
$ diff [file_name] [file_name]
$ file [file_name]
```
<br><br><br>

<span class="frame3">Zip, Unzip</span><br>
```bash
$ tar -cf [name.tar] a b c        # tie and zip a b c files 
$ tar -zcf [name.tar.gz] a b c    # zip a b c files 
$ tar -xvf [name.tar]             # unzip .tar
$ tar -zxvf [name.tar.gz]         # unzip .tar.gz
```
<details markdown="1">
<summary class='jb-small' style="color:blue">SUPPLEMENT</summary>
<hr class='division3'>

- f : set name of file that will be tied 
- c : tie files with .tar
- x : unzip .tar(.gz) file
- v : print list detaily
- z : zip to gzip
- t : print list
- p : save authroity of files
- C : set path

<hr class='division3'>
</details>



<br><br><br>


### ***Process on Linux***

<br><br><br>

---

### ***Authority on Linux***

- OS manages permissions of users and resources
- Linux manages permissions with users and groups
- root is super user(manager)
- On each files, for owner, owner group, and others(all users)
    - management of 'read', 'write', 'execute'
    - save access authority information in data structure of inode

<br><br><br>

<hr class="division2">

## **Shell**

### ***Multiuser system***
#### User
<span class="frame3">all user name currently accessed on linux</span>
```bash
$ who
```
<br><br><br>


<span class="frame3">my user name currently accessed on linux</span>
```bash
$ whoami
```
<details markdown="1">
<summary class='jb-small' style="color:blue">detail</summary>
<hr class='division3'>
```bash
$ id
```
<hr class='division3'>
</details>

<br><br><br>

<span class="frame3">change password</span>
```bash
$ passwd [user_name]         # change or create
$ passwd -d [user_name]      # delete
$ passwd -u [user_name]      # unlock
$ passwd -l [user_name]      # lock
```
<details markdown="1">
<summary class='jb-small' style="color:blue">when error</summary>
<hr class='division3'>
<a href="https://www.codevoila.com/post/26/fix-authentication-token-manipulation-error-when-changing-user-password-in-ubuntu" target="_blank">https://www.codevoila.com/post/26/fix-authentication-token-manipulation-error-when-changing-user-password-in-ubuntu</a>
```bash
$ mount -rw -o remount /
# or
$ mount -o remount,rw /
```
```bash
$ ls -l /etc/shadow
```
<details markdown="1">
<summary class='jb-small' style="color:red">OUTPUT</summary>
<hr class='division3_1'>
```
-rw-r----- 1 root shadow 1025 Feb  11 22:11 /etc/shadow
```
<hr class='division3_1'>
</details>
<br>
```bash
$ passwd
```
<hr class='division3'>
</details>
<br><br><br>

<span class="frame3">create another user id on linux</span><br>
/etc/passwd : ID <br>
/etc/shadow : Password <br>
```bash
$ adduser [user_name]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">If you concretely set user id</summary>
<hr class='division3'>
![image](https://user-images.githubusercontent.com/52376448/68995715-1e243600-08d4-11ea-837a-252d4cd68830.png)
```bash
$ useradd [user_name]
```
```bash
$ passwd [user_name]
```
```bash
$ mkdir /home/[user_name]
$ chown [user_name]:[user_name] /home/user_name
```
```bash
$ usermod -a -G [group_name] [user_name]
```
```bash
$ userdel -r [user_name]
```
<details markdown="1">
<summary class='jb-small' style="color:red">SUPPLEMENT : delete user</summary>
<hr class='division3_1'>
<span>userdel</span><br>
```bash
$ userdel [user_name]         # only account
$ userdel -r [user_name]      # account, home directory
```
<br>
<span>deluser</span><br>
```bash
$ deluser [user_name]                    # only account
$ deluser --remove [user_name]           # account, home directory
$ deluser --remove-all-files [user_name] # account, home directory, all files
```

<hr class='division3_1'>
</details>

<br><br><br>


<span class="frame3">script file</span><br>
```bash
$ touch adduser
$ vim adduser
```
```vim
useraddd $1
tail -n2 /etc/passwd
mkdir /home/$1
chown $1:$1 /home/$1
echo "$1 user added"
```
```bash
$ sudo ./adduser [user_name]
```
<hr class='division3'>
</details>
<br><br><br>

<span class="frame3">change user</span>
```bash
$ su [user_name]        # change id including my present .bashrc, .profile files.
$ su - [user_name]       # change id excluding my present .bashrc, .profile files.
```
```bash
$ exit
```
<br><br><br>

<span class="frame3">authorize superuser to user </span>
```bash
$ usermod -a -G sudo [user_name]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Another way: Edit</summary>
<hr class='division3'>
```bash
:~$ cd /etc
:/etc$ vim sudoers       # change id excluding my present .bashrc, .profile files.
```
```vim
root    ALL=(ALL:ALL) ALL
```
![image](https://user-images.githubusercontent.com/52376448/68069209-833e3e80-fda0-11e9-8279-47d6d5ee45a4.png)

```vim
root    ALL=(ALL:ALL) ALL
[username]    ALL=(ALL:ALL) ALL
```
![image](https://user-images.githubusercontent.com/52376448/68069219-a9fc7500-fda0-11e9-98c7-8c3c07f60cc2.png)
<hr class='division3'>
</details>
<br><br><br>

#### Group
<span class="frame3">Group management</span><br>
/etc/group
```bash
$ groupadd [group_name]                     # create group
$ usermod -a -G [group_name] [user_name]    # add user to group
$ groups [user_name]                        # list of groups including user
```
<br><br><br>

<span class="frame3">file permission</span> ｜ <a href="https://en.wikipedia.org/wiki/Chmod" target="_blank" class="jb-medium">URL</a><br>
<span class="frame3_1">for file</span>
```bash
$ chmod u=rwx, g=rw, o=rx [file_name]
$ chmod 765 [file_name]
```
<span class="frame3_1">for folder</span>
```bash
$ chmod u=rwx, g=rw, o=rx [folder_name]
$ chmod -R u=rwx, g=rw, o=rx [folder_name]

$ chmod 765 [folder_name]
$ chmod -R 765 [folder_name]
```
<br><br><br>

<span class="frame3">change owner</span><br>
<span class="frame3_1">for file</span><br>
```bash
$ chown [owner:owner_group] [file_name]

$ chown [owner:] [file_name]

$ chown [:owner_group] [file_name]
$ chgrp [owner_group] [file_name]
```
<br>
<span class="frame3_1">for folder</span><br>
```bash
$ chown [owner:owner_group] [folder_name]
$ chown -R [owner:owner_group] [folder_name]

$ chown [owner:] [folder_name]
$ chown -R [owner:] [folder_name]

$ chown [:owner_group] [folder_name]
$ chown -R [:owner_group] [folder_name]
$ chgrp [owner_group] [folder_name]
$ chgrp -R [owner_group] [folder_name]
```


<br><br><br>

#### Super user

<span class="frame3">Password</span>
```bash
$ passwd root       : create password
$ passwd -d root    : delete password
```

<br><br><br>


<span class="frame3">Super user</span>
```bash
$ su
$ su -l
$ su --login
$ sudo
```

<br><br><br>

<br><br><br>

---

### ***Standard Stream***

<a href="https://en.wikipedia.org/wiki/Redirection_(computing)" target="_blank" style="font-size: 70%;">URL</a>

- <b>0 : stdin</b> : standard input stream from keyboard
- <b>1 : stdout</b> : standard output stream to screen
- <b>2 : stderr</b> : standard error stream to screen

```bash
$ echo [arbitary_stdout]
```
```
arbitary_stdout
```
<br><br><br>


#### Redirection

<div style="font-size: 70%;">redirection(>,<) change stream's direction for a process</div>

```bash
$ [command] > [file_name]        # overwrite with stdout
$ [command] 1> [file_name]       # overwrite with stdout
$ [command] 2> [file_name]       # overwrite with stderr

$ [command] >> [file_name]       # add text of stdout at the end
$ [command] 1>> [file_name]      # add text of stdout at the end
$ [command] 2>> [file_name]      # add text of stderr at the end

$ [command] 1>&2                 # stdout to stderr
$ [command] 2>&1                 # stderr to stdout
$ [command] >& [file_name]       # overwrite with stdout and stderr

$ [command] < [file_name] 
$ [command] < [file_name_1] > [file_name_2]
```

<br><br><br>

#### Pipe

<div style="font-size: 70%;">pipe(|) connect a process to another process</div>

```bash
:/etc$ ls | grep issue
```
```
issue
issue.net
```
<br><br><br>

---

### ***Process and Binary***

#### background

```bash
$ ./[file_name] &                       # execute on background
$ jobs                                  # list of process
$ bg                                    # re-execute last stoped process on background
$ bg [job_number_of_stoped_process]     # re-execute stoped process on background
$ kill -9 %[number_of_process]          # terminate process  
```
<br><br><br>

#### foreground

- <b>ctrl + c</b> : terminate process
- <b>ctrl + z</b> : puase process

```bash
$ jobs
$ fg                                 # execute process denoted '+' on foreground
$ fg %[number_of_process]            # execute process corresponding number on foreground
```

<br><br><br>

#### management of process

<a href="https://www.techonthenet.com/linux/commands/ps.php" taraget="_blank">URL</a><br>
<span class="frmae3">process structure</span><br>
![image](https://user-images.githubusercontent.com/52376448/69005885-74908380-096b-11ea-804e-a662688a0d4b.png)

- stack
- heap
- BSS : uninitialized
- DATA : initialized
- TEXT : code

```bash
$ ps             # displaying list of process for me
$ ps -a          # displaying list of process for all users(Select all processes except both session leaders and processes not associated with a terminal.)
$ ps -u          # displaying details about owner of process
$ ps -l          # displaying details about process
$ ps -x          # displaying deamon process
$ ps aux         # frequently used(-a, -u, -x)
                 # ps aux | more
                 # ps aux | grep [pattern]
                 # '$ top' or '$ htop' 
$ ps -e          # displaying list of all process
$ ps -f          # displaying relationship about process
```
```bash
$ kill -9 [pid]
```
<br><br><br>

#### deamon process
<span class="frame3">/etc/init.d/</span><br>
```bash
$ service --status-all
$ service [process] start
$ service [process] status
$ service [process] stop
$ service [process] restart
```
<br>
<span class="frame3">/etc/rc3.d/</span><br>
auto-start : linking
<br><br><br>

---

### ***File System***

#### linking
```bash
$ copy [origin_file_name] [new_file_name]       # copy : different inode for origin file,
                                                #        be live after deleting origin file
$ ln [origin_file_name] [new_file_name]         # hard : equal inode for origin file,
                                                #        be live after deleting origin file
$ ln -s [origin_file_name] [new_file_name]      # soft : different inode for origin file,
                                                #        be die after deleting origin file
```
<br><br><br>

#### Specific files
<span class="frame3">device</span><br>

- block device
- character device


<br><br><br>
<hr class="division2">

## **Key factors on system programming**

### ***System call***
#### C compiler
<span class="frame3">gcc(GNU cc) installation</span>
```bash
$ apt update
$ apt upgrade
$ apt install gcc
$ gcc --version
```
<details markdown="1">
<summary class='jb-small' style="color:blue">gcc usage</summary>
<hr class='division3'>
```bash
$ gcc -o [.c_file_name] [.exe_file_name]
```
<hr class='division3'>
</details>

<br><br><br>

---

### ***API and ABI(Application Binary Interface)***

<br><br><br>

---

### ***Standard***
Linux supports POSIX and ANSI C

- POSIX, Portable Operating System Interface
- ANSI C, American National Standards Institute C Language standard

<br><br><br>

<hr class="division2">

## **Management of process**
![image](https://user-images.githubusercontent.com/52376448/69009159-14630700-0996-11ea-8024-72efeef49f4a.png)
<br><br><br>



### ***Process ID***
/proc/

- program
- process : executing program
- thread

<br>
maximun of PID, usually signed 16 bit = $$2^{15}$$ = 32768
```bash
$ cat /proc/sys/kernel/pid_max
```
<br>
First process, PID1, init process(/sbin/init)<br>
```bash
$ ps -fe
```
<br>

parent process(<b>ppid</b>) <-> child process(<b>pid</b>)<br><br>
<span class="frame3">origin</span><br>
```c
#include <sys/types.h>
#include <unistd.h>
pid_t getpid (void);
pid_t getppid (void);
```
<br>
<span class="frame3">practice</span><br>
```c
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

int main(){
        printf("pid=%d\n", getpid());
        printf("ppid=%d\n", getppid());
        return 0;
}

// $ ps
```
<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>
![image](https://user-images.githubusercontent.com/52376448/69005831-c08ef880-096a-11ea-931a-05b9165abfd4.png)
<hr class='division3'>
</details>

---

### ***Create process(system call) 1 : fork() and exec()***

- fork() : copy and paste a process space / parent process is conserved
- exec() : overwrite TEXT, DATA, BSS on process space / parent process dissappear


<br><br><br>

---


### ***Create process(system call) 2 : wait() and copy()***


---

### ***Terminate process(system call) : exit()***

<br><br><br>

<hr class="division2">

## **IPC, Inter-Process Communication**

<br><br><br>

<hr class="division2">

## **Signal action**

|signal|action|
|:--|:--|
|SIGKILL||
|SIGALARM||
|SIGSTP||
|SIGCONT||
|SIGINT||
|SIGSEGV||

```bash
$ kill -l
```

<br><br><br>

### ***Transfer signal***
```c
#include <sys/types.h>
#include <signal.h>
// pid : 프로세스의 PID
// sig : 시그널 번호
int kill(pid_t pid, int sig);
```
```bash
$ ./loop &
$ ./sigkill 1806 2
$ ps
```
<br><br><br>

### ***Signal action***

```c
#include <signal.h>

void (*signal(int signum, void (*handler)(int)))(int);

// 예1
// void (*handler)(int): SIG_IGN - 시그널 무시, SIG_DFL - 디폴트 동작
signal(SIGINT, SIG_IGN);

// 예2
// SIGINT 시그널 수신 시, signal_handler 함수를 호출
signal(SIGINT, (void *)signal_handler);
```


<br><br><br>
### ***Signal handler***
```c
static void signal_handler (int signo) {
    printf("Catch SIGINT!, but no stop\n");
}

int main (void) {
    if (signal (SIGINT, signal_handler) == SIG_ERR) {
        printf("Can't catch SIGINT!\n");
        exit (1);
    }
    for (;;)
    pause();
    return 0;
}
```
```bash
$ ./sigloop &
$ ./sigkill 1894 2
$ ps
$ kill -9 1894
```
<br><br><br>

### ***Process PCB(Process control block)***
![image](https://user-images.githubusercontent.com/52376448/69009194-728fea00-0996-11ea-8191-f20a9a261a1a.png)

<br><br><br>
<hr class="division2">

## **Shell script**
### ***.sh file***

#### Basic
```bash
#!/bin/bash
```
```bash
$ chmod u+x [file_name]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Example</summary>
<hr class='division3'>
```bash
#!/bin/bash

echo "Hello bash"
```
<hr class='division3'>
</details>

<br><br><br>



#### Comment
```bash
#!/bin/bash

# comment
```
<br><br><br>

#### $Variable
```bash
#!/bin/bash

mysql_id='root'
mysql_directory='/etc/mysql'

echo $mysql_id
echo $mysql_directory
```
<b>Caution</b> : do not put space

<br><br><br>


#### List = ${Variable[Index]}
```bash
#!/bin/bash

daemons=("httpd" "mysqld" "vsftpd")
echo ${daemons[1]}             # $daemons 배열의 두 번째 인덱스에 해당하는 mysqld 출력
echo ${daemons[@]}             # $daemons 배열의 모든 데이터 출력
echo ${daemons[*]}             # $daemons 배열의 모든 데이터 출력
echo ${#daemons[@]}            # $daemons 배열 크기 출력

filelist=( $(ls) )             # 해당 쉘스크립트 실행 디렉토리의 파일 리스트를 배열로 $filelist 변수에 입력
echo ${filelist[*]}            # $filelist 모든 데이터 출력
```
<br><br><br>


#### Pre-defined local variables

|Local variables|Description|
|:--|:--|
|$$||
|$0 ||
|$1~$9 ||
|$* ||
|$# ||
|$? ||

- 0(success), 1~125(error)
- 126(not executable)
- 128~ 255(generated signal)

<br><br><br>

#### Operator, \`expr\`
```bash
#!/bin/bash

num=`expr \( 3 \* 5 \) / 4 + 7`
echo $num
```
<b>Caution 1</b> : do put space among all of words, numbers, symbols<br>
<b>Caution 2</b> : Arithmetics operator [ +, -, \\*, / ] <br>
<b>Caution 3</b> : associative symbol \(, \)
<br><br><br>

#### Selection statements
```bash
#!/bin/bash


# if 1
if [ conditional sentence ]
then
    [command]
fi

# if 2
if [ conditional sentence ]; then [command]; fi


# if else
if [ conditional sentence ]
then
    [command]
else
    [command]
fi
```
<details markdown="1">
<summary class='jb-small' style="color:blue">Example</summary>
<hr class='division3'>
```bash
$ if [ -z $1 ]; then echo "Insert arguments"; fi
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">Conditional sentence</summary>
<hr class='division3'>
<span class="frame3">Compare words</span><br>

- 문자1 == 문자2 # 문자1 과 문자2가 일치
- 문자1 != 문자2 # 문자1 과 문자2가 일치하지 않음
- -z 문자 # 문자가 null 이면 참
- -n 문자 # 문자가 null 이 아니면 참

<br><br><br>
<span class="frame3">Compare numbers</span><br>

- 값1 -eq 값2 # 값이 같음(equal)
- 값1 -ne 값2 # 값이 같지 않음(not equal)
- 값1 -lt 값2 # 값1이 값2보다 작음(less than)
- 값1 -le 값2 # 값1이 값2보다 작거나 같음(less or equal)
- 값1 -gt 값2 # 값1이 값2보다 큼(greater than)
- 값1 -ge 값2 # 값1이 값2보다 크거나 같음(greater or equal)

<br><br><br>
<span class="frame3">Inspect files</span><br>

- -e[file_name]     #파일이존재하면참
- -d[file_name]     #파일이디렉토리면참
- -h[file_name]     #심볼릭링크파일
- -f[file_name]     #파일이일반파일이면참
- -r[file_name]     #파일이읽기가능이면참
- -s[file_name]     #파일크기가0이아니면참
- -u[file_name]     #파일이set-user-id가설정되면참
- -w[file_name]     #파일이쓰기가능상태이면참
- -x[file_name]     #파일이실행가능상태이면참

<br><br><br>
<span class="frame3">Logical operation</span><br>

- 조건1 -a 조건2           # AND
- 조건1 -o 조건2           # OR
- 조건1 && 조건2           # 양쪽 다 성립
- 조건1 || 조건2           # 한쪽 또는 양쪽다 성립
- !조건                    # 조건이 성립하지 않음
- true                    # 조건이 언제나 성립
- false                   # 조건이 언제나 성립하지 않음

<hr class='division3'>
</details>

<br><br><br>

#### Iteration statements
```bash
#!/bin/bash


# for 1
for [variable] in value1 value2 ...
do
    [command]
done

# for 2
for [variable] in value1 value2 ...; do [command]; done


# while
while [ conditional sentence ]
do
    [command]
done
```
<details markdown="1">
<summary class='jb-small' style="color:blue">for, Example</summary>
<hr class='division3'>
```bash
#!/bin/bash

for database in $(ls)
do
    echo $database
done
```
```bash
#!/bin/bash

for database in $(ls); do
    echo $database
done
```
```bash
#!/bin/bash

for database in $(ls); do echo $database; done
```
<hr class='division3'>
</details>
<details markdown="1">
<summary class='jb-small' style="color:blue">while, Example</summary>
<hr class='division3'>
```bash
#!/bin/bash

lists=$(ls)
num=${#lists[@]}
index=0
while [ $num -ge 0 ]
do
    echo ${lists[$index]}
    index=`expr $index + 1`
    num=`expr $num - 1`
done
```
<hr class='division3'>
</details>

<br><br><br>

---

### ***Examples***
#### backup
```bash
#!/bin/bash

if [ -z $1 ]||[ -z $2 ]; then
    echo usage: $0 sourcedir targetdir
else
    SRCDIR=$1
    DSTDIR=$2
    BACKUPFILE=backup.$(date +%y%m%d%H%M%S).tar.gz
    if [ -d $DSTDIR ]; then
        tar -cvzf $DSTDIR/$BACKUPFILE $SRCDIR
    else
        mkdir $DSTDIR
        tar -cvzf $DSTDIR/$BACKUPFILE $SRCDIR
    fi
fi
```
<br><br><br>

#### clean log file
```bash
#!/bin/bash

LOGDIR=/var/log
GZIPDAY=1
DELDAY=2
cd $LOGDIR
echo "cd $LOGDIR"

sudo find . -type f -name '*log.?' -mtime +$GZIPDAY -exec bash -c "gzip {}" \; 2> /dev/null
sudo find . -type f -name '*.gz' -mtime +$DELDAY -exec bash -c "rm -f {}" \; 2> /
```

<br><br><br>
<hr class="division2">

## **Thread**
### ***Pthread***
<br><br><br>

<hr class="division2">

## **System programming**

<br><br><br>

<hr class="division2">

<hr class="division1">

List of posts followed by this article
- [post1](https://userdyk-github.github.io/)
- <a href='https://userdyk-github.github.io/'>post2</a>
- <a href='https://userdyk-github.github.io/'>post3</a>

---

Reference
- <a href="https://repl.it/languages/c" target="_blank">Implementation with C on web</a>
- <a href="https://repl.it/languages/bash" target="_blank">Implementation with bash on web</a>
- <a href='https://www.youtube.com/playlist?list=PLuHgQVnccGMBT57a9dvEtd6OuWpugF9SH' target="_blank">생활코딩</a>
- <a href="https://www.techonthenet.com/index.php" target="_blank">tech on the net</a>


---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>



