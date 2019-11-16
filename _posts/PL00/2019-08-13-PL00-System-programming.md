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
<hr class='division3'>
</details>
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


```bash
$ find 
```
```bash
$ cat [file_name]
$ head [file_name]
$ more [file_name]      # spacebar : next page
                        # enter : next line
$ less [file_name]                        
$ tail [file_name]
```
```bash
$ rm [file_name]
$ rm -rf [folder_name]
```
<br><br><br>


### ***Process on Linux***

<br><br><br>

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

<span class="frame3">create another user id on linux</span>
```bash
$ adduser [user_name]
```
<details markdown="1">
<summary class='jb-small' style="color:blue">If you concretely create user id, setting details of user</summary>
<hr class='division3'>
```bash
$ useradd [user_name]
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

<span class="frame3">Group management</span><br>
/etc/group
```bash
$ groupadd [group_name]                     # create group
$ usermod -a -G [group_name] [user_name]    # add user to group
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

<a href="https://www.techonthenet.com/linux/commands/ps.php" taraget="_blank">URL</a>

```bash
$ ps             # displaying list of process for me
$ ps -a          # displaying list of process for all users
$ ps -u          # displaying details about owner of process
$ ps -l          # displaying details about process
$ ps -x          # displaying deamon process
$ ps aux         # frequently used(-a, -u, -x)
                 # ps aux | more
                 # ps aux | grep [pattern]
                 # '$ top' or '$ htop' 
$ ps -e          # displaying environment variables of process
$ ps -f          # displaying relationship about process
```
```bash
$ kill -9 [pid]
```
<br><br><br>

#### deamon process
<span class="frame3">/etc/init.d/</span><br>
```bash
$ service [process] start
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

<br><br><br>

---

### ***API***

<br><br><br>

---

### ***ABI and Standard***

<br><br><br>

<hr class="division2">

## **Management of process**

<br><br><br>

<hr class="division2">

## **IPC, Inter-Process Communication**

<br><br><br>

<hr class="division2">

## **Signal action**

<br><br><br>

<hr class="division2">

## **Shell script**

<br><br><br>

<hr class="division2">

## **Thread**

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
- <a href='https://www.youtube.com/playlist?list=PLuHgQVnccGMBT57a9dvEtd6OuWpugF9SH' target="_blank">생활코딩</a>
- <a href="https://www.techonthenet.com/index.php" target="_blank">tech on the net</a>

---

<details markdown="1">
<summary class='jb-small' style="color:blue">OUTPUT</summary>
<hr class='division3'>

<hr class='division3'>
</details>



