## Zhaoyang's Page
This is Zhaoyang's code commit folder.

## About Git-Guidance
I have wrote a basic [guidance](Git-Guidance.pdf) for who are first using git.

1.如何使用git呢？
```
首先下载git并配置环境变量
检测是否配置好环境可以打开cmd或者powershell
输入 git 出现git的配置信息即代表环境变量配置成功
```
![git](https://raw.githubusercontent.com/Raibows/MarkdownPhotos/master/ncepu-summer-2019/1.png)
2.如何将别人的项目下载到本地呢？
```
在命令行中使用 git clone https://github.com/kicky104/ncepu-summer-2019
后面的一串链接就是项目的地址，如下图所示
```
![tip](https://raw.githubusercontent.com/Raibows/MarkdownPhotos/master/ncepu-summer-2019/2.png)
3.我作了一些修改，如何提交呢？
```
首先使用 git add . 选择提交的范围， add . 代表所有
接着将更改提交到本地的暂存区，使用 git commit -m "description about this commit"
最后使用 git push origin master  这个操作的意思是将本地的一个叫master分支，提交到一个叫origin的远程服务器并与它已经建立联系的远程分支（remote master）,如果远程没有和本地的已经建立联系的，那么该操作会尝试创建一个remote master远程分支
```
4.我已经clone了项目，可是项目更新了，如何将更新拉取到我本地呢？
```
使用 git pull origin master:master
该命令的意思是从一个叫origin服务器的地方，选取master（remote master）分支，和自己本地的叫master的分支合并，自然而然本地的master分支就与远程的master同步了。
```
5.为什么我会提交失败？
```
设想一个简单的场景，一个项目由2个人A、B协作完成，
首先A、B都clone项目到了本地。
A将clone到本地的项目删除了一个文件，将本此操作提交到了GitHub远程端，远程端自然而然也将该文件删除，push成功
B将clone到本地的项目添加了一个文件，此时想要push提示失败
为什么呢？因为如果想要push，必须要保证自己的项目是最新的，再修改，才能提交更新
```
6.如何解决上述提交失败问题呢？
```
每次提交前使用 git pull命令，拉取更新，确保自己的项目是最新的
```
7.如果两人同时对某一个文件修改了怎么办？
```
为了避免这种情况，我们必须
不修改公共区文件夹
创建属于自己的文件夹
只修改属于自己文件夹的内容
```