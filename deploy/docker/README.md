## 简介

本脚本在单机 docker 环境下部署测试

首先修改 `install.sh` 中的镜像为最新镜像
```shell
IMG=registry.cn-hangzhou.aliyuncs.com/linuzb/pytorch:pytorch-ddp-01fcd18
```

```shell
chmod +x install.sh
./install.sh
```