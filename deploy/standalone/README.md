## 简介

本脚本在单机环境下部署测试

## 部署运行

### docker 运行

```shell
chmod +x deploy/standalone/run_docker.sh

# 第一个参数为镜像地址
./deploy/standalone/run_docker.sh registry.cn-hangzhou.aliyuncs.com/linuzb/pytorch:pytorch-ddp-01fcd18
```

### shell 运行

```shell
chmod +x deploy/standalone/run_shell.sh

./deploy/standalone/run_shell.sh
```