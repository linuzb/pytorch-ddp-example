## 简介

本配置在 Kubernetes 下执行

### 前置准备

1. 部署[Kubernetes](https://kubernetes.io/docs/setup/production-environment/tools/)
2. 安装[kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl)
3. 部署[traing-operator](https://www.kubeflow.org/docs/components/training/installation/#installing-training-operator)

### 部署

```shell
kubectl apply -f deploy/kubernetes/pytorch_job.yaml
```

### 卸载

```shell
kubectl delete -f deploy/kubernetes/pytorch_job.yaml
```