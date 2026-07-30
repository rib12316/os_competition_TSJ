# 竞赛交付物占位清单

本目录为最终竞赛材料预留固定路径。当前代码交付不包含下列二进制材料；参赛团队确认最终版本后再逐项上传并单独提交。

| 交付物 | 固定路径 | 当前状态 | 要求 |
|---|---|---|---|
| 设计文档 | `submission/design-document.pdf` | 待上传 | 最终定稿 PDF |
| 参赛承诺书 | `submission/competition-commitment.pdf` | 待上传 | 按大赛模板签署并转为 PDF |
| 演示 PPT | `submission/demo-slides.pptx` | 待上传 | 可编辑 PPTX；需要时可同时附 PDF |
| 演示视频 | `submission/demo-video.mp4` | 待上传 | MP4，文件大小不得超过 100 MB |

## 上传前检查

1. 文件名和路径与上表一致，避免使用“最终版2”“new”等临时命名。
2. 文档中的项目名称、团队信息、软硬件版本和 README 保持一致。
3. PPT 中的实验数字能够追溯到 `docs/` 中的对应报告。
4. 视频使用常见 H.264/AAC 编码，建议控制在 95 MB 以内，为平台计量差异留出余量。
5. 材料不得包含 API key、访问令牌、个人账号口令、模型权重或未脱敏日志。

视频大小可在仓库根目录检查：

```bash
bytes=$(stat -c %s submission/demo-video.mp4)
test "$bytes" -le 100000000
```

全部材料上传后，建议执行：

```bash
find submission -maxdepth 1 -type f -printf '%10s  %f\n' | sort
git status --short
```

确认无误后再单独提交竞赛材料，以便代码版本与大文件材料在历史中清晰区分。
