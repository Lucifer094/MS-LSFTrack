# 公开上传前检查清单

本目录已经整理为可公开代码版本，但正式上传 GitHub 前仍建议人工确认以下事项：

- [x] 已将 BoxMOT ReID baselines 所需的公开权重打包到 `weights/reid/osnet_x0_25_msmt17.pt`。
- [ ] 选择并添加开源 license，例如 MIT、Apache-2.0 或自定义非商用协议。
- [ ] 在 README 中补充论文标题、作者、单位、BibTeX。
- [ ] 确认数据集是否公开，以及数据下载链接是否可以放在 README。
- [ ] 确认 `third_party/unified_eval_suite` 的来源与许可说明。
- [ ] 将 `results/MS-LSFTrack-results` 压缩并上传到 GitHub Release、网盘或其他下载位置。
- [ ] 上传结果包后，把下载链接写回 `README.md`。
- [ ] 如果不公开 checkpoint，需要从 README 中移除 checkpoint 复现命令或改成“训练后运行”。
