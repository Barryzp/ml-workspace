// 获取序列中图像的数量
n = nSlices();
for (i = 10; i <= n; i++) {
  // 设置当前图像
  setSlice(i);
  // 构建文件名
  path = "D:/workspace/ml-workspace/registration/datasets/sample4/ct/s1/slice_" + i + ".bmp"; // 修改路径和文件名格式
  // 保存当前图像
  saveAs("BMP", path);
}