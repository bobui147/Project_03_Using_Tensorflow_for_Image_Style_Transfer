# Project 03 — Using TensorFlow for Image Style Transfer

Project Neural Style Transfer (NST) hoàn chỉnh bằng TensorFlow 2. Chương trình nhận một ảnh nội dung và một ảnh phong cách, sau đó tạo ảnh mới giữ bố cục của ảnh nội dung nhưng mang màu sắc và texture của ảnh phong cách.

## Tính năng

- `classic`: thuật toán Gatys sử dụng VGG19, Gram matrix và tối ưu trực tiếp pixel bằng `tf.GradientTape`.
- `fast`: model Arbitrary Image Stylization có sẵn trên TensorFlow Hub.
- Tự resize ảnh mà vẫn giữ tỷ lệ.
- Chạy CPU hoặc GPU, có kiểm tra thiết bị rõ ràng.
- Ghi lịch sử loss và thông tin lần chạy ra JSON.
- Có thể lưu ảnh trung gian trong quá trình tối ưu.
- Có unit test và smoke test không cần tải trọng số VGG19.

Project không cần dataset lớn. Bạn chỉ cần một ảnh content và một ảnh style.

## Cấu trúc

```text
Project_03_Using_Tensorflow_for_Image_Style_Transfer/
├── style_transfer/
│   ├── config.py
│   ├── engine.py
│   ├── hub_transfer.py
│   ├── image_utils.py
│   └── model.py
├── samples/
│   ├── content/
│   └── style/
├── outputs/
├── tests/
├── check_environment.py
├── download_samples.py
├── run_style_transfer.py
├── smoke_test.py
└── requirements.txt
```

## Cài đặt trên Windows

Mở PowerShell tại thư mục repository. Môi trường được đặt ở thư mục cha với tên ngắn để tránh giới hạn độ dài đường dẫn của TensorFlow trên Windows:

```powershell
py -3.11 -m venv ..\.venv-style-transfer
..\.venv-style-transfer\Scripts\python.exe -m pip install --upgrade pip
..\.venv-style-transfer\Scripts\python.exe -m pip install -r requirements.txt
..\.venv-style-transfer\Scripts\python.exe check_environment.py
```

TensorFlow hiện đại chạy CPU trên Windows native. Muốn dùng NVIDIA GPU, nên chạy project trong Linux hoặc WSL2 và cài `tensorflow[and-cuda]` theo [hướng dẫn chính thức](https://www.tensorflow.org/install/pip).

## Tải ảnh mẫu

```powershell
..\.venv-style-transfer\Scripts\python.exe download_samples.py
```

Hai ảnh công khai từ tutorial TensorFlow sẽ được lưu tại:

- `samples/content/labrador.jpg`
- `samples/style/kandinsky.jpg`

## Smoke test

Smoke test tạo ảnh tổng hợp 64×64 và chạy đúng hai bước với VGG19 random weights. Nó kiểm tra toàn bộ pipeline mà không tải model hoặc chạy tối ưu dài:

```powershell
..\.venv-style-transfer\Scripts\python.exe smoke_test.py
```

Chạy unit test:

```powershell
..\.venv-style-transfer\Scripts\python.exe -m pytest -q
```

## Chạy nhanh bằng TensorFlow Hub

Lần đầu model sẽ được tải từ TensorFlow Hub:

```powershell
..\.venv-style-transfer\Scripts\python.exe run_style_transfer.py `
  --content samples\content\labrador.jpg `
  --style samples\style\kandinsky.jpg `
  --output outputs\fast_result.png `
  --method fast
```

## Chạy thuật toán VGG19 chuẩn

Trước tiên chạy ngắn 10 bước để kiểm tra:

```powershell
..\.venv-style-transfer\Scripts\python.exe run_style_transfer.py `
  --content samples\content\labrador.jpg `
  --style samples\style\kandinsky.jpg `
  --output outputs\classic_smoke.png `
  --method classic `
  --max-dim 256 `
  --steps 10 `
  --log-every 1
```

Khi smoke test thành công, chạy chất lượng cao hơn:

```powershell
..\.venv-style-transfer\Scripts\python.exe run_style_transfer.py `
  --content samples\content\labrador.jpg `
  --style samples\style\kandinsky.jpg `
  --output outputs\classic_result.png `
  --method classic `
  --max-dim 512 `
  --steps 400 `
  --save-every 100
```

Lần chạy classic đầu tiên sẽ tải VGG19 pretrained weights. Không nên chạy hàng trăm bước trước khi smoke test thành công.

## Các tham số quan trọng

- `--content-weight`: mức độ giữ bố cục của ảnh content, mặc định `1e4`.
- `--style-weight`: mức độ áp dụng style, mặc định `1e-2`.
- `--variation-weight`: làm ảnh mượt và giảm nhiễu, mặc định `30`.
- `--steps`: số bước tối ưu của phương pháp classic.
- `--max-dim`: cạnh tối đa; giảm xuống 256 nếu thiếu RAM/VRAM.
- `--init content|noise`: khởi tạo từ ảnh content hoặc nhiễu.
- `--device auto|cpu|gpu`: chọn thiết bị chạy.
- `--save-every N`: lưu ảnh trung gian sau mỗi N bước.

Sau mỗi lần chạy, chương trình tạo:

```text
outputs/result.png
outputs/result.metrics.json
```

File JSON chứa thời gian chạy, thiết bị, phiên bản TensorFlow và lịch sử loss.

## Thuật toán

VGG19 được đóng băng và chỉ dùng để trích xuất đặc trưng:

- Content representation: activation của `block5_conv2`.
- Style representation: Gram matrix tại `block1_conv1` đến `block5_conv1`.
- Total loss: `content loss + style loss + total variation loss`.

Ở mỗi bước, TensorFlow tính gradient của total loss theo chính tensor ảnh kết quả, cập nhật ảnh bằng Adam và clip pixel về `[0, 1]`. Đây là cách triển khai theo [tutorial Neural Style Transfer chính thức của TensorFlow](https://www.tensorflow.org/tutorials/generative/style_transfer).

Chế độ `fast` dùng model Magenta `arbitrary-image-stylization-v1-256/2` theo [tutorial TensorFlow Hub](https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization).

