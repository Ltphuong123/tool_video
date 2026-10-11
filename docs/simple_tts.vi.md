# Simple TTS: Turbo và Rubber Band

Chạy `run_simple_tts.bat`. Tool riêng với hai tab: **Văn bản** và **Thử công nghệ**.

## Sử dụng

1. Chọn giọng đọc Turbo, nhập văn bản rồi bấm **Tạo bản gốc**.
2. Trong **Thử công nghệ**, chọn **Rubber Band chất lượng cao** để đổi tốc độ từ 0.5–2.0x, hoặc chọn **Không đổi tốc độ** và giữ 1.0x.
3. Bấm **Thử trên bản gốc**. Mỗi lần xử lý dùng bản gốc, không xử lý chồng lên kết quả trước.
4. Dùng **Nghe bản gốc** và danh sách kết quả / **Nghe file** để so sánh.

Có thể **Mở WAV để thử** với file mono có sẵn. WAV và JSON thiết lập được lưu ở `outputs/simple_tts/`. Danh sách so sánh thuộc phiên hiện tại; các file vẫn giữ lại sau khi đóng app.

## Xử lý âm thanh

- **Rubber Band qua Pedalboard**: chất lượng cao, giữ cao độ; 1.0x bỏ qua xử lý tốc độ.

Audio giữ biên độ gốc của model, chỉ chỉnh tốc độ khi được yêu cầu. WAV float32 giữ cả các đỉnh vượt 0 dBFS khi lưu.

## Tối ưu

Mô hình và danh sách giọng được dùng lại trong suốt phiên chạy. Bản gốc gần nhất được giữ trong RAM để thử tốc độ mà không đọc lại WAV hoặc tạo lại lời đọc. Với WAV ngoài, thay đổi kích thước hoặc thời gian sửa file sẽ làm cache được tải lại. Bộ nhớ đệm chỉ giữ một bản gốc, không tích lũy audio theo lịch sử.

Tốc độ 1.0x bỏ qua xử lý và không sao chép toàn bộ mảng audio. Giữ nguyên tham số tổng hợp mặc định của SDK và chất lượng cao của Rubber Band.

## Cài thư viện

Pedalboard đã cài trong môi trường hiện tại. Cài lại bằng:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-simple-tts-audio.txt
```

## Kiểm tra

```powershell
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -p "test_simple_tts*.py" -v
.\\.venv\\Scripts\\python.exe simple_tts.py --check
```

TTS dùng model giả trong kiểm tra; Rubber Band được chạy thật với tín hiệu tổng hợp.

## Gọi một hàm để tạo giọng

Hàm nằm ở `apps/simple_tts_engine.py`, không phụ thuộc Tkinter. Chạy từ thư mục dự án (cài SDK hoặc đặt PYTHONPATH=src):

```python
from apps.simple_tts_engine import generate_speech
from apps.simple_tts_audio import AudioOptions

result = generate_speech(
    text="Xin chào, đây là giọng đọc Turbo.",
    output_dir="outputs/simple_tts",
    options=AudioOptions(
        tempo="rubberband", speed=1.2,
    ),
)
print(result.path)
# result.audio: mảng float32; result.sample_rate: sample rate audio
```

Bỏ `options` để giữ âm gốc. `voice` nhận ID giọng Turbo hoặc None để dùng giọng mặc định. Hàm tự tải Turbo lần đầu, dùng lại ở các lần sau và đóng model khi tiến trình kết thúc. Có thể truyền `model=engine` đã tải; khi đó người gọi chịu trách nhiệm đóng engine và điều phối gọi đồng thời.

Giao diện cũng gọi hàm này, truyền model đang dùng và giữ audio trả về trong RAM để thử tốc độ.
