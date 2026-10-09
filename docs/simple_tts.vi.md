# Simple TTS: Turbo, Rubber Band và hiệu ứng giọng nói

Chạy `run_simple_tts.bat`. Tool riêng với hai tab: **Văn bản** và **Thử công nghệ**.

## Sử dụng

1. Chọn giọng đọc Turbo, nhập văn bản rồi bấm **Tạo bản gốc**.
2. Trong **Thử công nghệ**, chọn **Rubber Band chất lượng cao** để đổi tốc độ từ 0.5–2.0x, hoặc chọn **Không đổi tốc độ** và giữ 1.0x.
3. Bật/tắt **EQ sáng nhẹ**, **Compressor** và **Giảm đỉnh âm lượng** theo nhu cầu.
4. Bấm **Thử trên bản gốc**. Mỗi lần xử lý dùng bản gốc, không xử lý chồng lên kết quả trước.
5. Dùng **Nghe bản gốc** và danh sách kết quả / **Nghe file** để so sánh.

Có thể **Mở WAV để thử** với file mono có sẵn. WAV và JSON thiết lập được lưu ở `outputs/simple_tts/`. Danh sách so sánh thuộc phiên hiện tại; các file vẫn giữ lại sau khi đóng app.

## Xử lý âm thanh

- **Rubber Band qua Pedalboard**: chất lượng cao, giữ cao độ; 1.0x bỏ qua xử lý tốc độ.
- **EQ**: high-pass 65 Hz, high-shelf +1.5 dB tại 3.5 kHz.
- **Compressor**: threshold -18 dB, ratio 2:1, attack 15 ms, release 120 ms.
- **Giảm đỉnh**: giảm gain toàn file nếu peak vượt -1 dBFS; không khuếch đại và không cắt đỉnh.

Thứ tự: Rubber Band → EQ/compressor → giảm đỉnh. EQ/compressor chạy cùng một chuỗi Pedalboard. Mặc định tắt hiệu ứng. WAV float32 giữ các đỉnh vượt 0 dBFS khi lưu; bật giảm đỉnh nếu cần tránh quá mức khi phát hoặc chuyển sang PCM. EQ hoặc giảm âm lượng không phục hồi lỗi đã có trong âm gốc.

## Tối ưu

Mô hình và danh sách giọng được dùng lại trong suốt phiên chạy. Bản gốc gần nhất được giữ trong RAM để thử hiệu ứng mà không đọc lại WAV hoặc tạo lại lời đọc. Với WAV ngoài, thay đổi kích thước hoặc thời gian sửa file sẽ làm cache được tải lại. Bộ nhớ đệm chỉ giữ một bản gốc, không tích lũy audio theo lịch sử.

Tắt các hiệu ứng không cần dùng để giảm xử lý. Chế độ bỏ qua không sao chép toàn bộ mảng audio. Giữ nguyên tham số tổng hợp mặc định của SDK và chất lượng cao của Rubber Band; chưa đo mức tăng tốc của mô hình thật.

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

TTS dùng model giả trong kiểm tra; Rubber Band, EQ và compressor được chạy thật với tín hiệu tổng hợp.

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
        bright=True, compress=True, peak_guard=True,
    ),
)
print(result.path)
# result.audio: mảng float32; result.sample_rate: sample rate audio
```

Bỏ `options` để giữ âm gốc. `voice` nhận ID giọng Turbo hoặc None để dùng giọng mặc định. Hàm tự tải Turbo lần đầu, dùng lại ở các lần sau và đóng model khi tiến trình kết thúc. Có thể truyền `model=engine` đã tải; khi đó người gọi chịu trách nhiệm đóng engine và điều phối gọi đồng thời.

Giao diện cũng gọi hàm này, truyền model đang dùng và giữ audio trả về trong RAM để thử hiệu ứng.
