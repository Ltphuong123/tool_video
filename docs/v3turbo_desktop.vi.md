# VieNeu v3 Turbo Studio — ứng dụng desktop Python

Chạy trên Windows bằng Tkinter, không cần trình duyệt, dùng Python 3.10–3.13. Model tự tải trên luồng nền ngay khi mở tool. Giao diện hiển thị **Đang tải model…** và thanh tiến trình; sau khi tải xong có thể tạo giọng. Lần đầu có thể cần tải dữ liệu model về máy. Nếu tải lỗi, vào **Cấu hình** kiểm tra thiết lập rồi bấm **Tải model** để thử lại.

## Khởi chạy

Trong thư mục dự án:

```powershell
uv sync --extra srt-quality --extra video
uv run --extra srt-quality --extra video python v3turbo_desktop.py
```

Cũng có thể bấm đúp `run_v3turbo_desktop.bat`. Nếu `.venv` đã có, launcher dùng Python trong môi trường đó. Sau khi cài package có thể chạy `vieneu-desktop`.

Để chạy CUDA:

```powershell
uv sync --extra cuda --extra srt-quality --extra video
uv run --extra cuda --extra srt-quality --extra video python v3turbo_desktop.py
```

Với môi trường `.venv` đã cài đủ các chức năng desktop, có thể bổ sung CUDA mà giữ các thư viện khác:

```powershell
uv pip install --python .venv/Scripts/python.exe "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" "transformers>=4.57,<5" safetensors --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple --index-strategy unsafe-best-match
```

Đây là bộ [PyTorch 2.8.0 / CUDA 12.8 cho Windows](https://pytorch.org/get-started/previous-versions/#v280). Khởi động lại tool rồi chọn **Backend auto/PyTorch**, **Thiết bị auto/CUDA**, **Kiểu số GPU float32** và bấm **Tải model**. Dòng trạng thái ghi **PYTORCH / CUDA · float32** khi thực sự dùng GPU. FP32 là mặc định desktop; không tự chuyển INT8 hoặc đổi model để tăng tốc. Lượt đầu có thêm chi phí tải model và tạo CUDA graph, nên so sánh các lượt sau khi model đã sẵn sàng.

Đọc văn bản, SRT và vietsub dùng lại model đã tải. Khi đổi tốc độ, Rubber Band chất lượng cao chạy song song tối đa 4 luồng. Với nhiều nhóm câu trên CUDA, GPU tạo nhóm kế trong lúc CPU chỉnh tốc độ nhóm trước; chỉ giữ tối đa hai nhóm đang xử lý. Suy luận model và cập nhật tiến trình vẫn trên một luồng, giữ thứ tự câu và mốc SRT. ONNX/CPU xử lý theo thứ tự để tránh tranh CPU với model. Cấu hình FP32, batch và chất lượng âm thanh được giữ nguyên.

Đo sau khi model đã nóng trên RTX 5050: 12 câu, audio 31,298 giây, tốc độ 1,25×, batch 1, temperature 0 để đối chiếu chính xác. Trung vị hai lượt tuần tự là 6,288 giây; hai lượt có xử lý xen kẽ là 5,862 giây, giảm khoảng 6,8% thời gian. Audio PCM16 trùng từng mẫu và SRT trùng hoàn toàn giữa hai cách chạy. Mức cải thiện tùy độ dài, batch và nhu cầu đổi tốc độ; không đại diện cho mọi tác vụ. Báo cáo tại `.cache/studio_refresh/benchmark/report.json`.

Log sau mỗi audio + SRT ghi `backend`, `total`, `model`, `tempo`, `export`, `other` và `RTF`. `model` là tạo giọng; `tempo` là đổi tốc độ; `export` là ghi audio/SRT. Trên CUDA, thời gian `model` và `tempo` có thể chồng nhau, nên không cộng chúng để suy ra `total`; `other` là thời gian ngoài tạo giọng trong lượt thực tế. `RTF = thời gian xử lý / độ dài audio`, nhỏ hơn 1 nghĩa là tạo nhanh hơn thời gian phát. Không tính thời gian tải model vào log tạo audio; lượt tải và lượt CUDA graph đầu nên đánh giá riêng.

Đo tại máy RTX 5050 ngày 10/10/2026, cùng văn bản 3 câu, model v3 Turbo và FP32: ONNX/CPU tạo audio 7,82 giây trong 4,75 giây; PyTorch/CUDA tạo audio 7,88 giây trong 0,41 giây sau khởi tạo, khoảng 11,7 lần nhanh hơn về thời gian xử lý. Lượt CUDA đầu mất 1,63 giây, tải model mất khoảng 16 giây. Đây là đoạn mẫu ngắn với tốc độ đọc 1.0x, hiệu ứng tắt, batch 8; chưa đại diện cho mọi văn bản. CPU/GPU dùng bộ lấy mẫu khác nhau nên waveform không được coi là giống từng mẫu; cả hai giữ độ chính xác FP32 và cùng cấu hình tổng hợp.

Các dependency tùy chọn:

```powershell
uv sync --extra cuda --extra srt-quality --extra video --extra pdf --extra watermark --extra finetune
```

`pdf` thêm PyMuPDF; `watermark` thêm Resemble Perth; `finetune` thêm thư viện train LoRA. CPU chỉ cần `uv sync`. Trên Linux cần cài Tkinter của hệ điều hành và `sounddevice` để phát audio. Bản ghi micro tích hợp dùng Windows MCI.

`srt-quality` thêm Pedalboard có Rubber Band tích hợp để xử lý tốc độ cho cả SRT và các chức năng đọc khác. Tool dùng duy nhất Rubber Band; khi cần đổi tốc độ hoặc đọc SRT mà thiếu thư viện sẽ báo lỗi trước khi sinh audio. Đọc văn bản ở tốc độ gốc `1.0x` không cần chạy bộ chỉnh tốc độ. Nếu cập nhật môi trường đang dùng CUDA, giữ `--extra cuda` trong lệnh sync.

`video` thêm MoviePy 2 và imageio-ffmpeg để đọc/xuất video trong quy trình tự động vietsub. Các chức năng TTS không cần import MoviePy khi khởi động. Launcher dùng `.venv` sẵn có thì cần cài extra này vào môi trường đó trước khi chạy vietsub video.

## Giao diện và điều hướng

Thanh bên trái gồm năm trang: **Đọc văn bản → Đọc phụ đề SRT → Tự động vietsub → Lịch sử & file → Cấu hình**. Tool mở ở trang Văn bản và tự tải model; các nút chạy tác vụ được khóa trong lúc tải. Có thể đổi nơi lưu tại Cấu hình. Trang cấu hình có thanh cuộn. Trạng thái model và tiến trình tách khỏi khu vực nhập nội dung. Tab chỉnh tốc độ video riêng đã được gỡ; bước căn tốc độ vẫn chạy tự động trong quy trình vietsub.

Màu tím/xanh/xanh ngọc phân biệt thao tác và trạng thái. Hàng đợi Văn bản/SRT dùng bảng gọn, có phần trăm, số mục xong và màu riêng cho lỗi/đã dừng; **Ctrl/Shift** chọn nhiều mục, di chuột lên dòng để đọc đầy đủ thông tin. Giao diện giữ nút chạy/dừng ở vị trí dễ thấy tại 980×620 và 1280×850. **Ctrl+1…5** chuyển trang, **F5** làm mới kết quả, **Esc** dừng tác vụ. Trong ô văn bản, **Ctrl+Enter** thêm nội dung vào hàng đợi.

1. **Cấu hình**: nơi lưu, model/thiết bị, định dạng và tốc độ đọc ở ba nhóm riêng. Bật **Thiết lập model nâng cao / Tham số tạo giọng nâng cao** để xem các ô ONNX, luồng CPU, streaming, thử lại, temperature, top_k, top_p, phạt lặp, frame, ký tự và batch. Các giá trị đã có vẫn áp dụng khi thu gọn khung. Mã tham chiếu giữ đặc trưng giọng. Xuất WAV/FLAC/MP3; MP3 cần libsndfile có bộ mã hóa phù hợp.
2. **Văn bản**: nhập nội dung hoặc thêm nhiều TXT, chọn giọng và tốc độ ngay trong tab. Ô nội dung có bộ đếm ký tự. Mỗi mục hàng đợi xuất audio + SRT từng câu; bảng hàng đợi và lịch sử nằm cạnh nội dung, có thể kéo thanh chia để chỉnh độ rộng.
3. **SRT**: thêm nhiều file vào hàng đợi, xem tiến trình từng file và lịch sử ngay trong tab; luôn đọc theo mốc bắt đầu tuyệt đối của phụ đề và giữ các khoảng trống. Tự tăng tốc từng câu khi cần vừa khung thời gian, không đẩy câu sau lùi đi. Có tốc độ ban đầu/tối thiểu riêng.
4. **Tự động vietsub**: thêm video và SRT vào hàng đợi, chọn giọng/tốc độ, tự tạo giọng đọc, SRT mới và video căn theo mốc. Mỗi video có một thư mục kết quả riêng.
5. **Lịch sử & file**: tự đọc lại file đã tạo khi mở tool, tìm theo tên/đường dẫn, lọc loại file, xem thời gian và dung lượng. Nghe audio, lưu bản sao, mở thư mục chứa file hoặc xóa nhiều file. Nhật ký hoạt động nằm ở thẻ riêng trong cùng trang.

## Tự động vietsub video và hàng đợi

Tab **Tự động vietsub video** có ba phần trên cùng một màn hình: **1 · Thêm video**, **2 · Hàng đợi** và **3 · Kết quả**, kèm tiến trình. Bảng hàng đợi luôn hiển thị, không có khung phát video trong tab. Kết quả gồm video đã căn tốc độ không có âm thanh, SRT mới và giọng đọc WAV riêng. Mốc thời gian được tính trong bộ nhớ; các file tạm được dọn khi hoàn tất hoặc dừng/lỗi.

1. Chọn **Video**. Tool tự điền SRT cùng tên cạnh video nếu có; nếu SRT mang tên khác, dùng nút **Chọn…** ở dòng **SRT**.
2. Chọn **Giọng đọc** và **Tốc độ** từ 0.5–2.0×. Danh sách giọng có sẵn trước khi tải model; model tự tải theo Cấu hình nếu cần và được dùng lại cho toàn bộ hàng đợi.
3. Với một video, bấm **Chạy 1 video**. Với nhiều video, bấm **Thêm vào hàng đợi** để thêm video/SRT đang chọn, hoặc **Thêm nhiều video…** để chọn nhiều video. Cách thêm nhiều tự tìm SRT cùng tên cạnh mỗi video, ví dụ `clip.mp4` và `clip.srt`. Mỗi mục nhớ giọng/tốc độ tại lúc thêm; thay đổi lựa chọn sau đó chỉ áp dụng cho mục thêm tiếp theo. Sau khi thêm thành công, ô video/SRT được xóa, giọng trở về mặc định và tốc độ về 1.0×; các mục đã thêm vẫn giữ thiết lập đã chọn. Khi thêm nhiều, phần nhập chỉ reset sau khi thêm xong cả nhóm.
4. Bấm **Chạy N video** để chạy các mục đang chờ theo thứ tự. Một mục lỗi không chặn các mục sau. **Dừng** hủy mục đang chạy và giữ các mục chưa chạy; nút **Tiếp tục (N)** chạy tiếp các mục chờ/lỗi/đã dừng, không chạy lại mục đã xong.
5. Chọn một mục trong bảng để xem nguồn, giọng/tốc độ, kết quả hoặc lỗi. Với mục đã xong, các nút **Video MP4 / Phụ đề SRT / Giọng WAV / Mở thư mục** ở thanh Kết quả mở bằng ứng dụng mặc định trên máy. **Bỏ mục chọn / Bỏ mục đã xong** chỉ xóa dòng khỏi hàng đợi, giữ các file đã xuất.

Thanh tiến trình ghi số thứ tự video/tổng số video và bước hiện tại: **1/5 Kiểm tra đầu vào → 2/5 Chuẩn bị giọng → 3/5 Tạo giọng và SRT → 4/5 Căn video → 5/5 Lưu kết quả**. Tải model/suy luận chưa có phần trăm nội bộ dùng thanh chuyển động; bước đọc cập nhật theo số mục đã xong, bước video cập nhật theo FFmpeg.

Video và WAV được mở/nghe riêng bằng các nút kết quả. Chọn file hoặc chọn dòng trong hàng đợi không khởi động bộ giải mã/phát video; tab chỉ xử lý file khi bắt đầu tác vụ.

Mỗi mục SRT gốc tương ứng đúng một mục SRT mới, kể cả khi thiếu dấu câu hoặc chứa nhiều câu. SRT mới lấy timeline từ audio đã tạo; nội dung mỗi dòng tối đa 70 ký tự, ưu tiên ngắt giữa từ. Một từ đơn dài hơn giới hạn được ngắt để vừa dòng. Hai danh sách mốc dùng để căn video là dữ liệu nội bộ. SRT đầu vào giữ nguyên ngôn ngữ; quy trình này dùng lời thoại đã có trong SRT.

Nếu SRT gốc bắt đầu sau 0 mà audio mới bắt đầu ở 0, tool thêm khoảng lặng mở đầu bằng mốc đầu gốc và dịch SRT tương ứng để hai mốc căn được mà giữ phần mở đầu video. Nếu mốc gốc ở 0 nhưng model có khoảng lặng đầu, mục SRT đầu bao gồm khoảng lặng đó. Phần sau mốc cuối giữ tốc độ **1×**.

Video được căn bằng luồng **Giữ nguyên hình**: giữ dữ liệu các khung hình và đổi timestamp, không mã hóa hình hoặc âm thanh. Giọng đọc được xuất WAV PCM 16-bit, 48 kHz theo logic đọc văn bản. SRT xuất riêng, không đốt chữ lên hình. Video và WAV có thể dài khác nhau vì phần đuôi video giữ 1×; WAV luôn giữ toàn bộ lời đọc. Video/SRT gốc không bị ghi đè. Video cần có codec tương thích với MP4 để giữ nguyên hình.

Mỗi video được lưu trong một thư mục có tên video gốc, ví dụ `Nơi lưu/clip/`, gồm ba file cùng tên `video_<mã>.mp4`, `.srt` và `.wav`. Nếu thư mục đã tồn tại, tạo `clip_2`, `clip_3` để giữ kết quả cũ. Bộ ba được công bố cùng lúc khi hoàn tất; dừng/lỗi dọn thư mục tạm của mục đang chạy, giữ các mục đã xong. Lịch sử đọc cả các thư mục con này khi mở lại tool; xóa MP4 kèm SRT qua lịch sử vẫn giữ file WAV.

SRT cần UTF-8 hoặc UTF-16 có BOM; các mốc bắt đầu phải tăng dần, không trùng và nằm trong thời lượng video. File nguồn được kiểm tra trước khi tải model. Hàng đợi giữ khóa xuyên suốt lượt chạy và dùng lại model, không tạo nhiều bản model tranh GPU. Tool đọc SRT/metadata một lần mỗi video, xử lý mốc trong bộ nhớ, sinh giọng theo batch như đọc văn bản và lưu kết quả bằng đổi tên thư mục trên cùng ổ đĩa, tránh chép/mã hóa video thêm một lần.


## Nơi lưu và lịch sử file

Bấm **Đổi thư mục…** trên thanh **Nơi lưu**, hoặc vào **Cấu hình → Lưu trữ kết quả**, chọn/nhập thư mục rồi bấm **Áp dụng nơi lưu**. Tool kiểm tra quyền ghi và nhớ lựa chọn cho lần mở sau. Có thể đổi nơi lưu khi model đã tải; file tạo tiếp theo sử dụng thư mục mới. Đổi nơi lưu không di chuyển các file đã có, thư viện giọng clone vẫn ở vị trí ban đầu.

Trang **Lịch sử & file** hiển thị các kết quả trong thư mục hiện tại và các thư mục từng sử dụng, mới nhất ở trên. Bao gồm WAV/FLAC/MP3, MP4 đã xuất, SRT từng câu, ZIP batch, thư viện giọng JSON đã xuất, NPZ tham chiếu và bản ghi micro. Tool đọc thông tin file để khôi phục lịch sử khi khởi động; file đã bị xóa hoặc thư mục không còn truy cập được sẽ không hiện. Nút **Làm mới** cập nhật thay đổi bên ngoài. File đầu vào và `user_voices.json` không nằm trong danh sách xóa.

Nhập từ khóa vào **Tìm file**, chọn loại để lọc. Chọn một file để xem đường dẫn, nghe audio, lưu bản sao hoặc mở thư mục chứa file. Dùng **Ctrl / Shift** để chọn nhiều file, hoặc bấm **Chọn tất cả** để chọn các file đang hiển thị. **Xóa file đã chọn** hiển thị danh sách xác nhận rồi xóa vĩnh viễn khỏi máy. Mặc định **Xóa kèm SRT cùng tên** xóa phụ đề đi kèm khi chọn audio; bỏ dấu chọn nếu muốn giữ SRT. Đang tạo audio, phát audio hoặc ghi micro thì cần dừng/chờ hoàn tất trước khi xóa hay đổi nơi lưu.

Cài đặt nơi lưu nằm ở `outputs/v3turbo_desktop_settings.json`, gồm thư mục hiện tại, các thư mục từng dùng và vị trí thư viện giọng. `--output-dir PATH` ưu tiên thư mục chỉ định và lưu lại lựa chọn khi chạy bình thường; `--check` không ghi cài đặt.

## Đọc văn bản và tạo phụ đề từng câu

Trong tab **Văn bản**, nhập nội dung rồi bấm **Tạo audio + SRT từng câu**. Tool chia câu theo dấu `. ! ? …` và xuống dòng, giữ nguyên câu có ngoặc/trích dẫn, không cắt dấu chấm trong số thập phân. Mỗi câu được tổng hợp thành đoạn audio riêng; câu dài vẫn được SDK chia đoạn bên trong. Sau khi chỉnh tốc độ, tool ghép các câu, đo độ dài và khoảng nghỉ bằng số mẫu của chính audio đó để tạo phụ đề.

Hai file cùng tên được lưu trong thư mục kết quả, ví dụ `speech_subtitled_abc.wav` và `speech_subtitled_abc.srt`. Trang **Lịch sử & file** liệt kê cả hai; chọn một file trong cặp rồi bấm **Lưu cặp audio + SRT…** để chép sang thư mục khác.

Mỗi mục SRT chứa nguyên văn một câu, với thời gian bắt đầu/kết thúc theo đoạn audio của câu đó. Khoảng lặng đầu/cuối được loại khỏi thời gian hiển thị khi có thể đo được, nhưng audio giữ nguyên. Mốc đã tính theo tốc độ đang chọn; không cần chạy nhận dạng giọng nói. Nên xuất WAV/FLAC để tránh khác biệt độ trễ MP3 giữa các trình phát. Chế độ này đọc từng câu nên nhịp ngắt có thể khác khi đọc liền cả văn bản. Đây là phụ đề theo câu, không phải karaoke từng từ hoặc công cụ căn phụ đề cho audio bất kỳ.

## Đọc SRT và tự tăng tốc để vừa thời gian

Trong tab **SRT**, chọn giọng đọc và tốc độ ban đầu, bấm **Thêm nhiều file SRT**, rồi **Chạy hàng đợi**. Tab này luôn giữ đúng mốc bắt đầu và mọi khoảng lặng; không có tùy chọn chuyển sang nối câu liên tiếp. Nếu app đang mở khi cập nhật code, cần đóng hẳn và mở lại bằng `run_v3turbo_desktop.bat`.

Mỗi lượt trả audio và SRT mới cùng tên, ví dụ `srt_<mã>.wav` và `srt_<mã>.srt`, trong một thư mục mang tên SRT gốc: `episode.srt` → `Nơi lưu/episode/`. Nếu thư mục đã tồn tại, tạo `episode_2`, `episode_3` để giữ các bản cũ. Chỉ công bố thư mục khi cả hai file đã hoàn tất; dừng/lỗi dọn dữ liệu tạm. SRT mới giữ số thứ tự, timeline và nội dung gốc, chỉ sắp xếp lại phần lời thoại: mỗi dòng tối đa **70 ký tự**, ưu tiên ngắt giữa từ; câu dài hơn tiếp tục xuống các dòng kế tiếp trong cùng mục phụ đề. File SRT đầu vào không bị ghi đè. Trong **Lịch sử tạo** của tab SRT, chọn audio rồi bấm **Mở SRT**, **Lưu audio + SRT…** hoặc **Thư mục**. Hai file cũng có trong lịch sử chung kể cả khi mở lại tool; xóa audio kèm phụ đề sẽ xóa cả SRT mới.

Đặt **Tốc độ ban đầu / tối thiểu (x)** ngay trong tab SRT, từ `0.5` đến `2.0`, mặc định `1.0`. Đây là tốc độ riêng cho đọc SRT, không lấy hệ số từ **Cấu hình → Tốc độ đọc**. Ví dụ `0.8` cho nhịp đọc chậm hơn hoặc `1.2` cho nhịp đọc nhanh hơn. Tool không tự giảm tốc xuống dưới giá trị này để kéo dài câu ngắn.

Tool sinh audio cho từng mục SRT, loại khoảng lặng đầu/cuối model sinh thêm rồi đo độ dài phần giọng đọc bằng số mẫu. Tốc độ cuối cùng là `max(tốc độ ban đầu, thời lượng giọng gốc / thời lượng cho phép)`, áp dụng trong một lượt xử lý. Ví dụ câu từ `00:00:05,000` đến `00:00:07,000` cho phép 2 giây, giọng gốc dài 3 giây và tốc độ ban đầu `1.2x`: tool dùng `1.5x` để vừa khung, đặt phần giọng đọc ở giây thứ 5. Câu ngắn hơn khung giữ tốc độ ban đầu; phần còn lại và khoảng trống giữa các câu là khoảng lặng.

Mốc bắt đầu được đặt theo vị trí mẫu của SRT, giữ cả khoảng lặng trước câu đầu; không ghép bằng cách nối các câu liên tiếp. Việc bỏ khoảng lặng dựa vào tín hiệu waveform với ngưỡng nhỏ để giữ phụ âm nhẹ, không phải nhận dạng từ/âm tiết. Log ghi mốc bắt đầu, kết thúc audio, giới hạn khung và tốc độ của từng câu để kiểm tra.

Ví dụ câu đầu bắt đầu ở giây thứ 5, kết thúc ở giây thứ 7, câu sau bắt đầu ở giây thứ 90: file giữ 5 giây im lặng đầu và toàn bộ khoảng từ giây 7 đến giây 90. Khi lời đọc kết thúc trước giây thứ 7, phần còn lại cũng là im lặng. Bộ đọc nhận diện các mục bằng dòng thời gian, nên file thiếu dòng trắng giữa các mục vẫn giữ từng câu và mốc riêng. Log đầu tác vụ báo số câu và tổng thời lượng SRT để đối chiếu với file audio.

SRT dùng **Rubber Band** để chỉnh tốc độ và ép câu vừa khung. Tool dùng chế độ chất lượng cao, pitch shift bằng 0, giữ tính liên tục pha và bật preserve_formants theo [tài liệu Pedalboard](https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.time_stretch). Câu không cần đổi tốc độ được bỏ qua bước time-stretch; tool làm mềm 5 ms cuối phần có tiếng để giảm tiếng tách khi chuyển sang khoảng lặng, không thay mốc bắt đầu hoặc số mẫu. Mỗi câu được ghi trực tiếp vào waveform của timeline sau khi xử lý, tránh giữ thêm danh sách toàn bộ audio câu trong bộ nhớ.

Tự căn SRT có thể cần hệ số vượt 2x, độc lập với giới hạn tốc độ nhập tay; ép quá nhanh có thể làm mất độ tự nhiên và giảm độ rõ lời dù dùng bộ xử lý nào. Kết quả báo số câu đã tăng tốc so với tốc độ ban đầu và tốc độ lớn nhất. Audio WAV/FLAC được đặt theo số mẫu, với tổng độ dài đến mốc kết thúc muộn nhất trong file SRT. MP3 có thể có độ trễ mã hóa tùy trình phát; ưu tiên WAV khi ghép với video. Cần sinh lại audio từ SRT để dùng bộ mới; không thể phục hồi chắc chắn lỗi đã có trong audio cũ bằng cách đổi bộ tăng tốc.

Nếu hai câu chồng thời gian, khung câu trước kết thúc ở mốc bắt đầu câu kế tiếp để không chồng tiếng. Hai câu có cùng mốc bắt đầu, hoặc mốc kết thúc không sau mốc bắt đầu, sẽ báo lỗi trước khi sinh audio; cần sửa SRT. Tool không tự sửa hoặc ghi đè file phụ đề đầu vào.

Watermark được gắn sau khi chỉnh tốc độ và căn thời gian. Nếu watermark làm thay đổi số mẫu, tool báo lỗi thay vì làm lệch mốc. Nút đọc SRT của desktop luôn truyền chế độ giữ mốc và tự căn thời gian, nên không thể vô tình bỏ mất các khoảng nghỉ bằng lựa chọn đọc nối tiếp.

## Fine-tune một giọng

Chuẩn bị `metadata.csv` chứa `file_name|text`, và thư mục `raw_audio/` bên cạnh. Trong tab Fine-tune:

1. Chọn **Chuẩn bị dữ liệu**, chọn thư mục dataset rồi chạy. Kết quả mặc định `dataset/train.parquet`.
2. Chọn **Train LoRA**, chọn parquet, thư mục xuất, tên lượt train, epoch và rank. Tool bật merge khi train xong; ví dụ xuất `finetune/output/my_voice/merged`.
3. **Merge LoRA** dùng khi đã có adapter và muốn merge riêng.
4. Chọn **Đóng gói giọng**, audio của đúng người đã train, tên giọng và thư mục model merge. Codec/speaker encoder cần dùng model gốc ở cấu hình.
5. Sang Cấu hình, nhập thư mục `merged`, chọn `pytorch` / `cuda`, giữ subfolder `update`, tải model rồi dùng giọng đã đóng gói.

Tool giải phóng model desktop trước khi chạy các script này. LoRA cần CUDA. Model merge dùng PyTorch; bản ONNX mặc định chứa trọng số gốc. Các tham số train nâng cao khác dùng script gốc trong `finetune/`.

## Chỉnh tốc độ, streaming và dừng

Trong **Cấu hình → Tốc độ đọc**, nhập hệ số từ `0.5` đến `2.0`: `0.8` chậm hơn, `1.0` tốc độ gốc, `1.2` nhanh hơn.

Tool tự dùng **Rubber Band** cho đọc văn bản, tạo audio + SRT từng câu, hàng loạt, hội thoại và streaming. Không còn lựa chọn bộ xử lý khác trong tab Cấu hình hoặc SRT. Tool dùng chế độ chất lượng cao, không dịch cao độ, giữ tính liên tục pha và bật preserve_formants.

Không cần tải lại model. Tốc độ tác động cả file xuất và âm thanh nghe; khoảng nghỉ giữa các lượt hội thoại vẫn theo cấu hình. Tab SRT có tốc độ ban đầu riêng, luôn giữ mốc phụ đề. Ở `1.0x`, các chức năng đọc thông thường không xử lý đổi tốc độ. Watermark được gắn sau bước chỉnh tốc độ. Nên bắt đầu ở `0.85–1.15x`, nghe so sánh WAV ở cùng âm lượng. Đây là xử lý sau khi model sinh giọng: thay đổi lớn vẫn có thể tạo âm lặp hoặc làm phụ âm kém tự nhiên. Muốn sát giọng gốc nhất, dùng `1.0x` hoặc audio tham chiếu có nhịp đọc phù hợp. Không có bảo đảm rằng chỉnh tốc độ vẫn giữ nguyên toàn bộ chất giọng.

Streaming ở `1.0x` vẫn phát ngay từng phần. Ở tốc độ khác `1.0x`, tool sinh xong toàn bộ audio, điều chỉnh tốc độ một lần rồi phát/lưu WAV; thời gian chờ nghe dài hơn. Audio nguồn được giữ dưới dạng float trong bộ đệm, không ghi rồi đọc lại WAV trung gian. Bộ đệm tối đa 8 MiB trước khi tự chuyển sang file tạm; đọc nguồn và chạy Rubber Band vẫn cần RAM tương ứng với toàn bộ audio. WAV cuối chỉ được ghi một lần. Lần chỉnh tốc độ đầu tiên có thể chậm hơn do khởi tạo bộ xử lý audio. Rubber Band kiểm tra dừng trước và sau khi xử lý đoạn hiện tại.

Streaming ghi từng phần vào WAV và phát liên tục bằng hàng đợi waveOut trên Windows. Nút **Dừng tác vụ / phát** dừng streaming/phát và kết thúc tiến trình fine-tune đang chạy. Lượt `infer`/batch thông thường chỉ dừng được sau lượt suy luận hiện tại. Dừng streaming xóa file còn dang dở. Những file batch đã hoàn thành được giữ lại.

Khi API đang chạy, dừng API trước khi chạy tác vụ desktop để tránh tranh chấp GPU. API chỉ được xem là sẵn sàng khi log báo ready. Khi đóng app, các tiến trình do tool khởi chạy được kết thúc.

## Dữ liệu và giới hạn

Khi chưa đặt nơi lưu, mặc định dùng `outputs/v3turbo_desktop/`; giọng riêng ở `user_voices.json`, không sửa giọng trong package. Đổi thư mục ngay trên giao diện hoặc dùng `--output-dir PATH`. Các file tồn tại sau khi đóng app và có thể quản lý/xóa trong lịch sử. Thư viện giọng ở vị trí ban đầu được giữ lại khi thay đổi nơi lưu audio.

Đầu ra gốc 48 kHz. Tag thử nghiệm: `[cười]`, `[thở dài]`, `[hắng giọng]`. Bản thân v3 Turbo bỏ qua `speed`; tool desktop điều chỉnh tốc độ bằng xử lý audio. `style` và `instructions` vẫn không có nút điều khiển. API riêng trong tab Fine-tune/API vẫn giữ hành vi SDK và không áp dụng tốc độ của desktop. CPU int8 cần phần cứng tương thích. Bộ scheduler CUDA giữ giới hạn streaming/frame như SDK; desktop xử lý một tác vụ tại một thời điểm.

## Kiểm tra

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_v3turbo_desktop.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_speech_speed.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_speech_subtitles.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_srt_timing.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_desktop_ui.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_video_editor.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_video_copy_export.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_vietsub_prepare.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_video_fast_export.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_video_export_audio.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_video_preview.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_video_preview_audio.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_video_markers.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_video_marker_timeline.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_video_marker_workflow.py -v
.\.venv\Scripts\python.exe v3turbo_desktop.py --check
```

Test dùng model giả để kiểm tra thứ tự batch, lưu WAV, streaming, dừng/đóng generator, khóa model, truyền tham số, SRT, thư viện giọng và kiểm tra dữ liệu. Test Rubber Band kiểm tra thời lượng/cao độ ở nhiều sample rate, áp dụng cho các chức năng đọc, streaming giữ nguồn float, watermark sau đổi tốc độ và báo lỗi thiếu dependency trước suy luận. Test streaming kiểm tra không ghi WAV trung gian, model tái sử dụng mảng audio, tràn bộ đệm ra file tạm và dọn sạch khi dừng. Test hội thoại kiểm tra nhóm giọng, giới hạn batch, thứ tự và khoảng nghỉ. Test phụ đề kiểm tra chia câu, khoảng nghỉ, mốc thời gian sau chỉnh tốc độ và việc xóa cặp file chưa hoàn tất khi gặp lỗi. Những kiểm tra này không thay thế việc nghe so sánh để đánh giá độ tự nhiên của giọng. `--check` dựng giao diện ẩn, không tải model và không ghi micro/phát audio.

Test lưu trữ và giao diện kiểm tra nhớ nơi lưu sau khi mở lại, lịch sử nhiều thư mục, lọc/tìm kiếm, xóa nhiều file và audio + SRT, hủy thao tác xóa, bảo vệ thư viện giọng và chặn thao tác khi đang chạy. Bố cục được kiểm tra ở kích thước `980×620` và `1280×850` bằng cửa sổ thử nghiệm ẩn. Test video kiểm tra kéo chọn/di chuyển/thu ngắn đoạn, giữ chính xác các mốc liền kề, thay tốc độ qua slider và nhập số, kế hoạch xuất nhiều đoạn, đồng bộ âm thanh và hình. Test preview kiểm tra tua gộp, loại kết quả của file cũ, đóng reader và phát/tua một video mẫu thật trong khung Tkinter.

Benchmark đọc/ghi và điều phối streaming, dùng model giả và thay time-stretch bằng phép lấy mẫu đơn giản để tách chi phí này khỏi suy luận/DSP: median 5 lượt với nguồn 20/60/600 giây lần lượt giảm từ 33.87/105.42/784.89 ms xuống 19.82/61.73/411.20 ms trên máy phát triển. Đây là cải thiện khoảng 41–48% của phần đọc/ghi và điều phối, chưa phải mức tăng tốc tạo giọng của model. Script và số liệu nằm trong `outputs/audio_benchmark/` ở máy phát triển. Hội thoại nhiều câu liên tiếp cùng giọng có thể tận dụng batching CUDA; mức tăng tốc thực tế phụ thuộc phần cứng và nội dung, CPU vẫn có thể suy luận tuần tự.

## Hàm tạo giọng dùng chung

Phần tổng hợp trong `apps/v3turbo_tool.py` gọi `generate_speech()` từ `apps/simple_tts_engine.py` cho Tạo audio, Audio + SRT, batch, hội thoại và đọc SRT. Hàm nhận model đang tải, tham số sampling/watermark và giọng hoặc audio tham chiếu; trả audio trong RAM bằng `save=False`. Các nhóm câu vẫn dùng `infer_batch` để giữ hiệu suất batch của SDK.

Tool desktop tiếp tục phụ trách xuất WAV/FLAC/MP3, lịch sử file, dừng tác vụ, Rubber Band và căn mốc SRT. Không ghi WAV trung gian hoặc nạp thêm model. Streaming tiếp tục dùng `infer_stream` để phát các chunk ngay khi có dữ liệu. Các hiệu ứng tùy chọn trong simple_tts không tự bật cho desktop.

## Hàng đợi trong tab Văn bản và SRT

Tab **Văn bản** có tốc độ đọc ngay trong tab (dùng chung giá trị với Cấu hình), không còn audio tham chiếu trực tiếp. Muốn dùng giọng clone, tạo/lưu giọng trong **Clone / Giọng**, rồi chọn giọng đó.

- **Thêm nhiều TXT**: chọn nhiều file để thêm vào hàng đợi.
- **Thêm nội dung vào hàng đợi**: mỗi lần thêm giữ toàn bộ nội dung thành một mục.
- **Tách nhiều văn bản (===)**: phân cách các văn bản bằng một dòng chỉ có `===`; các đoạn trống trong cùng văn bản vẫn được giữ.
- Bấm **Chạy hàng đợi**: chốt giọng, tốc độ, định dạng và chạy từng mục một.

Tab **SRT** chọn nhiều file bằng **Thêm nhiều file SRT**. Mỗi file tạo một audio riêng, giữ mốc bắt đầu/khoảng lặng và tốc độ ban đầu đã chọn.

Mỗi mục có trạng thái và thanh tiến trình riêng. Audio văn bản dùng thanh chuyển động khi SDK suy luận vì chưa có phần trăm nội bộ; hoàn thành hiển thị 100%. SRT hiển thị tiến trình theo số câu hoàn thành. Mục lỗi không chặn các mục tiếp theo; chạy lại hàng đợi sẽ thử các mục lỗi/đã dừng và các mục còn chờ, bỏ qua mục đã xong. Có thể chọn và xóa mục khỏi hàng đợi khi không chạy.

**Dừng** yêu cầu dừng mục đang chạy và không bắt đầu mục tiếp theo; SDK có thể cần hoàn tất lượt suy luận hiện tại. Hàng đợi thuộc phiên hiện tại, không tự lưu khi đóng tool.

Hai tab có **Lịch sử tạo** riêng, tự cập nhật và nạp lại các file đã xuất khi mở tool. Có thể nghe/mở file hoặc thư mục ngay tại đây. Kết quả không tự chuyển sang tab Lịch sử chung. Thanh nơi lưu được ẩn trên hai tab; nơi lưu chung vẫn được chỉnh ở **Cấu hình**. Xóa file vẫn ở **Lịch sử & file**.

## Các xử lý âm thanh được giữ lại

Desktop và simple_tts dùng Rubber Band để đổi tốc độ. Audio giữ biên độ từ model; đã bỏ bộ lọc EQ, compressor, giảm vượt đỉnh và các ô lựa chọn tương ứng. Streaming ở 1.0x phát ngay từng phần; tốc độ khác 1.0x chờ tổng hợp xong rồi chỉnh tốc độ và phát bản được xuất.

Đã bỏ tính năng khử nhiễu, watermark và fade cuối câu trong tool. Clone và xuất reference luôn dùng audio mẫu không khử nhiễu; các tham số False được truyền rõ để SDK không tự bật xử lý mặc định. Dữ liệu embedding/reference codes là phần của mô hình để nhận diện giọng, không phải hiệu ứng hậu xử lý.

Tốc độ 1.0x bỏ qua Rubber Band. Căn mốc SRT, bỏ padding im lặng và chèn khoảng trống vẫn giữ để lời đọc khớp phụ đề. Tham số tổng hợp của model được giữ nguyên.

## Giao diện rút gọn và tạo theo nhóm

Tool hiện có đúng năm tab: **Văn bản**, **SRT**, **Video**, **Lịch sử & file**, **Cấu hình**. Đã bỏ tab Hàng loạt/ZIP, Hội thoại, Thư viện giọng và Fine-tune/API, cùng các hàm quản lý riêng của chúng. Các file đã tạo và giọng đã lưu trước đây không bị xóa; giọng đã lưu vẫn được nạp khi tải model.

Hàng đợi văn bản và SRT vẫn nằm ngay trong hai tab. Hàng đợi văn bản gom các mục hợp lệ theo **Số mục / batch** và gọi `infer_batch`, xuất từng file audio riêng, không tạo ZIP. Trên GPU có thể tận dụng xử lý batch; CPU/ONNX vẫn xử lý tuần tự trong engine. Một lỗi suy luận của nhóm làm nhóm đó báo lỗi; có thể thử lại. Các file TXT lỗi/trống được báo riêng trước khi tổng hợp và không chặn các mục hợp lệ.

Model được dùng lại khi bấm Tải model với cùng cấu hình; đổi cấu hình sẽ tải lại. Giải phóng model rồi tải lại nếu cần đọc lại trọng số/giọng đã thay đổi trên đĩa. Danh sách giọng chỉ được cập nhật khi model thay đổi. Các sự kiện kết quả hàng đợi được gộp để tránh quét lịch sử nhiều lần trong cùng lượt cập nhật giao diện.

Các tối ưu giữ chất lượng FP32, tham số sampling và Rubber Band chất lượng cao; không tuyên bố mức tăng tốc khi chưa đo trên model thật.

## Tối ưu Audio + SRT theo cách tạo từng câu

Vẫn tạo từng câu theo batch, không dùng Stable-ts hoặc mô hình căn chỉnh bổ sung. Rubber Band chỉ chạy một lần trên mỗi câu khi tốc độ khác 1.0x. Các câu được ghép trực tiếp để lấy mốc phụ đề và xuất audio.

Trên CUDA, thử Số mục / batch từ 8 lên 16 nếu đủ bộ nhớ; không bảo đảm luôn nhanh hơn, cần đo trên máy. Trên CPU/ONNX, tăng batch không làm các câu chạy song song. Tốc độ 1.0x bỏ qua bước chỉnh tốc độ. Chất lượng mô hình và chế độ Rubber Band vẫn được giữ.

## Mặc định Cấu hình

Ô Dùng mã audio tham chiếu đã bỏ khỏi giao diện; model vẫn dùng mã giọng theo mặc định. Cấu hình chỉ có tham số model, tốc độ và định dạng xuất.

## Tab Văn bản: hàng đợi audio + SRT

Tab Văn bản có hai cột: nội dung/giọng/tốc độ bên trái, **Hàng đợi** và **Lịch sử tạo** bên phải. Chỉ giữ **Thêm văn bản**, **Thêm TXT** và các nút quản lý hàng đợi. Mỗi lần thêm văn bản là một mục; nội dung không tự tách bằng dòng ===.

Bấm **Chạy hàng đợi** luôn xuất audio và SRT từng câu cho mỗi mục, kể cả file TXT. Các mục chạy lần lượt; câu trong mỗi mục vẫn được tổng hợp theo batch. Thanh tiến trình hiển thị số câu hoàn thành của mục đang chạy.

Trong **Lịch sử tạo**, chọn file rồi bấm **Phát** hoặc nhấp đúp để nghe ngay bằng trình phát nội bộ của tool; bấm **Dừng** để dừng. **Xóa** yêu cầu xác nhận và xóa audio cùng SRT đi kèm, cập nhật cả lịch sử trong tab và lịch sử chung. Không thể xóa khi đang tạo/phát. Nút **Thư mục** vẫn mở nơi chứa file.

Bố cục tương tự cho phần hàng đợi/lịch sử của tab SRT. Các nút nhập TXT/PDF đơn, tạo audio riêng, streaming và tách nhiều văn bản đã được bỏ khỏi tab Văn bản.

## Nghe thử SRT có mốc bắt đầu muộn

Audio SRT vẫn giữ timeline tuyệt đối, bao gồm khoảng lặng dài trước câu đầu. Nút Phát trong lịch sử SRT/lịch sử chung tự tìm phần có tín hiệu và nghe từ trước đó 100 ms; không sửa file, không dời mốc xuất, không bỏ khoảng lặng giữa các câu. Trạng thái hiển thị mốc bắt đầu phát. Nếu không có tín hiệu trên toàn file, trình phát báo lỗi; luồng tạo SRT cũng chặn xuất track toàn im lặng.

## Đo thời gian Audio + SRT

Log của mỗi mục văn bản ghi tổng thời gian, thời gian model và thời gian hậu xử lý/xuất file. So sánh sau khi model đã tải và chạy thử một lượt, vì lần đầu có chi phí khởi tạo.

Trên PyTorch/CUDA, tool gom cửa sổ tối đa 4 × batch (không quá 128 câu) rồi để SDK sắp theo độ dài và xử lý từng batch. Giới hạn batch GPU vẫn đúng giá trị Cấu hình; không tăng số câu đồng thời trên GPU. CPU/ONNX giữ nhóm cũ và chạy tuần tự. Cách này giảm số lần gọi và cho SDK nhiều câu để ghép theo độ dài, nhưng chưa có số đo tăng tốc trên model thật.

## Tối ưu đọc file SRT

PyTorch/CUDA dùng cửa sổ câu lớn hơn để SDK sắp theo độ dài trước khi batch; giới hạn batch GPU vẫn giữ như cấu hình. ONNX/CPU giữ nhóm cũ. Rubber Band chỉ chạy khi cần chỉnh thời lượng và luôn giữ chất lượng cao; các khoảng lặng được chèn theo timeline SRT.

Các câu trong mỗi nhóm được căn thời lượng bằng Rubber Band song song tối đa 4 luồng CPU. Suy luận model, thông báo tiến trình và ghép timeline vẫn chạy tuần tự; thứ tự câu, khoảng lặng, tốc độ tối thiểu và mốc bắt đầu được giữ nguyên. Batch một câu và nhóm câu vốn vừa khung ở 1.0x không tạo pool. Khi dừng hoặc lỗi, tool chờ các luồng native đang chạy kết thúc trước khi giải phóng dữ liệu hoặc nhận tác vụ mới.

Đo trên RTX 5050 / PyTorch CUDA FP32, batch 8, ngày 10/10/2026: SRT mẫu 24 câu cần tăng tốc để vừa khung có tổng thời gian trung bình hai lượt sau khởi tạo giảm từ 4,72 xuống 2,55 giây (khoảng 1,85 lần). Riêng căn thời lượng giảm từ 2,88 xuống 0,76 giây. Chạy lại bằng cùng audio nguồn cho file WAV giống bản xử lý tuần tự từng byte. Kết quả phụ thuộc nội dung SRT; nếu đa số câu đã vừa khung, phần chỉnh thời lượng ít nên mức cải thiện tổng thể sẽ nhỏ hơn.

Log mỗi file SRT ghi Total, model, tempo và export để biết phần nào chậm. Tốc độ ban đầu SRT 1.0x sẽ bỏ qua Rubber Band với những câu vốn vừa khung; câu dài vẫn được tự tăng tốc.

## Thư viện mở video

File `run_v3turbo_desktop.bat` kiểm tra MoviePy 2.2+, Pillow và FFmpeg qua imageio-ffmpeg trước khi mở ứng dụng. Nếu thiếu, launcher dùng uv pip để cài bổ sung MoviePy vào chính .venv hiện có, không uv sync toàn bộ môi trường và không gỡ cấu hình CUDA. Lần cài bổ sung cần Internet. Nếu không cài được, tool vẫn khởi chạy để dùng các chức năng khác; lỗi thư viện video được báo khi mở video.

Có thể cài thủ công:
```powershell
uv pip install --python .venv/Scripts/python.exe "moviepy>=2.2,<3"
```

### Tốc độ xuất video

Trên trích đoạn 12 giây, 960×720, CPU libx264, preset fast và chất lượng 20: đường MoviePy cũ 2,55 giây, FFmpeg trực tiếp 1,73 giây. Đây là phép đo mẫu; không bảo đảm cùng mức tăng cho video khác hay toàn bộ file. Chọn preset **veryfast** để giảm chi phí mã hóa CPU; file có thể lớn hơn. Video vẫn phải mã hóa lại khi đổi tốc độ.
