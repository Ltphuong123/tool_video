# VieNeu v3 Turbo Studio — ứng dụng desktop Python

Chạy trên Windows bằng Tkinter, không cần trình duyệt, dùng Python 3.10–3.13. Model chỉ tải khi bấm **Tải model**.

## Khởi chạy

Trong thư mục dự án:

```powershell
uv sync --extra srt-quality
uv run --extra srt-quality python v3turbo_desktop.py
```

Cũng có thể bấm đúp `run_v3turbo_desktop.bat`. Nếu `.venv` đã có, launcher dùng Python trong môi trường đó. Sau khi cài package có thể chạy `vieneu-desktop`.

Để chạy CUDA:

```powershell
uv sync --extra cuda --extra srt-quality
uv run --extra cuda --extra srt-quality python v3turbo_desktop.py
```

Các dependency tùy chọn:

```powershell
uv sync --extra cuda --extra srt-quality --extra pdf --extra watermark --extra finetune
```

`pdf` thêm PyMuPDF; `watermark` thêm Resemble Perth; `finetune` thêm thư viện train LoRA. CPU chỉ cần `uv sync`. Trên Linux cần cài Tkinter của hệ điều hành và `sounddevice` để phát audio. Bản ghi micro tích hợp dùng Windows MCI.

`srt-quality` thêm Pedalboard có Rubber Band tích hợp để xử lý tốc độ cho cả SRT và các chức năng đọc khác. Tool dùng duy nhất Rubber Band; khi cần đổi tốc độ hoặc đọc SRT mà thiếu thư viện sẽ báo lỗi trước khi sinh audio. Đọc văn bản ở tốc độ gốc `1.0x` không cần chạy bộ chỉnh tốc độ. Nếu cập nhật môi trường đang dùng CUDA, giữ `--extra cuda` trong lệnh sync.

## Giao diện và điều hướng

Thanh bên trái sắp xếp theo quy trình: **Đọc văn bản → Đọc phụ đề SRT → Tạo hàng loạt → Hội thoại → Thư viện giọng → Lịch sử & file → Cấu hình → Fine-tune & API**. Tool mở ở trang Văn bản; thanh trên luôn hiển thị nơi lưu và nút đổi thư mục. Các trang cấu hình dài có thanh cuộn; bố cục cấu hình tự chuyển về một cột khi cửa sổ hẹp. Nút tạo audio được làm nổi bật, trạng thái model và tiến trình tách khỏi khu vực nhập nội dung.

1. **Cấu hình**: model/repo hoặc thư mục merge, auto/CPU/CUDA, ONNX/PyTorch, fp32/int8, dtype GPU, số luồng CPU, giới hạn streaming, thử lại câu ngắn. Chỉnh tốc độ đọc, temperature, top_k, top_p, phạt lặp/cửa sổ lặp, frame/đoạn, ký tự/đoạn và batch. Bật khử nhiễu, mã tham chiếu, watermark. Xuất WAV/FLAC/MP3; MP3 cần libsndfile có bộ mã hóa phù hợp.
2. **Văn bản**: nhập văn bản hoặc TXT/Markdown/PDF, chọn giọng, clone trực tiếp bằng audio mẫu, tạo audio, audio + SRT từng câu hoặc streaming và nghe. Audio mẫu được ưu tiên hơn giọng chọn. PDF scan cần OCR trước.
3. **Clone / Giọng**: chọn audio hoặc ghi micro, clone rồi lưu giọng, xóa giọng riêng, khử nhiễu, xuất embedding và codes thành NPZ, nhập/xuất thư viện JSON. Dùng clip sạch khoảng 3–8 giây, không cần lời thoại mẫu. Giọng có sẵn và bí danh được bảo vệ.
4. **Hàng loạt**: mỗi dòng là một văn bản hoặc chọn nhiều file. Tạo ZIP gồm audio đánh số theo đúng thứ tự và manifest JSON. Kết quả lẻ cũng giữ trong thư mục xuất. Giọng chung áp dụng cho toàn bộ batch.
5. **Hội thoại**: mỗi dòng `Nhân vật: lời thoại`, gán giọng cho từng nhân vật, đặt khoảng nghỉ giữa các lượt. Chọn giọng trước khi bấm tạo. Các lượt liên tiếp cùng giọng được tạo theo batch, tối đa **Số mục / batch**, để tận dụng GPU; thứ tự lượt và khoảng nghỉ được giữ nguyên.
6. **SRT**: luôn đọc theo mốc bắt đầu tuyệt đối của phụ đề và giữ các khoảng trống. Tự tăng tốc từng câu khi cần vừa khung thời gian, không đẩy câu sau lùi đi. Có tốc độ ban đầu/tối thiểu riêng.
7. **Fine-tune / API**: chạy lần lượt prepare/train/merge/đóng gói giọng bằng script sẵn có; hoặc chạy API streaming model gốc với CPU/CUDA, cổng và API key. API chỉ bind `127.0.0.1`. Giọng của desktop và giọng đăng ký trong API là hai thư viện riêng.
8. **Lịch sử & file**: tự đọc lại file đã tạo khi mở tool, tìm theo tên/đường dẫn, lọc loại file, xem thời gian và dung lượng. Nghe audio, lưu bản sao, mở thư mục chứa file hoặc xóa nhiều file. Nhật ký hoạt động nằm ở thẻ riêng trong cùng trang.

## Nơi lưu và lịch sử file

Bấm **Đổi thư mục…** trên thanh **Nơi lưu**, hoặc vào **Cấu hình → Lưu trữ kết quả**, chọn/nhập thư mục rồi bấm **Áp dụng nơi lưu**. Tool kiểm tra quyền ghi và nhớ lựa chọn cho lần mở sau. Có thể đổi nơi lưu khi model đã tải; file tạo tiếp theo sử dụng thư mục mới. Đổi nơi lưu không di chuyển các file đã có, thư viện giọng clone vẫn ở vị trí ban đầu.

Trang **Lịch sử & file** hiển thị các kết quả trong thư mục hiện tại và các thư mục từng sử dụng, mới nhất ở trên. Bao gồm WAV/FLAC/MP3, SRT từng câu, ZIP batch, thư viện giọng JSON đã xuất, NPZ tham chiếu và bản ghi micro. Tool đọc thông tin file để khôi phục lịch sử khi khởi động; file đã bị xóa hoặc thư mục không còn truy cập được sẽ không hiện. Nút **Làm mới** cập nhật thay đổi bên ngoài. File đầu vào và `user_voices.json` không nằm trong danh sách xóa.

Nhập từ khóa vào **Tìm file**, chọn loại để lọc. Chọn một file để xem đường dẫn, nghe audio, lưu bản sao hoặc mở thư mục chứa file. Dùng **Ctrl / Shift** để chọn nhiều file, hoặc bấm **Chọn tất cả** để chọn các file đang hiển thị. **Xóa file đã chọn** hiển thị danh sách xác nhận rồi xóa vĩnh viễn khỏi máy. Mặc định **Xóa kèm SRT cùng tên** xóa phụ đề đi kèm khi chọn audio; bỏ dấu chọn nếu muốn giữ SRT. Đang tạo audio, phát audio hoặc ghi micro thì cần dừng/chờ hoàn tất trước khi xóa hay đổi nơi lưu.

Cài đặt nơi lưu nằm ở `outputs/v3turbo_desktop_settings.json`, gồm thư mục hiện tại, các thư mục từng dùng và vị trí thư viện giọng. `--output-dir PATH` ưu tiên thư mục chỉ định và lưu lại lựa chọn khi chạy bình thường; `--check` không ghi cài đặt.

## Đọc văn bản và tạo phụ đề từng câu

Trong tab **Văn bản**, nhập nội dung rồi bấm **Tạo audio + SRT từng câu**. Tool chia câu theo dấu `. ! ? …` và xuống dòng, giữ nguyên câu có ngoặc/trích dẫn, không cắt dấu chấm trong số thập phân. Mỗi câu được tổng hợp thành đoạn audio riêng; câu dài vẫn được SDK chia đoạn bên trong. Sau khi chỉnh tốc độ, tool ghép các câu, đo độ dài và khoảng nghỉ bằng số mẫu của chính audio đó để tạo phụ đề.

Hai file cùng tên được lưu trong thư mục kết quả, ví dụ `speech_subtitled_abc.wav` và `speech_subtitled_abc.srt`. Trang **Lịch sử & file** liệt kê cả hai; chọn một file trong cặp rồi bấm **Lưu cặp audio + SRT…** để chép sang thư mục khác.

Mỗi mục SRT chứa nguyên văn một câu, với thời gian bắt đầu/kết thúc theo đoạn audio của câu đó. Khoảng lặng đầu/cuối được loại khỏi thời gian hiển thị khi có thể đo được, nhưng audio giữ nguyên. Mốc đã tính theo tốc độ đang chọn; không cần chạy nhận dạng giọng nói. Nên xuất WAV/FLAC để tránh khác biệt độ trễ MP3 giữa các trình phát. Chế độ này đọc từng câu nên nhịp ngắt có thể khác khi đọc liền cả văn bản. Đây là phụ đề theo câu, không phải karaoke từng từ hoặc công cụ căn phụ đề cho audio bất kỳ.

## Đọc SRT và tự tăng tốc để vừa thời gian

Trong tab **SRT**, chọn file `.srt`, giọng đọc và tốc độ ban đầu, rồi bấm **Tạo audio từ SRT**. Tab này luôn giữ đúng mốc bắt đầu và mọi khoảng lặng; không có tùy chọn chuyển sang nối câu liên tiếp. Nếu app đang mở khi cập nhật code, cần đóng hẳn và mở lại bằng `run_v3turbo_desktop.bat`. Giao diện mới hiển thị dòng **Chế độ SRT: luôn giữ đúng mốc bắt đầu và mọi khoảng lặng.**

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
.\.venv\Scripts\python.exe v3turbo_desktop.py --check
```

Test dùng model giả để kiểm tra thứ tự batch, lưu WAV, streaming, dừng/đóng generator, khóa model, truyền tham số, SRT, thư viện giọng và kiểm tra dữ liệu. Test Rubber Band kiểm tra thời lượng/cao độ ở nhiều sample rate, áp dụng cho các chức năng đọc, streaming giữ nguồn float, watermark sau đổi tốc độ và báo lỗi thiếu dependency trước suy luận. Test streaming kiểm tra không ghi WAV trung gian, model tái sử dụng mảng audio, tràn bộ đệm ra file tạm và dọn sạch khi dừng. Test hội thoại kiểm tra nhóm giọng, giới hạn batch, thứ tự và khoảng nghỉ. Test phụ đề kiểm tra chia câu, khoảng nghỉ, mốc thời gian sau chỉnh tốc độ và việc xóa cặp file chưa hoàn tất khi gặp lỗi. Những kiểm tra này không thay thế việc nghe so sánh để đánh giá độ tự nhiên của giọng. `--check` dựng giao diện ẩn, không tải model và không ghi micro/phát audio.

Test lưu trữ và giao diện kiểm tra nhớ nơi lưu sau khi mở lại, lịch sử nhiều thư mục, lọc/tìm kiếm, xóa nhiều file và audio + SRT, hủy thao tác xóa, bảo vệ thư viện giọng và chặn thao tác khi đang chạy. Bố cục được kiểm tra ở kích thước `980×620` và `1280×850` bằng cửa sổ thử nghiệm ẩn.

Benchmark đọc/ghi và điều phối streaming, dùng model giả và thay time-stretch bằng phép lấy mẫu đơn giản để tách chi phí này khỏi suy luận/DSP: median 5 lượt với nguồn 20/60/600 giây lần lượt giảm từ 33.87/105.42/784.89 ms xuống 19.82/61.73/411.20 ms trên máy phát triển. Đây là cải thiện khoảng 41–48% của phần đọc/ghi và điều phối, chưa phải mức tăng tốc tạo giọng của model. Script và số liệu nằm trong `outputs/audio_benchmark/` ở máy phát triển. Hội thoại nhiều câu liên tiếp cùng giọng có thể tận dụng batching CUDA; mức tăng tốc thực tế phụ thuộc phần cứng và nội dung, CPU vẫn có thể suy luận tuần tự.
