# VieNeu v3 Turbo Studio — ứng dụng desktop Python

Chạy trên Windows bằng Tkinter, không cần trình duyệt, dùng Python 3.10–3.13. Model chỉ tải khi bấm **Tải model**.

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

Các dependency tùy chọn:

```powershell
uv sync --extra cuda --extra srt-quality --extra video --extra pdf --extra watermark --extra finetune
```

`pdf` thêm PyMuPDF; `watermark` thêm Resemble Perth; `finetune` thêm thư viện train LoRA. CPU chỉ cần `uv sync`. Trên Linux cần cài Tkinter của hệ điều hành và `sounddevice` để phát audio. Bản ghi micro tích hợp dùng Windows MCI.

`srt-quality` thêm Pedalboard có Rubber Band tích hợp để xử lý tốc độ cho cả SRT và các chức năng đọc khác. Tool dùng duy nhất Rubber Band; khi cần đổi tốc độ hoặc đọc SRT mà thiếu thư viện sẽ báo lỗi trước khi sinh audio. Đọc văn bản ở tốc độ gốc `1.0x` không cần chạy bộ chỉnh tốc độ. Nếu cập nhật môi trường đang dùng CUDA, giữ `--extra cuda` trong lệnh sync.

`video` thêm MoviePy 2 và imageio-ffmpeg để đọc/xuất video. Các chức năng TTS không cần import MoviePy khi khởi động. Launcher dùng `.venv` sẵn có thì cần cài extra này vào môi trường đó trước khi mở trang Video.

## Giao diện và điều hướng

Thanh bên trái sắp xếp theo quy trình: **Đọc văn bản → Đọc phụ đề SRT → Chỉnh tốc độ video → Tạo hàng loạt → Hội thoại → Thư viện giọng → Lịch sử & file → Cấu hình → Fine-tune & API**. Tool mở ở trang Văn bản; thanh nơi lưu được ẩn ở Văn bản/SRT, có thể đổi thư mục tại Cấu hình. Các trang cấu hình dài có thanh cuộn; bố cục cấu hình tự chuyển về một cột khi cửa sổ hẹp. Nút tạo audio được làm nổi bật, trạng thái model và tiến trình tách khỏi khu vực nhập nội dung.

1. **Cấu hình**: model/repo hoặc thư mục merge, auto/CPU/CUDA, ONNX/PyTorch, fp32/int8, dtype GPU, số luồng CPU, giới hạn streaming, thử lại câu ngắn. Chỉnh tốc độ đọc, temperature, top_k, top_p, phạt lặp/cửa sổ lặp, frame/đoạn, ký tự/đoạn và batch. Chọn EQ, compressor, giảm đỉnh; mã tham chiếu giữ đặc trưng giọng. Xuất WAV/FLAC/MP3; MP3 cần libsndfile có bộ mã hóa phù hợp.
2. **Văn bản**: nhập văn bản hoặc TXT/Markdown/PDF, chọn giọng và tốc độ ngay trong tab, tạo audio, audio + SRT từng câu hoặc streaming và nghe. Thêm nhiều TXT hoặc nhiều văn bản vào hàng đợi; xem tiến trình và lịch sử trong tab. Dùng giọng clone đã lưu ở Thư viện giọng. PDF scan cần OCR trước.
3. **Clone / Giọng**: chọn audio hoặc ghi micro, clone rồi lưu giọng, xóa giọng riêng, xuất embedding và codes thành NPZ, nhập/xuất thư viện JSON. Dùng clip sạch khoảng 3–8 giây, không cần lời thoại mẫu. Giọng có sẵn và bí danh được bảo vệ.
4. **Hàng loạt**: mỗi dòng là một văn bản hoặc chọn nhiều file. Tạo ZIP gồm audio đánh số theo đúng thứ tự và manifest JSON. Kết quả lẻ cũng giữ trong thư mục xuất. Giọng chung áp dụng cho toàn bộ batch.
5. **Hội thoại**: mỗi dòng `Nhân vật: lời thoại`, gán giọng cho từng nhân vật, đặt khoảng nghỉ giữa các lượt. Chọn giọng trước khi bấm tạo. Các lượt liên tiếp cùng giọng được tạo theo batch, tối đa **Số mục / batch**, để tận dụng GPU; thứ tự lượt và khoảng nghỉ được giữ nguyên.
6. **SRT**: thêm nhiều file vào hàng đợi, xem tiến trình từng file và lịch sử ngay trong tab; luôn đọc theo mốc bắt đầu tuyệt đối của phụ đề và giữ các khoảng trống. Tự tăng tốc từng câu khi cần vừa khung thời gian, không đẩy câu sau lùi đi. Có tốc độ ban đầu/tối thiểu riêng.
7. **Fine-tune / API**: chạy lần lượt prepare/train/merge/đóng gói giọng bằng script sẵn có; hoặc chạy API streaming model gốc với CPU/CUDA, cổng và API key. API chỉ bind `127.0.0.1`. Giọng của desktop và giọng đăng ký trong API là hai thư viện riêng.
8. **Lịch sử & file**: tự đọc lại file đã tạo khi mở tool, tìm theo tên/đường dẫn, lọc loại file, xem thời gian và dung lượng. Nghe audio, lưu bản sao, mở thư mục chứa file hoặc xóa nhiều file. Nhật ký hoạt động nằm ở thẻ riêng trong cùng trang.
9. **Video**: xem, tua và chỉnh tốc độ video. Xem trước và xuất MP4 chỉ có hình, không có âm thanh. Không cần tải model TTS.

## Xem video và chỉnh tốc độ nhiều đoạn trên timeline

Trong trang **Chỉnh tốc độ video**, bấm **Chọn…** hoặc nhập đường dẫn rồi **Mở video**. **Phát / Tạm dừng** xem hình theo tốc độ đã áp dụng, không phát tiếng.

Xem trước chỉ giải mã khung hình; không khởi tạo bộ phát hay bộ xử lý âm thanh.

Kéo trên thước thời gian để tua, hoặc dùng phím trái/phải khi timeline có focus để tua một giây. Kéo trên vùng trống của thanh bên dưới để chọn mốc bắt đầu/kết thúc, chọn tốc độ rồi bấm **Thêm đoạn**. Chọn một đoạn màu tím để chỉnh: kéo thanh tốc độ từ `0.25x` đến `4.0x`, kéo hai đầu để đổi độ dài, hoặc kéo thân đoạn để di chuyển. Có thể nhập số chính xác ở bảng bên phải rồi bấm **Áp dụng**. **Xóa đoạn / Xóa tất cả** bỏ các thay đổi tương ứng. Các đoạn không được chồng lên nhau; thao tác kéo được giới hạn bởi đoạn kề bên.

Mọi mốc tính theo **giây trong video gốc**, kể cả sau khi thay đổi tốc độ. Các vùng chưa chọn giữ `1.0x`. Danh sách bên phải hiển thị từng đoạn/tốc độ và tổng thời lượng video xuất dự kiến. Bấm **Xuất video** để tạo một MP4 mới từ toàn bộ các đoạn đã áp dụng; các số nhập nhưng chưa áp dụng không thay đổi bản xuất. Chọn video khác sẽ xóa kế hoạch chỉnh của video trước.

Ví dụ thêm đoạn `30–45s` ở `1.5x` và đoạn `60–70s` ở `0.8x`. Mỗi đoạn có thời gian chuyển tốc độ riêng ở hai đầu; mặc định `0.5s`. Trong đoạn đầu, tốc độ tăng từ `1.0x` lên `1.5x`, giữ ở giữa, rồi giảm về `1.0x`. Chuyển `0s` để đổi ngay; thời gian chuyển tối đa nửa độ dài đoạn. Kéo thu ngắn đoạn sẽ tự giảm thời gian chuyển nếu cần. Đổi tốc độ sẽ thay đổi tổng thời lượng và vị trí của phần sau đoạn chỉnh.

Bản xuất luôn không có track âm thanh. Tool bỏ qua đọc, chỉnh tốc độ, mã hóa và ghép âm thanh; thời gian tiết kiệm phụ thuộc video.

Tool xuất MP4 H.264, giữ FPS danh nghĩa của nguồn. Mức chất lượng thấp hơn cho chất lượng cao hơn/file lớn hơn; mặc định `20` (CRF khi dùng CPU, CQ khi dùng NVIDIA). Hai bộ mã hóa không bảo đảm cùng chất lượng hoặc kích thước file ở cùng một giá trị. Preset `ultrafast/veryfast/faster/fast/medium` đi từ nhanh đến chậm; mặc định `fast`, chọn `veryfast` để ưu tiên tốc độ. Kích thước lẻ được đệm thêm tối đa một pixel để phù hợp H.264/yuv420p. Chức năng này chưa nội suy chuyển động, nên độ mượt khi làm chậm phụ thuộc FPS gốc. Track phụ đề trong video hoặc file SRT đi kèm chưa được tự chuyển mốc.

Khi xuất, tool tự thử mã hóa một khung hình ở kích thước video bằng NVIDIA NVENC. Nếu GPU/bộ mã hóa không hỗ trợ, tool dùng CPU và cho bộ mã hóa tự tận dụng số luồng của máy. Cả căn mốc, đổi tốc độ ngay và chuyển tốc độ mượt đều xử lý hình trực tiếp trong FFmpeg, tránh chuyển từng khung RGB qua Python. Đường cong tốc độ được rút gọn với sai số thời gian tối đa 0,01 khung hình; giữ nguyên mốc đầu/cuối và tổng thời lượng dự kiến. Mức chất lượng, FPS và kích thước xuất giữ theo cấu hình. Thanh trạng thái báo giai đoạn mã hóa video, phần trăm và thời gian đã chạy. Cần đóng và mở lại app bằng `run_v3turbo_desktop.bat` để dùng bộ xuất mới.

File `video_<mã>.mp4` lưu trong thư mục đang chọn, xuất hiện trong **Lịch sử & file → Video**. **Mở / nghe** mở video bằng trình phát mặc định; có thể lưu bản sao hoặc xóa như các file khác. Nút **Dừng tác vụ / phát** hủy lượt xuất và dọn file dở. Giới hạn bộ nhớ theo các cửa sổ audio và khung hình, không nạp toàn bộ video vào RAM.

## Tự co giãn video theo hai file mốc

Mở video, rồi dùng khung **Tự căn video theo mốc** ở bên phải:

1. Bấm **Mốc cũ…** để nhập các thời điểm trong video gốc. Các mốc được ghim cố định lên timeline; bấm pin hoặc chọn dòng trong bảng để tua tới đúng vị trí nguồn.
2. Bấm **Mốc mới…** để nhập thời điểm đích. Bảng đối chiếu hiển thị số thứ tự và thời gian cũ/mới theo giây; di chuột lên pin để xem timestamp đầy đủ.
3. Bấm **Căn theo mốc**. Hệ thống tự tính tốc độ cho từng khoảng để mốc cũ trùng mốc mới có cùng số thứ tự. Bấm **Phát** để xem trước cả hình và tiếng; đồng hồ hiển thị thời gian gốc và thời gian mới.
4. Bấm **Xuất video**. Phần sau mốc cuối giữ tốc độ **1x**. Video gốc được giữ lại.

File văn bản UTF-8 có mỗi số thứ tự trên một dòng, timestamp `HH:MM:SS,mmm` ở dòng tiếp theo, có thể thêm dòng trống giữa các cặp. Hai file cần cùng tập số thứ tự, không trùng ID, và thời gian tăng dần. Mốc cũ không được vượt thời lượng video. Có sẵn [file mốc cũ](../examples/video_markers/old_marks.txt) và [file mốc mới](../examples/video_markers/new_marks.txt) theo ví dụ:

| Mốc | Cũ | Mới |
| --- | --- | --- |
| 1 | 00:00:00,060 | 00:00:00,133 |
| 2 | 00:00:03,130 | 00:00:02,995 |
| 3 | 00:00:06,290 | 00:00:05,666 |
| 4 | 00:00:08,950 | 00:00:07,666 |

Tốc độ mỗi khoảng bằng **độ dài khoảng cũ / độ dài khoảng mới**. Đoạn từ đầu video đến mốc đầu cũng được co giãn từ gốc `(0, 0)`. Nếu một file bắt đầu ở `00:00:00,000`, file kia cũng cần bắt đầu ở 0. Các đoạn đổi tốc độ ngay tại mốc; không áp dụng ramp vì ramp sẽ làm lệch thời điểm đích. Chế độ căn mốc cho phép tốc độ ngoài giới hạn chỉnh tay `0.25–4x` để giữ đúng yêu cầu, nên hãy nghe/xem trước nếu phải kéo giãn mạnh. Mốc được tính theo mili giây; hình xuất vẫn có độ phân giải thời gian theo FPS của nguồn.

Khi đang căn theo mốc, các nút/ô chỉnh tốc độ bằng tay được khóa. **Bỏ căn** khôi phục các đoạn chỉnh tay trước đó và giữ hai file mốc để có thể căn lại. Nhập file mốc khác sẽ bỏ bản căn hiện tại trước khi áp dụng bộ mốc mới. Đổi video xóa cả mốc và bản căn của video trước.

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
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_video_editor.py -v
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

Desktop và simple_tts chỉ dùng Rubber Band để đổi tốc độ, EQ sáng nhẹ, compressor và giảm đỉnh âm lượng. Desktop bật/tắt ba hiệu ứng ở **Cấu hình**, áp dụng cho văn bản, hàng đợi, Audio + SRT, hội thoại và SRT. Streaming có hiệu ứng sẽ chờ tổng hợp xong rồi xử lý một lần và phát đúng bản được xuất.

Đã bỏ tính năng khử nhiễu, watermark và fade cuối câu trong tool. Clone và xuất reference luôn dùng audio mẫu không khử nhiễu; các tham số False được truyền rõ để SDK không tự bật xử lý mặc định. Dữ liệu embedding/reference codes là phần của mô hình để nhận diện giọng, không phải hiệu ứng hậu xử lý.

Mặc định các hiệu ứng tắt. Khi tắt, tool không dựng chuỗi Pedalboard và không sao chép audio cho hậu xử lý; tốc độ 1.0x bỏ qua Rubber Band. Các hiệu ứng EQ/compressor được gộp thành một chuỗi. Căn mốc SRT, bỏ padding im lặng và chèn khoảng trống vẫn giữ để lời đọc khớp phụ đề. Không thay đổi tham số tổng hợp để đánh đổi chất lượng lấy tốc độ; chưa đo hiệu suất model thật.

## Giao diện rút gọn và tạo theo nhóm

Tool hiện có đúng năm tab: **Văn bản**, **SRT**, **Video**, **Lịch sử & file**, **Cấu hình**. Đã bỏ tab Hàng loạt/ZIP, Hội thoại, Thư viện giọng và Fine-tune/API, cùng các hàm quản lý riêng của chúng. Các file đã tạo và giọng đã lưu trước đây không bị xóa; giọng đã lưu vẫn được nạp khi tải model.

Hàng đợi văn bản và SRT vẫn nằm ngay trong hai tab. Hàng đợi văn bản gom các mục hợp lệ theo **Số mục / batch** và gọi `infer_batch`, xuất từng file audio riêng, không tạo ZIP. Trên GPU có thể tận dụng xử lý batch; CPU/ONNX vẫn xử lý tuần tự trong engine. Một lỗi suy luận của nhóm làm nhóm đó báo lỗi; có thể thử lại. Các file TXT lỗi/trống được báo riêng trước khi tổng hợp và không chặn các mục hợp lệ.

Model được dùng lại khi bấm Tải model với cùng cấu hình; đổi cấu hình sẽ tải lại. Giải phóng model rồi tải lại nếu cần đọc lại trọng số/giọng đã thay đổi trên đĩa. Danh sách giọng chỉ được cập nhật khi model thay đổi. Các sự kiện kết quả hàng đợi được gộp để tránh quét lịch sử nhiều lần trong cùng lượt cập nhật giao diện.

Các tối ưu giữ chất lượng FP32, tham số sampling và Rubber Band chất lượng cao; không tuyên bố mức tăng tốc khi chưa đo trên model thật.

## Tối ưu Audio + SRT theo cách tạo từng câu

Vẫn tạo từng câu theo batch, không dùng Stable-ts hoặc mô hình căn chỉnh bổ sung. Rubber Band chỉ chạy một lần trên mỗi câu khi tốc độ khác 1.0x. Sau khi ghép các câu và lấy mốc phụ đề, EQ/compressor/giảm đỉnh chỉ chạy một lần trên toàn track; giảm việc khởi tạo hiệu ứng và tránh nén/giảm gain riêng từng câu. Các hiệu ứng này không đổi số mẫu audio.

Trên CUDA, thử Số mục / batch từ 8 lên 16 nếu đủ bộ nhớ; không bảo đảm luôn nhanh hơn, cần đo trên máy. Trên CPU/ONNX, tăng batch không làm các câu chạy song song. Giữ tốc độ 1.0x và tắt hiệu ứng không cần dùng để giảm hậu xử lý. Chất lượng mô hình và chế độ Rubber Band vẫn được giữ; chưa đo mức tăng tốc với model thật.

## Mặc định Cấu hình

EQ, compressor và giảm đỉnh -1 dBFS mặc định bật khi mở desktop, có thể bỏ chọn. Ô Dùng mã audio tham chiếu đã bỏ khỏi giao diện; model vẫn dùng mã giọng theo mặc định.

## Tab Văn bản: hàng đợi audio + SRT

Tab Văn bản có hai cột: nội dung/giọng/tốc độ bên trái, **Hàng đợi** và **Lịch sử tạo** bên phải. Chỉ giữ **Thêm văn bản**, **Thêm TXT** và các nút quản lý hàng đợi. Mỗi lần thêm văn bản là một mục; nội dung không tự tách bằng dòng ===.

Bấm **Chạy hàng đợi** luôn xuất audio và SRT từng câu cho mỗi mục, kể cả file TXT. Các mục chạy lần lượt; câu trong mỗi mục vẫn được tổng hợp theo batch, hiệu ứng chạy một lần sau khi ghép. Thanh tiến trình hiển thị số câu hoàn thành của mục đang chạy.

Trong **Lịch sử tạo**, chọn file rồi bấm **Phát** hoặc nhấp đúp để nghe ngay bằng trình phát nội bộ của tool; bấm **Dừng** để dừng. **Xóa** yêu cầu xác nhận và xóa audio cùng SRT đi kèm, cập nhật cả lịch sử trong tab và lịch sử chung. Không thể xóa khi đang tạo/phát. Nút **Thư mục** vẫn mở nơi chứa file.

Bố cục tương tự cho phần hàng đợi/lịch sử của tab SRT. Các nút nhập TXT/PDF đơn, tạo audio riêng, streaming và tách nhiều văn bản đã được bỏ khỏi tab Văn bản.

## Nghe thử SRT có mốc bắt đầu muộn

Audio SRT vẫn giữ timeline tuyệt đối, bao gồm khoảng lặng dài trước câu đầu. Nút Phát trong lịch sử SRT/lịch sử chung tự tìm phần có tín hiệu và nghe từ trước đó 100 ms; không sửa file, không dời mốc xuất, không bỏ khoảng lặng giữa các câu. Trạng thái hiển thị mốc bắt đầu phát. Nếu không có tín hiệu trên toàn file, trình phát báo lỗi; luồng tạo SRT cũng chặn xuất track toàn im lặng.

## Đo thời gian Audio + SRT

Log của mỗi mục văn bản ghi tổng thời gian, thời gian model và thời gian hậu xử lý/xuất file. So sánh sau khi model đã tải và chạy thử một lượt, vì lần đầu có chi phí khởi tạo.

Trên PyTorch/CUDA, tool gom cửa sổ tối đa 4 × batch (không quá 128 câu) rồi để SDK sắp theo độ dài và xử lý từng batch. Giới hạn batch GPU vẫn đúng giá trị Cấu hình; không tăng số câu đồng thời trên GPU. CPU/ONNX giữ nhóm cũ và chạy tuần tự. Cách này giảm số lần gọi và cho SDK nhiều câu để ghép theo độ dài, nhưng chưa có số đo tăng tốc trên model thật.

## Tối ưu đọc file SRT

PyTorch/CUDA dùng cửa sổ câu lớn hơn để SDK sắp theo độ dài trước khi batch; giới hạn batch GPU vẫn giữ như cấu hình. ONNX/CPU giữ nhóm cũ. Chuỗi EQ/compressor dựng một lần cho mỗi file SRT rồi dùng lại, luôn reset trước mỗi câu để không mang trạng thái âm thanh sang câu tiếp theo. Rubber Band vẫn chỉ chạy khi cần chỉnh thời lượng và luôn giữ chất lượng cao; khoảng lặng/timeline không được đưa qua EQ.

Log mỗi file SRT ghi Total, model, tempo/effects và export để biết phần nào chậm. Đây là tối ưu giảm xử lý thừa; chưa đo mức tăng tốc với model thật. Tốc độ ban đầu SRT 1.0x sẽ bỏ qua Rubber Band với những câu vốn vừa khung; câu dài vẫn được tự tăng tốc.

## Thư viện mở video

File `run_v3turbo_desktop.bat` kiểm tra MoviePy 2.2+, Pillow và FFmpeg qua imageio-ffmpeg trước khi mở ứng dụng. Nếu thiếu, launcher dùng uv pip để cài bổ sung MoviePy vào chính .venv hiện có, không uv sync toàn bộ môi trường và không gỡ cấu hình CUDA. Lần cài bổ sung cần Internet. Nếu không cài được, tool vẫn khởi chạy để dùng các chức năng khác; lỗi thư viện video được báo khi mở video.

Có thể cài thủ công:
```powershell
uv pip install --python .venv/Scripts/python.exe "moviepy>=2.2,<3"
```

### Tốc độ xuất video

Trên trích đoạn 12 giây, 960×720, CPU libx264, preset fast và chất lượng 20: đường MoviePy cũ 2,55 giây, FFmpeg trực tiếp 1,73 giây. Đây là phép đo mẫu; không bảo đảm cùng mức tăng cho video khác hay toàn bộ file. Chọn preset **veryfast** để giảm chi phí mã hóa CPU; file có thể lớn hơn. Video vẫn phải mã hóa lại khi đổi tốc độ.
