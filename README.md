# CSC14005 - Đồ án 1: Hồi quy và Phân lớp

Kho lưu trữ này chứa mã nguồn, dữ liệu, notebook và báo cáo cho Đồ án 1 môn
CSC14005 - Machine Learning. Dự án gồm hai phần chính:

- Phần 1: bài toán hồi quy trên bộ dữ liệu California Housing Prices.
- Phần 2: bài toán phân lớp nhị phân trên bộ dữ liệu Adult Census Income, kèm
  một số dữ liệu tổng hợp để kiểm tra bài toán đa lớp và phi tuyến.

## Mục lục

- Tổng quan dự án
- Thành viên nhóm
- Bộ dữ liệu
- Cấu trúc thư mục
- Hướng tiếp cận kỹ thuật
- Hướng dẫn chạy
- Thư viện phụ thuộc
- Thông tin học thuật

## Tổng quan dự án

Dự án khảo sát các phương pháp học có giám sát nền tảng ở cả hai hướng hồi quy
và phân lớp. Mỗi phần đều có phân tích dữ liệu khám phá, tiền xử lý, huấn luyện
mô hình, đánh giá, kiểm định độ ổn định và phân tích kết quả trong báo cáo tổng
hợp.

Ở phần hồi quy, mục tiêu là dự đoán `median_house_value` từ các đặc trưng về vị
trí, dân cư, thu nhập và loại vị trí so với biển của các block group tại
California. Ở phần phân lớp, mục tiêu là dự đoán một cá nhân có thu nhập `>50K`
hay `<=50K` từ các thuộc tính trong bộ Adult Census. Với các yêu cầu không phù
hợp trực tiếp với Adult, chẳng hạn phân lớp đa lớp hoặc dữ liệu phi tuyến, nhóm
sử dụng thêm dữ liệu tổng hợp có kiểm soát.

## Thành viên nhóm

Nhóm: `Group_11`

| MSSV | Họ và tên | Phần phụ trách chính |
| --- | --- | --- |
| 23120084 | Nguyễn Mạnh Thắng | Classification: nhóm mô hình Logistic Regression, gồm GD, Newton-Raphson/IRLS, softmax, OvR/OvO, loss curves và so sánh hội tụ. |
| 23120063 | Nguyễn Thành Nguyên Anh | Classification: LDA/QDA, Fisher ratio, Perceptron, L1/L2, class weighting, CV, ROC/PR, calibration, McNemar, robustness, bonus models và tổng hợp báo cáo classification. |
| 21521062 | Trần Kim Ngọc | Regression: EDA, tiền xử lý, phần nâng cao gồm GPR, Robust Regression, phân tích Bias-Variance thực nghiệm và tổng hợp báo cáo. |
| 23120059 | Trần Đình Luân | Regression: regularization, feature selection, mô hình với hàm cơ sở phi tuyến, sensitivity và robustness. |
| 23120064 | Nguyễn Thiện Nhân | Regression: hồi quy tuyến tính, ablation study, Bayesian regression, Evidence Maximization và Kernel Ridge Regression. |

## Bộ dữ liệu

Dự án sử dụng hai file CSV được lưu trong thư mục `data/`:

| File | Bài toán | Số dòng | Biến mục tiêu | Ghi chú |
| --- | --- | ---: | --- | --- |
| `data/housing.csv` | Hồi quy | 20,640 | `median_house_value` | California Housing Prices. Notebook điền 207 giá trị thiếu ở `total_bedrooms`, tạo đặc trưng tỉ lệ, one-hot encode `ocean_proximity` và chia train/validation/test theo tỉ lệ 70/10/20. |
| `data/adult.csv` | Phân lớp | 32,561 | `income` | Adult Census Income. Notebook ánh xạ `<=50K` thành 0, `>50K` thành 1, xử lý giá trị `?`, tạo `capital_net`, label-encode một số đặc trưng phân loại và chia dữ liệu theo tỉ lệ 70/10/20. |

Chi tiết hơn về từng bộ dữ liệu nằm trong `data/README.md`.

## Cấu trúc thư mục

```text
Group_11/
|-- report/
|   |-- report.pdf
|   `-- report.tex
|-- code/
|   |-- Part1_Regression/
|   |   |-- notebook.ipynb
|   |   `-- utils.py
|   `-- Part2_Classification/
|       |-- notebook.ipynb
|       `-- utils.py
|-- data/
|   |-- adult.csv
|   |-- housing.csv
|   `-- README.md
|-- requirements.txt
`-- README.md
```

Trong repo hiện tại, file báo cáo có thể vẫn đang giữ tên gốc như
`report/Report.pdf` và `report/report_combined.tex`. Khi đóng gói nộp bài, có
thể đổi tên/copy sang `report/report.pdf` và `report/report.tex` theo đúng mẫu
thư mục yêu cầu.

## Hướng tiếp cận kỹ thuật

### Phần 1 - Hồi quy

Quy trình gồm: EDA, xử lý giá trị thiếu, phân tích ngoại lệ, tạo đặc trưng, mã
hóa biến phân loại, chia dữ liệu, chuẩn hóa đặc trưng, huấn luyện mô hình và
đánh giá. Các nhóm mô hình chính:

- OLS, Mini-batch Gradient Descent và Weighted Least Squares.
- Ridge, Lasso, Elastic Net và lựa chọn đặc trưng.
- Polynomial Basis, Gaussian RBF Basis và Cubic Spline.
- Bayesian Ridge, Kernel Ridge Regression, Gaussian Process Regression.
- Robust Regression bằng Huber IRLS.

Một số kết quả chính trong báo cáo:

- KRR với RBF kernel đạt kết quả mạnh nhất trong nhóm mô hình chính:
  RMSE khoảng `$60,088`, `R^2 = 0.7230`.
- Cubic Spline là lựa chọn thực tế tốt nhất: RMSE khoảng `$62,198`,
  `R^2 = 0.7032`, ổn định khi cross-validation và huấn luyện nhanh.
- Các mô hình tuyến tính và tuyến tính có regularization hội tụ quanh
  `R^2 ~= 0.656`, cho thấy nút thắt chính là bias cấu trúc chứ không phải
  variance.

### Phần 2 - Phân lớp

Phần phân lớp cài đặt và so sánh nhiều mô hình từ scratch hoặc qua các helper
có kiểm soát:

- Logistic Regression bằng Gradient Descent.
- Newton-Raphson/IRLS.
- Softmax Regression, One-vs-Rest và One-vs-One.
- LDA, QDA và Fisher ratio.
- Perceptron.
- Logistic Regression có L1/L2 regularization và class-weighted loss.
- Probit Regression, Laplace approximation, Kernel Logistic Regression và
  Gaussian Naive Bayes.

Một số kết quả chính trong báo cáo:

- LR-Newton là baseline cân bằng nhất ở threshold mặc định: Accuracy `0.8267`,
  F1 `0.5529`.
- LR-Weighted phù hợp nhất khi ưu tiên lớp thiểu số `>50K`: Recall `0.7666`,
  F1 `0.6091`.
- QDA có AUC cao nhất (`0.8526`) nhưng F1 thấp ở threshold `0.5`, cho thấy
  ranking xác suất và phân lớp theo một ngưỡng cố định có thể dẫn đến kết luận
  khác nhau.

## Hướng dẫn chạy

Tạo môi trường Python, kích hoạt môi trường và cài thư viện:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nên mở notebook từ đúng thư mục của từng phần vì notebook dùng đường dẫn tương
đối đến dữ liệu, ví dụ `../../data/housing.csv` và `../../data/adult.csv`.

Chạy phần hồi quy:

```powershell
cd code\Part1_Regression
jupyter notebook notebook.ipynb
```

Chạy phần phân lớp:

```powershell
cd code\Part2_Classification
jupyter notebook notebook.ipynb
```

Báo cáo tổng hợp nằm trong thư mục `report/`.

## Thư viện phụ thuộc

Dự án dùng một file `requirements.txt` chung ở thư mục gốc. Hai phần regression
và classification dùng các module con khác nhau, nhưng đều nằm trong cùng các
package chính như `scikit-learn`, `scipy`, `numpy` và `pandas`, nên không cần
tách thành hai file yêu cầu thư viện riêng.

Các package được ghim phiên bản:

- `numpy==2.1.3`
- `pandas==2.3.3`
- `scipy==1.16.3`
- `scikit-learn==1.7.2`
- `matplotlib==3.10.8`
- `seaborn==0.13.2`
- `jupyter==1.1.1`
- `ipykernel==7.2.0`

Khuyến nghị dùng Python 3.11 trở lên.

## Thông tin học thuật

### Thông tin môn học

- Môn học: CSC14005 - Machine Learning (Máy học)
- Bài nộp: Đồ án 1 - Hồi quy và Phân lớp
- Học kỳ: Học kỳ 2, năm học 2025-2026
- Khoa: Công nghệ Thông tin
- Trường: Trường Đại học Khoa học Tự nhiên, Đại học Quốc gia TP. Hồ Chí Minh

### Giảng viên và trợ giảng

- GV Lý thuyết: TS. Bùi Tiến Lên
  - E-mail: btlen@fit.hcmus.edu.vn
- GV Thực hành: Th.S Lê Nhựt Nam
  - E-mail: lnnam@fit.hcmus.edu.vn
- Trợ giảng: Trần Huy Bân
  - E-mail: huyban.han@gmail.com

### Cam kết học thuật

Nhóm cam kết phần mã nguồn, thí nghiệm và phân tích trong repository này do các
thành viên nhóm thực hiện cho mục đích học tập của môn CSC14005. Các thư viện
bên ngoài được sử dụng đúng vai trò công cụ và được thể hiện trong mã nguồn,
notebook hoặc báo cáo khi cần thiết.

### Hỗ trợ và câu hỏi

- Kênh trao đổi chính: nhóm Zalo của môn học.
- Lịch hỗ trợ: theo thông báo của giảng viên và trợ giảng.
- Vấn đề liên quan đến mã nguồn hoặc báo cáo: liên hệ thành viên phụ trách phần
  tương ứng trong nhóm.
