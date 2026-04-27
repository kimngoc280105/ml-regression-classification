# Tài liệu mô tả dữ liệu

Thư mục này chứa các file dữ liệu CSV gốc được sử dụng trong notebook của dự án.
Các notebook giữ nguyên dữ liệu gốc và thực hiện tiền xử lý trong bộ nhớ khi
chạy thí nghiệm.

## Danh sách file

| File | Bài toán | Số dòng | Số cột | Biến mục tiêu | Nguồn được ghi trong báo cáo |
| --- | --- | ---: | ---: | --- | --- |
| `housing.csv` | Hồi quy | 20,640 | 10 | `median_house_value` | Kaggle - California Housing Prices |
| `adult.csv` | Phân lớp | 32,561 | 15 | `income` | UCI / Kaggle - Adult Census Income |

## `housing.csv` - California Housing Prices

`housing.csv` được sử dụng trong `code/Part1_Regression/notebook.ipynb` cho bài
toán hồi quy. Mỗi dòng biểu diễn một block group tại California. Biến mục tiêu
là giá trị nhà trung vị của block group đó.

Nguồn được trích dẫn trong báo cáo:
<https://www.kaggle.com/datasets/camnugent/california-housing-prices>

### Mô tả cột

| Cột | Data type | Role | Mô tả |
| --- | --- | --- | --- |
| `longitude` | numeric | Feature | Kinh độ của block group. |
| `latitude` | numeric | Feature | Vĩ độ của block group. |
| `housing_median_age` | numeric | Feature | Tuổi trung vị của nhà trong khu vực. |
| `total_rooms` | numeric | Feature | Tổng số phòng. |
| `total_bedrooms` | numeric | Feature | Tổng số phòng ngủ; có giá trị thiếu. |
| `population` | numeric | Feature | Dân số. |
| `households` | numeric | Feature | Số hộ gia đình. |
| `median_income` | numeric | Feature | Thu nhập trung vị của hộ gia đình, tính theo đơn vị 10,000 USD. |
| `median_house_value` | numeric | Target | Giá trị nhà trung vị, đơn vị USD. |
| `ocean_proximity` | categorical | Feature | Nhóm vị trí so với biển. |

### Ghi chú chất lượng dữ liệu

- `total_bedrooms` có 207 giá trị thiếu, khoảng 1.0% toàn bộ dữ liệu.
- `median_house_value` bị cap tại `$500,001`, tạo spike ở đuôi phải của phân
  phối biến mục tiêu.
- Các cột quy mô như `total_rooms`, `total_bedrooms`, `population` và
  `households` lệch phải mạnh và có nhiều ngoại lệ.
- Phân phối của `ocean_proximity`:

| Nhóm | Số lượng |
| --- | ---: |
| `<1H OCEAN` | 9,136 |
| `INLAND` | 6,551 |
| `NEAR OCEAN` | 2,658 |
| `NEAR BAY` | 2,290 |
| `ISLAND` | 5 |

### Tiền xử lý trong notebook hồi quy

- Điền giá trị thiếu ở `total_bedrooms` bằng median.
- Tạo ba đặc trưng tỉ lệ:
  - `rooms_per_household = total_rooms / households`
  - `bedrooms_per_room = total_bedrooms / total_rooms`
  - `population_per_household = population / households`
- One-hot encode `ocean_proximity` thành năm cột nhị phân.
- Tạo `income_cat` từ `median_income` chỉ để chia dữ liệu theo stratified split.
- Chia dữ liệu thành train/validation/test theo tỉ lệ 70/10/20:
  - Train: 14,448 dòng
  - Validation: 2,064 dòng
  - Test: 4,128 dòng
- Fit `StandardScaler` chỉ trên tập train, sau đó transform validation và test
  để tránh rò rỉ dữ liệu.

## `adult.csv` - Adult Census Income

`adult.csv` được sử dụng trong `code/Part2_Classification/notebook.ipynb` cho
bài toán phân lớp nhị phân. Biến mục tiêu cho biết thu nhập của một cá nhân có
lớn hơn 50K hay không.

Nguồn được trích dẫn trong báo cáo:
<https://www.kaggle.com/datasets/uciml/adult-census-income>

### Mô tả cột

| Cột | Data type | Role | Mô tả |
| --- | --- | --- | --- |
| `age` | numeric | Feature | Tuổi. |
| `workclass` | categorical | Feature | Nhóm công việc hoặc loại hình lao động. |
| `fnlwgt` | numeric | Feature | Trọng số mẫu trong điều tra dân số. |
| `education` | categorical | Feature | Trình độ học vấn ở dạng nhãn. |
| `education.num` | numeric | Feature | Trình độ học vấn ở dạng số. |
| `marital.status` | categorical | Feature | Tình trạng hôn nhân. |
| `occupation` | categorical | Feature | Nghề nghiệp. |
| `relationship` | categorical | Feature | Quan hệ trong hộ gia đình. |
| `race` | categorical | Feature | Nhóm chủng tộc. |
| `sex` | categorical | Feature | Giới tính. |
| `capital.gain` | numeric | Feature | Lợi nhuận vốn. |
| `capital.loss` | numeric | Feature | Lỗ vốn. |
| `hours.per.week` | numeric | Feature | Số giờ làm việc mỗi tuần. |
| `native.country` | categorical | Feature | Quốc gia bản địa. |
| `income` | categorical | Target | `<=50K` hoặc `>50K`. |

### Phân phối biến mục tiêu

| Lớp | Số lượng | Ý nghĩa |
| --- | ---: | --- |
| `<=50K` | 24,720 | Lớp âm, được ánh xạ thành `0`. |
| `>50K` | 7,841 | Lớp dương, được ánh xạ thành `1`. |

Lớp dương chiếm khoảng 24% dữ liệu, vì vậy đây là bài toán mất cân bằng lớp.
Do đó báo cáo sử dụng thêm F1, Recall, AUC, AP, balanced accuracy và confusion
matrix thay vì chỉ dựa vào Accuracy.

### Giá trị thiếu

Bộ Adult dùng ký hiệu `?` để biểu diễn giá trị thiếu ở một số cột phân loại:

| Cột | Số lượng `?` |
| --- | ---: |
| `workclass` | 1,836 |
| `occupation` | 1,843 |
| `native.country` | 583 |

### Tiền xử lý trong notebook phân lớp

- Thay `?` trong `workclass`, `occupation` và `native.country` bằng mode của
  từng cột.
- Ánh xạ nhãn mục tiêu:
  - `<=50K` -> `0`
  - `>50K` -> `1`
- Tạo `capital_net = capital.gain - capital.loss`.
- Sử dụng các đặc trưng số trong ma trận đặc trưng chính:
  - `age`
  - `education.num`
  - `hours.per.week`
  - `capital_net`
  - `capital.gain`
  - `capital.loss`
  - `fnlwgt`
- Label-encode các đặc trưng phân loại:
  - `workclass`
  - `occupation`
  - `relationship`
  - `sex`
  - `race`
- Ma trận đặc trưng Adult sau xử lý có 12 đặc trưng.
- Chia dữ liệu thành train/validation/test theo tỉ lệ 70/10/20:
  - Train: 22,792 dòng
  - Validation: 3,256 dòng
  - Test: 6,513 dòng
- Fit `StandardScaler` chỉ trên tập train, sau đó transform validation và test.

## Dữ liệu tổng hợp trong phần phân lớp

Notebook phân lớp còn tạo dữ liệu tổng hợp bằng `sklearn.datasets`, gồm
`make_classification`, `make_moons` và dữ liệu dạng XOR. Các bộ dữ liệu này
không được lưu thành file CSV trong thư mục `data/`. Chúng chỉ được dùng cho
các thí nghiệm có kiểm soát, chẳng hạn:

- so sánh Softmax, One-vs-Rest và One-vs-One trên bài toán đa lớp thật sự;
- trực quan hóa phép chiếu LDA/QDA và decision boundary;
- minh họa giới hạn của mô hình tuyến tính trên dữ liệu phi tuyến;
- kiểm tra Kernel Logistic Regression với RBF kernel.

## Ghi chú tái lập kết quả

- Không chỉnh sửa trực tiếp các file CSV gốc.
- Seed chính là `42`, trừ những phần sensitivity analysis có chủ đích thay đổi
  seed.
- Một số kết quả trung gian, chỉ số đánh giá và split indices được lưu trong
  `code/Part2_Classification/artifacts/`.
- Việc sử dụng dữ liệu cần tuân theo điều khoản của nguồn Kaggle/UCI tương ứng.
