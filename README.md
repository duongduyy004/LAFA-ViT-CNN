# favit_lsda

`favit_lsda` là detector deepfake dựa trên FA-ViT và Latent Space Data
Augmentation (LSDA). Repo nhận face frame RGB đã crop và manifest từ bước
preprocess bên ngoài, sau đó huấn luyện và đánh giá theo protocol video-level.

## Pipeline và dữ liệu đầu vào

```text
FF++ / Celeb-DF-v2 video
        │ preprocess bên ngoài: detect, crop face, sample frame
        ▼
manifest train_pairs.csv ──► LSDA groups ──► FA-ViT + branch fusion
manifest validation/test ──► video-level selection/evaluation
```

`data.root` là thư mục gốc của ảnh. `data.train_pairs` phải có
`fake_path,real_path,method`; mỗi group gồm một real và các phương pháp giả mạo
được khai báo trong `model.forgery_methods`. Manifest frame dùng các cột
`path,label,video_id`, với `0 = real` và `1 = fake`.

Các trường manifest chính:

| Trường | Vai trò |
| --- | --- |
| `data.train_pairs` | Nguồn backprop, được nhóm theo `real_path`. |
| `data.validation_frames` | Nguồn FF++ để chọn checkpoint theo video AUC. |
| `data.celebdf_test_frames` | Target evaluation; nếu thiếu validation thì là fallback selection và có cảnh báo leakage. |
| `data.ffpp_test_frames` | Manifest mặc định cho `evaluate_ffpp.py`, không dùng để chọn checkpoint. |

Nếu đặt `validation_frames`, checkpoint được chọn trên AUC video-level của
manifest đó. Nếu không đặt, `train.py` cảnh báo và fallback sang
`celebdf_test_frames`. Target evaluation chỉ chạy sau khi `best.pt` đã cố định.

`FaceTransform` yêu cầu ảnh nguồn là RGB đúng ba kênh, áp dụng augmentation
chung hình học trong một group rồi normalize RGB về miền xấp xỉ `[-1, 1]`.
Dataset trả về mapping branch thay vì tensor positional:

```python
{"rgb": Tensor[3, H, W], "srm": Tensor[3, H, W], "fft": Tensor[3, H, W]}
```

Chỉ các key tương ứng branch bật mới xuất hiện. `GroupedForgeryDataset` stack
từng key thành `[groups, domains, 3, H, W]`; `FrameFaceDataset` trả về
`(branch_inputs, label, video_id)`.

## Kiến trúc multibranch

RGB/FA-ViT luôn bật và là nguồn duy nhất cho LSDA teachers, distillation, FAL
và domain-invariance. Hai branch forensic là tùy chọn:

| Slot | Input | Encoder |
| --- | --- | --- |
| RGB | ảnh RGB 3 kênh | `vit_base_patch16_224.augreg_in21k` + FA-ViT |
| SRM | tensor SRM 3 kênh | timm `xception` → projection về `embed_dim` |
| FFT | FFT log-magnitude 3 kênh | timm `mobilenetv3_small_100` → projection về `embed_dim` |

Fusion luôn giữ ba slot theo thứ tự `[RGB, SRM, FFT]`, mỗi slot rộng
`embed_dim`; slot branch tắt là tensor zero. `fixed_slot_concat` nối thành
`3 * embed_dim`, sau đó qua LayerNorm/GELU/dropout fusion trước binary head và
FAL. Branch tắt không được construct, không được execute và không tạo tham số.
Input mapping phải có đúng các key đã bật, đúng rank, shape, ba kênh và giá trị
finite.

### Toggle branch và pretrained policy

Các field hợp lệ trong `model`:

```yaml
enable_srm_branch: false
enable_fft_branch: false
srm_backbone: xception
fft_backbone: mobilenetv3_small_100
forensic_pretrained: true
```

`forensic_pretrained: true` dùng ImageNet-pretrained weights cho Xception và
MobileNetV3-Small. Hai encoder forensic và projection của chúng được
full-finetune; chỉ phần FA-ViT được điều chỉnh theo các cờ
`train_backbone_norms`, `train_cls_token` và `unfreeze_last_blocks`. Backbone
pretrained dùng `backbone_lr_multiplier`; projection, fusion và head dùng base
learning rate. Tests monkeypatch timm và không tải weights.

`model.pretrained: false` chỉ tắt pretrained weights của FA-ViT; nó không tự
ghi đè `forensic_pretrained`. Muốn chạy hoàn toàn offline/no-download, đặt cả
`model.pretrained: false` và `model.forensic_pretrained: false`. API
`build_model_from_config(..., pretrained=False)` và các đường `train.py` dùng
`--init-favit` hoặc `--resume` truyền override này, nên cũng tắt pretrained
weights của forensic encoders. Tất cả ảnh input của forensic encoder vẫn phải là
tensor floating-point hữu hạn với shape `[B, 3, H, W]`.

## Bốn ablation configs

Bốn file dưới đây giữ nguyên seed, data, augmentation, LSDA, loss và optimizer;
chỉ toggle branch và `output_dir` khác nhau:

| Config | SRM | FFT | Output |
| --- | ---: | ---: | --- |
| `configs/favit_lsda_rgb.yaml` | off | off | `outputs/favit_lsda_rgb` |
| `configs/favit_lsda_rgb_srm.yaml` | on | off | `outputs/favit_lsda_rgb_srm` |
| `configs/favit_lsda_rgb_fft.yaml` | off | on | `outputs/favit_lsda_rgb_fft` |
| `configs/favit_lsda_rgb_srm_fft.yaml` | on | on | `outputs/favit_lsda_rgb_srm_fft` |

Cấu hình Wavelet và tên config CNN legacy đã bị loại bỏ; chỉ bốn file trên
được `run_ffpp_tests.py` chạy.

## Cài đặt và lệnh chạy

```powershell
pip install -e ".[test]"

python train.py --config configs/favit_lsda_rgb.yaml --device cuda:0
python train.py --config configs/favit_lsda_rgb_srm.yaml --device cuda:0
python train.py --config configs/favit_lsda_rgb_fft.yaml --device cuda:0
python train.py --config configs/favit_lsda_rgb_srm_fft.yaml --device cuda:0
```

Resume checkpoint mới:

```powershell
python train.py `
  --config configs/favit_lsda_rgb_srm_fft.yaml `
  --resume outputs\favit_lsda_rgb_srm_fft\last.pt `
  --device cuda:0
```

Chạy đánh giá toàn bộ bốn case có checkpoint:

```powershell
python run_ffpp_tests.py --manifest E:\path\to\ffpp_c23_test_frames.csv --level video
```

Đánh giá một checkpoint:

```powershell
python evaluate_ffpp.py `
  --config configs/favit_lsda_rgb_srm_fft.yaml `
  --checkpoint outputs\favit_lsda_rgb_srm_fft\best.pt `
  --manifest E:\path\to\ffpp_c23_test_frames.csv `
  --level video --device cuda:0

python evaluate_celebdf.py `
  --config configs/favit_lsda_rgb_srm_fft.yaml `
  --checkpoint outputs\favit_lsda_rgb_srm_fft\best.pt `
  --level video --device cuda:0
```

`--level` nhận `frame` hoặc `video`, mặc định là `video`. Ở video-level, xác
suất các frame cùng `video_id` được trung bình trước khi tính AUC, accuracy,
F1, precision và recall.

## Checkpoint và migration

Checkpoint multibranch dùng:

```text
format_version: 4
architecture: favit_lsda_multibranch
enabled_branches: [rgb, srm?, fft?]
srm_backbone / fft_backbone: tên backbone hoặc null
fusion: fixed_slot_concat
```

Resume và evaluation kiểm tra strict architecture, version, enabled branches,
backbone names và fusion trước khi load `state_dict`. Checkpoint format v3 và
kiến trúc cũ bị từ chối có chủ đích vì shape/state contract không tương thích.
Các checkpoint v3 có thể còn chứa `artifact_mode`, `cnn_in_channels`,
`rgb_wavelet`, `srm_wavelet` hoặc tên backbone cũ như `FreqNet`; đây chỉ là
tham chiếu migration, không còn là runtime/config contract. Dùng
`--init-favit` để khởi tạo FA-ViT từ checkpoint tương thích, không dùng
`--resume` để lách kiểm tra v4.

`--init-favit` chỉ nạp các tensor FA-ViT tương thích; detector head, SRM/FFT
encoder, projection, late fusion và các module detector-specific luôn được
khởi tạo mới.

## Validation và protocol

```powershell
.venv\Scripts\python -m pytest -q
```

Không chọn hyperparameter hoặc checkpoint theo target test. Giữ cùng source
split, số frame, threshold và video-level protocol giữa các ablation; báo cáo
mean/std trên nhiều seed khi cần so sánh nghiên cứu.
