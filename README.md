# favit_lsda

`favit_lsda` là phiên bản FA-ViT được mở rộng bằng Latent Space Data
Augmentation (LSDA) cho generalized deepfake detection. Implementation dựa trên:

- [`fa_vit_remake`](../fa_vit_remake) cho GAM, LAM, FAL, manifest và protocol
  video-level Celeb-DF-v2;
- [LSDA, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_Transcending_Forgery_Specificity_with_Latent_Space_Augmentation_for_Generalizable_Deepfake_CVPR_2024_paper.pdf);
- [`lsda_detector_example.py`](lsda_detector_example.py) được cung cấp trong thư mục này.

## Tổng quan pipeline

Repo giả định khuôn mặt đã được phát hiện, crop thành frame ảnh và lập manifest
ở bước preprocess của `fa_vit_remake`. Pipeline trong repo này bắt đầu từ các
frame và manifest đó:

```text
video FF++ / Celeb-DF-v2
          │
          ├─ detect + crop face + lấy frame (bước preprocess bên ngoài repo này)
          │
          ├─ FF++ train_pairs.csv ──> nhóm LSDA ──> train FA-ViT + LSDA
          │                                      └─> best.pt / last.pt
          │
          ├─ FF++ validation_frames.csv ─────────> chọn best.pt, early stopping
          │  (nếu thiếu, fallback sang celebdf_test_frames.csv, kèm warning leak)
          │  (AUC cấp video: trung bình xác suất frame theo video_id)
          │
          └─ sau khi best.pt cố định, nếu celebdf_test_frames khác manifest chọn
             checkpoint ở trên, train.py chạy lại một lần duy nhất trên đó
             (cấp video, không ảnh hưởng selection)
```

Protocol mặc định (khi không cấu hình `validation_frames`) là train trên FF++ và
chọn checkpoint theo AUC cấp video trên Celeb-DF test — tức Celeb-DF được dùng
làm validation, chỉ không tham gia tính gradient. Nếu cần chọn checkpoint hoàn
toàn trong-miền (không leak), cấu hình `data.validation_frames` với một manifest
FF++ tách riêng.

## Xử lý dữ liệu

### Manifest và vai trò của từng split

| Trường cấu hình | Cột bắt buộc | Vai trò |
| --- | --- | --- |
| `data.train_pairs` | `fake_path,real_path,method` | Tạo các group FF++ dùng cho backprop; `video_id,sample_index` có thể được giữ để truy vết |
| `data.validation_frames` | `path,label,video_id` | **Khuyến nghị, không bắt buộc.** Nếu đặt, đây là tín hiệu chọn checkpoint: mỗi epoch, `best.pt` được cập nhật theo AUC **cấp video** trên manifest FF++ này. Nếu thiếu, `train.py` in warning và fallback sang dùng `data.celebdf_test_frames` làm tín hiệu chọn checkpoint |
| `data.ffpp_test_frames` | `path,label,video_id` | Không được `train.py` tự động dùng. Chỉ dùng làm manifest mặc định cho `evaluate_ffpp.py` khi không truyền `--manifest` |
| `data.celebdf_test_frames` | `path,label,video_id` | Nếu `validation_frames` không đặt, dùng làm tín hiệu chọn checkpoint. Nếu khác manifest đang dùng để chọn checkpoint, còn được `train.py` đánh giá **một lần** sau khi `best.pt` cố định |

Đường dẫn ảnh trong manifest có thể là đường dẫn tuyệt đối hoặc tương đối với
`data.root`. Nhãn nhị phân dùng `0 = real`, `1 = fake`.

> **Thay đổi hành vi:** ảnh nguồn không ở mode `RGB` (grayscale, RGBA, CMYK…)
> bị `FaceTransform` từ chối bằng `ValueError` kèm đường dẫn ảnh, thay vì
> được tự động convert sang RGB. Hãy convert sang RGB ở bước preprocess để mọi
> branch được sinh từ cùng dữ liệu ba kênh đã kiểm soát.

### Tạo group cho LSDA

`GroupedForgeryDataset` nhóm các dòng theo `real_path`. Một mẫu train có thứ tự
cố định:

```text
[real, Deepfakes, Face2Face, FaceSwap, NeuralTextures]
  0         1          2         3              4       <- domain label
  0         1          1         1              1       <- binary label
```

Chỉ các group có đủ bốn phương pháp giả mạo trong `model.forgery_methods` được
giữ lại; số group thiếu bị loại được in khi bắt đầu train. Mỗi lần đọc một group,
dataset chọn ngẫu nhiên một fake frame của từng method. Với cấu hình mặc định
`group_batch_size: 4`, tensor RGB đầu vào có dạng `[4, 5, 3, 224, 224]`,
tức 20 ảnh cho mỗi optimizer step. Batch phải có ít nhất hai group để việc tìm
hard example theo tâm miền trong LSDA có ý nghĩa.

### Augmentation ảnh và branch inputs

Trong cùng một group, real và bốn fake dùng chung crop và horizontal flip để giữ
căn chỉnh hình học. Ảnh được crop và resize về `image_size`; các biến đổi
photometric/codec sau đó được lấy mẫu độc lập cho từng ảnh, gồm color jitter,
grayscale, Gaussian blur, hạ rồi nâng độ phân giải và JPEG recompression. Cuối
cùng ảnh được chuyển thành tensor và normalize từng kênh bằng mean/std `0.5`
(miền giá trị xấp xỉ `[-1, 1]`).

Các augmentation mạnh chỉ được bật cho `train_pairs`. Validation và test dùng
`FaceTransform` sạch, không bật flip, color jitter hay degradation.

Sau augmentation, `FaceTransform` luôn tạo `rgb` và chỉ tạo các biểu diễn
forensic đã bật:

```python
{"rgb": Tensor[3, H, W], "srm": Tensor[3, H, W], "fft": Tensor[3, H, W]}
```

- `srm`: fixed 5x5 SRM convolution theo từng kênh;
- `fft`: `fftshift(log1p(abs(fft2(rgb))))` theo từng kênh;
- mỗi artifact được min-max normalize theo ảnh/kênh về `[-1, 1]`; kênh gần
  hằng được giữ bằng zero để không khuếch đại nhiễu số.

Mapping phải có đúng canonical keys của transform, mỗi giá trị là tensor float
hữu hạn `[3, image_size, image_size]`, và mọi branch có cùng hình học.
`GroupedForgeryDataset` stack mỗi key thành
`[groups, domains, 3, H, W]`; `FrameFaceDataset` trả về
`(branch_inputs, label, video_id)`.

> **`data.validation_frames` là tuỳ chọn:** nếu đặt, model selection dựa trên
> AUC **cấp video** đo trên manifest FF++ này, không leak Celeb-DF. Nếu không
> đặt, `train.py` fallback sang chọn checkpoint bằng AUC cấp video trên
> `data.celebdf_test_frames` (kèm warning). Target khác selection manifest chỉ
> được đánh giá một lần sau khi checkpoint đã cố định.

## Kiến trúc mô hình

### Sơ đồ khi huấn luyện

```text
group [real + 4 fake domains]
              │
              ├─ rgb ──► shared FA-ViT encoder ──► RGB feature ───────────┐
              │     │              │                                      │
              │     │              ├─ student/teacher maps ──► LSDA       │
              │     │              ├─ distillation                        │
              │     │              └─ domain + invariance objectives       │
              │     │                                                     │
              │     └─► CNNFeatureBranch ──► RGB-CNN slot ───────────────┤
              │                                                           │
              ├─ srm ──► Xception ──► projection ──► SRM slot / zero ─────┤
              │                                                           │
              └─ fft ──► EfficientNet-B4 ─────► projection ─► FFT slot/zero│
                                                                          ▼
                                                    fixed-slot concat + MLP
                                                                          │
                                                              fused features
                                                                  ┌───────┴───────┐
                                                                  ▼               ▼
                                                             binary head         FAL
```

### Các khối chính

1. **Shared FA-ViT encoder:** backbone
   `vit_base_patch16_224.augreg_in21k` tạo CLS token và patch map `14 x 14`.
   Global Adaptive Module (GAM) được chèn vào attention; nhánh spatial CNN và
   Local Adaptive Module (LAM) bổ sung đặc trưng cục bộ tại ba layer cấu hình.
2. **Student branch:** `ResidualLatentAdapter` biến đổi patch map bằng residual
   convolution có scale học được. Patch map sau adapter được mean-pool, ghép với
   CLS token rồi qua `vit_feature_fusion` để tạo RGB feature.
3. **Domain teacher branches:** một adapter cho real và một adapter riêng cho mỗi
   fake method tạo biểu diễn latent theo miền. Đây là các nhánh auxiliary nhẹ,
   không phải các mạng teacher pretrained độc lập.
4. **LSDA:** trên các fake teacher maps, Within-Domain (WD) chọn ngẫu nhiên một
   trong hard interpolation, centrifugal extrapolation, Gaussian perturbation,
   affine rotation hoặc difference transform. Cross-Domain (CD) Mixup trộn cặp
   fake domain bằng hệ số lấy từ phân phối Beta. WD, CD và latent gốc được fusion
   thành comprehensive target cho student.
5. **Domain invariance:** domain classifier học phân biệt real và từng fake
   method từ teacher maps. Một classifier khác nhận student RGB features qua
   Gradient Reversal Layer (GRL); gradient đảo chiều buộc student giảm thông tin
   đặc thù của từng phương pháp giả mạo.
6. **RGB CNN branch (`CNNFeatureBranch`):** **luôn bật**, không có toggle. Một
   CNN nhẹ chạy **song song** FA-ViT trên cùng tensor RGB — không đi qua ViT, không bơm vào block
   nào — rồi global-pool và projection `Linear -> LayerNorm` về `embed_dim`.
   Stack conv được port nguyên vẹn từ `CNN_feature_extractor_branch` của
   [Dual_Branch_FA_ViT_and_CNN](https://github.com/manhchienkmagpt/Dual_Branch_FA_ViT_and_CNN)
   (`training_model/models/favit_cnn.py`); chỉ độ rộng projection đổi từ
   `freq_dim` sang `embed_dim` để mọi fusion slot cùng width. Nhánh này khác
   hẳn `spatial_stem`/`SpatialCNN`: hai module đó tồn tại để bơm đặc trưng cục
   bộ **vào** ViT qua LAM và feature map cuối của chúng bị loại bỏ.
7. **SRM/Xception:** khi bật, Xception nhận duy nhất tensor SRM ba kênh, global
   pool rồi projection `Linear -> LayerNorm -> GELU -> Dropout` về
   `embed_dim`. Toàn bộ backbone và projection được fine-tune.
8. **FFT/EfficientNet-B4:** khi bật, EfficientNet-B4 nhận duy nhất FFT
   log-magnitude ba kênh và dùng cùng projection contract. Toàn bộ branch được
   fine-tune.
9. **Fixed-slot late fusion:** luôn concat theo thứ tự
   `[RGB, RGB-CNN, SRM, FFT]`. Hai slot đầu luôn được điền; SRM/FFT khi tắt
   không được construct hoặc execute và đóng góp `zeros_like(rgb_feature)`.
   Vector `4 * embed_dim` đi qua fusion MLP trước binary head và FAL.

LSDA teachers, latent augmentation, MSE distillation, teacher domain
classification và student domain invariance chỉ đọc biểu diễn RGB/FA-ViT.
Ngược lại, **FAL đọc late-fused features**, cùng representation đi vào binary
head; FAL không phải một objective RGB-only.

Khác với detector LSDA gốc, phiên bản này không chạy bốn EfficientNet teacher,
một ArcFace teacher và một student EfficientNet độc lập. Nó dùng một FA-ViT
encoder chung cùng các latent adapter nhẹ. Đây là thiết kế tích hợp LSDA vào
FA-ViT, không phải reproduction nguyên xi detector LSDA gốc.

### Cấu hình branch

```yaml
enable_srm_branch: false
enable_fft_branch: false
srm_backbone: tf_efficientnet_b4.ns_jft_in1k
fft_backbone: tf_efficientnet_b4.ns_jft_in1k
forensic_pretrained: true
```

RGB/FA-ViT và RGB CNN branch luôn bật, không cấu hình được. Hai toggle SRM/FFT
phải là YAML boolean thực (`true`/`false`), không phải chuỗi. RGB CNN branch
đọc lại `inputs["rgb"]` nên không thêm key vào input mapping và
`enabled_branches` không đổi. Field `enable_rgb_cnn_branch` bị từ chối như một
obsolete field, cùng nhóm với `artifact_mode` và `cnn_in_channels`. Backbone
names được kiểm tra qua allowlist (`SUPPORTED_SRM_BACKBONES`/
`SUPPORTED_FFT_BACKBONES` trong `favit_lsda/config.py`): SRM nhận `xception`,
`tf_efficientnet_b4`, `tf_efficientnet_b4.ns_jft_in1k`; FFT nhận
`tf_efficientnet_b4.ns_jft_in1k` và vẫn chấp nhận `efficientnet_b0` để tương
thích với config/checkpoint cũ.
`ProjectedForensicEncoder` re-normalize input từ quy ước pipeline ([-1, 1],
0.5/0.5) sang đúng mean/std pretrained của backbone (đọc từ
`backbone.pretrained_cfg`/`default_cfg`), nên đổi backbone không kéo theo lệch
chuẩn hoá. Re-normalize **chỉ chạy khi `forensic_pretrained: true`** — backbone
random init không có phân phối input kỳ vọng để khớp, nên biến đổi thành
identity. Với `xception` (mean=std=0.5) phép biến đổi cũng là identity; với
`efficientnet_b0` và `tf_efficientnet_b4` (ImageNet stats) thì không, nên
mọi checkpoint FFT/B4 train trước thay đổi này phải train lại — xem mục
checkpoint v6 bên dưới. Các config hiện dùng EfficientNet-B4 Noisy-Student cho
nhánh SRM.
`forensic_pretrained: true` dùng ImageNet initialization; encoder forensic và
projection được full-finetune. `model.pretrained: false` chỉ tắt pretrained
FA-ViT; để chạy hoàn toàn offline, đặt thêm `forensic_pretrained: false`.

Optimizer dùng `backbone_lr_multiplier` cho `backbone.*`,
`srm_encoder.backbone.*` và `fft_encoder.backbone.*`; `rgb_cnn_encoder.*`
(train from scratch), projection, fusion, adapter và head dùng base learning
rate.

### Sơ đồ khi inference

```text
frame RGB đã augment/normalize
        │
        ├─ rgb ─► shared FA-ViT ───────► RGB slot ───────────────┐
        ├─ rgb ─► CNNFeatureBranch ────► RGB-CNN slot ───────────┤
        ├─ srm ─► Xception ────────────► SRM slot (nếu bật) ─────┤
        └─ fft ─► EfficientNet-B4 ─────► FFT slot (nếu bật) ─────┤
                                                                ▼
                                                fixed-slot late fusion
                                                                │
                                                                ▼
                                                     binary head → P(fake)
```

Teacher adapters, LSDA augmenter, domain classifiers và các auxiliary loss
không được gọi khi inference.

## Phương pháp huấn luyện

### Hàm loss

Với mỗi grouped batch, model tối ưu tổng loss:

```text
L = λbin·Lbalanced-CE
  + λdomain·Ldomain-CE
  + λinvariance·Linvariance-CE
  + λdistill·(MSEreal + MSEfake)
  + λFAL·LFAL
```

- `Lbalanced-CE` lấy trung bình loss của lớp real và fake với trọng số ngang
  nhau, tránh tỷ lệ một real/bốn fake làm lớp fake lấn át;
- `Ldomain-CE` giám sát các domain teacher bằng năm nhãn miền;
- `Linvariance-CE` đi qua GRL: classifier cố nhận biết bốn fake method, trong
  khi student RGB encoder nhận gradient ngược để học đặc trưng bất biến;
- `MSEreal` distill real teacher map sang student real map; `MSEfake` distill
  comprehensive LSDA target sang student fake maps;
- `LFAL` dùng vector trọng số lớp real của binary head làm prototype, kéo
  **late-fused features** real lại gần prototype và đẩy fake ra xa theo cosine
  margin.

Trọng số thực tế lấy từ mục `loss` trong YAML, không mặc định đồng nhất với hệ
số trong paper. Cấu hình hiện tại dùng `λbin=1.0`, `λdomain=0.25`,
`λinvariance=0.1`, `λdistill=0.5` và `λFAL=0.25` sau warmup. Các loss
LSDA, invariance và FAL được ramp dần để detector học bài toán real/fake ổn định
trước khi nhận đầy đủ các ràng buộc auxiliary.

### Một epoch train

1. Dataloader shuffle các group và bỏ batch cuối nếu không đủ kích thước.
2. `forward_group` chạy encoder RGB chung một lần cho toàn bộ real/fake, chạy
   các forensic encoder đã bật, rồi tách RGB maps sang student và teacher/LSDA.
3. Tính năm thành phần loss, cộng theo trọng số của epoch rồi backprop.
4. Nếu chạy CUDA, AMP được bật theo cấu hình; gradient được clip bởi
   `max_grad_norm` trước optimizer step.
5. AdamW dùng learning rate nhỏ hơn cho các pretrained backbone
   (`backbone_lr_multiplier`), warmup rồi cosine decay. Mặc định backbone
   FA-ViT chủ yếu bị freeze; forensic backbones đã bật được full-finetune.

### Validation, checkpoint và cross-test

`data.validation_frames` là **tuỳ chọn**. Nếu đặt, mỗi epoch model được đánh
giá trên chính manifest này; nếu không, `train.py` in warning và dùng
`data.celebdf_test_frames` thay thế. Cả hai trường hợp đều dùng
`evaluate_at_level(...)` ở **cấp video** — xác suất các frame cùng `video_id`
được lấy trung bình trước khi tính AUC. `best.pt` được cập nhật khi AUC
video-level trên manifest chọn tăng, `last.pt` luôn lưu trạng thái mới nhất và
early stopping dựa trên cùng AUC này.

Sau khi vòng lặp train/early-stopping kết thúc, nếu
`data.celebdf_test_frames` khác manifest vừa dùng để chọn checkpoint,
`train.py` nạp lại `best.pt` và đánh giá **một lần duy nhất** trên đó, ghi
`celebdf_test_metrics` vào checkpoint và `final_target_evaluation` vào
`history.jsonl`. Đây là test post-selection thuần túy, không ảnh hưởng lựa
chọn checkpoint. `data.ffpp_test_frames` không được `train.py` dùng ở bước này.

### Nội dung `history.jsonl`

Mỗi dòng là một JSON object. Ba loại record:

| record | khi nào | nội dung |
| --- | --- | --- |
| epoch | mỗi epoch | `epoch`, `learning_rates`, `loss_weights`, `train`, và metrics của manifest chọn (`validation` hoặc `celebdf_test`) |
| `event: final_target_evaluation` | cuối run, chỉ khi có target riêng | `celebdf_test` metrics của `best.pt` |
| `event: best_model` | cuối mọi run có checkpoint mới | epoch được chọn và metrics của nó |

Record `best_model` là dòng cuối cùng và là chỗ duy nhất trong history nói
`best.pt` thuộc epoch nào:

```json
{
  "event": "best_model",
  "epoch": 7,
  "selection_name": "validation",
  "selection_metrics": { "level": "video", "auc": 0.9412, "...": "..." },
  "best_selection_auc": 0.9412,
  "checkpoint": "best.pt",
  "celebdf_test": { "level": "video", "auc": 0.8137, "...": "..." }
}
```

`celebdf_test` là `null` khi run không có target dataset riêng. Một run resume
mà không cải thiện AUC sẽ không ghi record này — epoch được chọn đã nằm trong
history của run sinh ra nó.

## Cài đặt và train

```powershell
pip install -e ".[test]"
python train.py --config configs/favit_lsda_rgb.yaml --device cuda:0
```

Resume checkpoint v6:

```powershell
python train.py `
  --config configs/favit_lsda_rgb_srm_fft.yaml `
  --resume outputs\favit_lsda_rgb_srm_fft\last.pt `
  --device cuda:0
```

### Bốn thí nghiệm branch có kiểm soát

Bốn config dùng chung seed, optimizer, schedule, augmentation, `image_size` và
manifest; chỉ branch toggles và `output_dir` thay đổi. Mọi config đều có
FA-ViT và RGB CNN branch — chỉ SRM/FFT là biến thí nghiệm:

| Config | SRM | FFT | `output_dir` |
| --- | ---: | ---: | --- |
| `configs/favit_lsda_rgb.yaml` | off | off | `outputs/favit_lsda_rgb` |
| `configs/favit_lsda_rgb_srm.yaml` | on | off | `outputs/favit_lsda_rgb_srm` |
| `configs/favit_lsda_rgb_fft.yaml` | off | on | `outputs/favit_lsda_rgb_fft` |
| `configs/favit_lsda_rgb_srm_fft.yaml` | on | on | `outputs/favit_lsda_rgb_srm_fft` |

```powershell
python train.py --config configs/favit_lsda_rgb.yaml
python train.py --config configs/favit_lsda_rgb_srm.yaml
python train.py --config configs/favit_lsda_rgb_fft.yaml
python train.py --config configs/favit_lsda_rgb_srm_fft.yaml
```

Tất cả config dùng `tf_efficientnet_b4.ns_jft_in1k` làm backbone cho cả
SRM và FFT; bốn config ablation chỉ khác nhau ở các
toggle SRM/FFT.

Cấu hình Wavelet và sáu tên config ArtifactCNN legacy đã bị loại bỏ.

## Baseline RGB + SRM + FFT không dùng LSDA

Pipeline baseline dùng lại face/frame loader và augmentation của repository,
sinh đủ ba input `rgb`, `srm`, `fft` cho mỗi frame. Mỗi nhánh đi qua một
backbone độc lập nhưng cùng kiến trúc được chọn bởi `model.backbone`. Ba feature
sau global pooling được concat cố định theo thứ tự `[RGB, SRM, FFT]`, rồi đi qua
linear binary head và được tối ưu trực tiếp bằng cross-entropy. Baseline không
có group LSDA, latent transform, teacher, distillation hay FAL.

```text
rgb ─► backbone RGB ─┐
srm ─► backbone SRM ─┼─► concat ─► binary head
fft ─► backbone FFT ─┘
```

Năm backbone và config tương ứng:

| Model | Config |
| --- | --- |
| EfficientNet-B4 | `configs/baselines/efficientnet_b4.yaml` |
| ResNet-50 | `configs/baselines/resnet50.yaml` |
| ViT-B/16 | `configs/baselines/vit_b16.yaml` |
| Swin-T | `configs/baselines/swin_t.yaml` |
| XceptionNet | `configs/baselines/xception.yaml` |

Mỗi config thực hiện cùng một protocol:

1. Flatten `ffpp_c23_train_pairs.csv` thành các frame real/fake độc lập; real
   frame trùng lặp được loại bỏ và cross-entropy được cân bằng theo số mẫu.
2. Fine-tune ba backbone pretrained cùng kiến trúc trên FF++.
3. Mỗi epoch đánh giá Celeb-DF ở **video level** (trung bình xác suất các frame
   cùng `video_id`) và lưu `best.pt` theo Celeb-DF video AUC.
4. Sau model selection, nạp lại `best.pt` và test một lần trên FF++ test ở
   video level. Kết quả được ghi vào checkpoint và `history.jsonl`.

Train từng model bằng Bash:

```bash
python train_baseline.py --config configs/baselines/efficientnet_b4.yaml --device cuda:0
python train_baseline.py --config configs/baselines/resnet50.yaml --device cuda:0
python train_baseline.py --config configs/baselines/vit_b16.yaml --device cuda:0
python train_baseline.py --config configs/baselines/swin_t.yaml --device cuda:0
python train_baseline.py --config configs/baselines/xception.yaml --device cuda:0
```

Test lại một checkpoint cụ thể trên FF++:

```bash
python evaluate_baseline.py \
  --config configs/baselines/resnet50.yaml \
  --checkpoint outputs/baselines/rgb_srm_fft/resnet50/best.pt \
  --level video \
  --device cuda:0
```

Test batch tất cả checkpoint hiện có (checkpoint chưa được train sẽ được bỏ
qua), đồng thời ghi `ffpp_test_result_video.json` vào từng output directory:

```bash
python run_baseline_ffpp_tests.py --level video --device cuda:0
```

Các đường dẫn mặc định nằm trong năm file YAML. `train_manifest` chấp nhận cả
manifest pair (`fake_path,real_path,method`) lẫn manifest frame
(`path,label,video_id`). Hai manifest validation/test phải có
`path,label,video_id` để aggregate metric theo video.

Checkpoint baseline ba nhánh dùng `format_version: 2`, architecture
`rgb_srm_fft_timm_concat_baseline`, `enabled_branches: [rgb, srm, fft]`,
`backbone_sharing: independent` và `fusion: concat`. Checkpoint RGB-only v1
không tương thích và không thể dùng để resume/evaluate kiến trúc mới.

## Checkpoint và migration

Checkpoint multibranch lưu:

```text
format_version: 6
architecture: favit_lsda_multibranch
enabled_branches: [rgb, srm?, fft?]
srm_backbone / fft_backbone: tên backbone hoặc null
fusion: fixed_slot_concat
```

Resume và evaluation kiểm tra architecture, version, enabled branches, backbone
names và fusion **trước khi xây model**. Checkpoint được deserialize trên CPU;
evaluation load weights khi model còn ở CPU rồi mới chuyển model sang device.
Resume vẫn phục hồi optimizer, scheduler, scaler và random state sau khi các
object tương ứng được xây.

Checkpoint format v3/`favit_lsda_cnn` và v4 đều bị từ chối vì state/shape
contract không tương thích — v4 có `late_fusion` rộng `3 * embed_dim`, trước
khi RGB-CNN slot bắt buộc được thêm. v5 bị từ chối vì lý do khác: shape vẫn
khớp, nhưng forensic branch của v5 được train trên input pipeline thô, trước khi
`ProjectedForensicEncoder` re-normalize sang mean/std của backbone pretrained;
load nó sẽ cho metric sai một cách âm thầm thay vì báo lỗi. Các từ legacy `artifact_mode`, `cnn_in_channels`,
`rgb_wavelet`, `srm_wavelet` và `FreqNet` chỉ còn được nhắc ở migration.
Dùng `--init-favit` để nạp các FA-ViT tensor tương thích vào một run mới;
detector head, RGB CNN branch, SRM/FFT encoders, projections và late fusion
được khởi tạo mới.

## Evaluate

Đánh giá FF++ ở video level:

```powershell
python evaluate_ffpp.py `
  --config configs/favit_lsda_rgb_srm_fft.yaml `
  --checkpoint outputs\favit_lsda_rgb_srm_fft\best.pt `
  --manifest E:\path\to\ffpp_c23_test_frames.csv `
  --level video `
  --device cuda:0
```

Nếu không truyền `--manifest`, script lần lượt tìm
`data.ffpp_test_frames` rồi `data.validation_frames`.

Đánh giá Celeb-DF-v2:

```powershell
python evaluate_celebdf.py `
  --config configs/favit_lsda_rgb_srm_fft.yaml `
  --checkpoint outputs\favit_lsda_rgb_srm_fft\best.pt `
  --level video `
  --device cuda:0
```

`--level` nhận `frame` hoặc `video` và mặc định là `video`. Cả hai script
trả về JSON gồm `accuracy`, `f1_score`, `precision`, `recall` và `auc`.
AUC dùng xác suất liên tục; bốn metric còn lại dùng `--threshold 0.5` (có thể
thay đổi), với fake (`label=1`) là positive class.

### Chạy batch cả năm case ablation

`run_ffpp_tests.py` gọi `evaluate_ffpp.py` lần lượt cho năm config branch
(`favit_lsda_rgb`, `..._rgb_srm`, `..._rgb_fft`, `..._rgb_srm_fft`,
`..._rgb_srm_effb4`) trên cùng một manifest FF++:

```powershell
python run_ffpp_tests.py `
  --manifest E:\Deepfake_Data_Chien\ffpp_celebdf_data\processed\manifests\ffpp_c23_test_frames.csv `
  --level video
```

Manifest mặc định đã là `ffpp_c23_test_frames.csv` nên có thể chạy gọn:

```powershell
python run_ffpp_tests.py
```

Dùng `--case CONFIG.yaml` (lặp lại được) để chỉ chạy một hoặc vài case thay vì
cả bốn — hữu ích khi chỉ một checkpoint sẵn sàng hoặc đang debug một case:

```powershell
python run_ffpp_tests.py --case favit_lsda_rgb.yaml
python run_ffpp_tests.py --case favit_lsda_rgb_srm.yaml
python run_ffpp_tests.py --case favit_lsda_rgb_fft.yaml
python run_ffpp_tests.py --case favit_lsda_rgb_srm_fft.yaml

# nhiều case cùng lúc
python run_ffpp_tests.py --case favit_lsda_rgb_srm.yaml --case favit_lsda_rgb_srm_fft.yaml
```

Với mỗi case, script đọc `output_dir` từ config, dùng checkpoint
`<output_dir>\best.pt` (đổi bằng `--checkpoint-name`) và ghi kết quả vào
`<output_dir>\ffpp_test_result_<level>.json`. Case thiếu checkpoint bị bỏ qua,
case chạy lỗi được ghi JSON có khoá `error`; các case còn lại vẫn chạy tiếp và
script thoát với exit code 1 nếu có case hỏng. Phải chạy từ thư mục gốc repo vì
đường dẫn `configs/` và `evaluate_ffpp.py` là tương đối.

## Test và ablation

```powershell
pytest
```

Ablation tối thiểu nên gồm:

1. FA-ViT: CE + FAL.
2. FA-ViT + domain adapters/domain loss.
3. FA-ViT + WD.
4. FA-ViT + CD.
5. `favit_lsda`: WD + CD + domain + distillation + FAL.
6. Lặp full recipe trên RGB, RGB+SRM, RGB+FFT và RGB+SRM+FFT.

Đây là một kiến trúc nghiên cứu mới; mức cải thiện cần được xác nhận bằng cùng
seed, split, số frame và checkpoint-selection protocol.

## Protocol thí nghiệm cross-dataset

Để kết quả phản ánh khả năng tổng quát hóa thay vì target-domain leakage:

1. Tách FF++ thành train/validation theo **video nguồn**, không theo frame.
2. Dùng duy nhất FF++ train để backprop và FF++ validation để chọn `best.pt`.
3. Không điều chỉnh hyperparameter, epoch hay checkpoint theo Celeb-DF test AUC.
4. Cố định checkpoint rồi mới test trên Celeb-DF-v2, DFDC hoặc WildDeepfake.
5. Giữ cùng split, số frame và cùng thang đo (mặc định là **cấp video**, khớp
   tín hiệu chọn checkpoint) giữa các phương pháp; chạy ít nhất ba seed và báo
   cáo mean/std AUC.

Ablation đề xuất, mỗi cấu hình chạy ít nhất ba seed:

1. baseline cũ;
2. thêm class-balanced CE + image degradation augmentation;
3. thêm residual-gated LSDA + auxiliary ramp;
4. thêm student domain invariance;
5. full recipe với AdamW/cosine và hai backbone block cuối được fine-tune;
6. so sánh bốn tổ hợp RGB/SRM/FFT dưới cùng protocol.

Không kết luận từ việc AUC tiếp tục tăng sau một epoch cụ thể. Tiêu chí quan
trọng là mean/std AUC trên target chưa thấy, với checkpoint chỉ được chọn từ
source validation.
