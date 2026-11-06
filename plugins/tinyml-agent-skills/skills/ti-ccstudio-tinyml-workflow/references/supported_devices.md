# Supported Target Devices

The complete list of devices this skill can target. Step 2 validates the user's device against it. The **Device ID** column is the exact string to use as `$TARGET_DEVICE` (case-sensitive, e.g. `F2807x`, `F2837xS`). Matching is done on the Device ID, ignoring case and spaces.

**NPU** = has the TinyEngine hardware NPU. Only these devices have NPU template variants (names ending in `_npu`) and `_NPU` models; every other device runs the model on the CPU.

## C2000, C28 DSP

| Device ID | NPU |
|---|---|
| F28P55 | yes |
| F28P551x | no |
| F28P65 | no |
| F2837 | no |
| F2837xS | no |
| F2838x | no |
| F2807x | no |
| F28003 | no |
| F28004 | no |
| F28002x | no |
| F280013 | no |
| F280015 | no |
| F28E12 | no |

## C2000, C29 DSP

| Device ID | NPU |
|---|---|
| F29H85 | no |
| F29P58 | no |
| F29P32 | no |

## MSPM0 (Arm Cortex-M0+)

| Device ID | NPU |
|---|---|
| MSPM0G5187 | yes |
| MSPM0G3507 | no |
| MSPM0G3519 | no |

## MSPM33C (Arm Cortex-M33)

| Device ID | NPU |
|---|---|
| MSPM33C32 | no |
| MSPM33C34 | no |

## AM13 (Arm Cortex-M33)

| Device ID | NPU |
|---|---|
| AM13E2 | yes |

## AM26x (Arm Cortex-R5)

| Device ID | NPU |
|---|---|
| AM263 | no |
| AM263P | no |
| AM261 | no |

## Connectivity (Arm Cortex-M33 / M4)

| Device ID | NPU |
|---|---|
| CC2755 | no |
| CC2745 | no |
| CC1312 | no |
| CC1314 | no |
| CC1352 | no |
| CC1354 | no |
| CC35X1 | no |

## Radar (Arm Cortex-M4)

| Device ID | NPU |
|---|---|
| IWRL6432 | no |
