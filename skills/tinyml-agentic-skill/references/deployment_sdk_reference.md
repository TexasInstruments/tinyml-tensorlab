# Device & SDK Reference

Single source of truth for device family → SDK mapping.

## Device Family Classification

| Device ID | Family | SDK Name | Download |
|---|---|---|---|
| F280013, F280015, F28003, F28004 | c2000 | C2000Ware | https://www.ti.com/tool/C2000WARE |
| F2837, F28P55, F28P65, F29H85, F29P58, F29P32 | c2000 | C2000Ware | https://www.ti.com/tool/C2000WARE |
| MSPM0G3507, MSPM0G3519, MSPM0G5187 | mspm0 | MSPM0 SDK | https://www.ti.com/tool/MSPM0-SDK |
| MSPM33C32, MSPM33C34 | mspm33 | MSPM33 SDK | https://www.ti.com/tool/download/MSPM33-SDK |
| AM13E2 | am13 | MCU+ SDK | https://www.ti.com/tool/MCU-PLUS-SDK-AM263X |
| AM263, AM263P, AM261 | am26x | MCU+ SDK | https://www.ti.com/tool/MCU-PLUS-SDK-AM263X |
| CC2755, CC1352, CC1354, CC35X1 | simplelink | SimpleLink LP SDK | https://www.ti.com/tool/SIMPLELINK-LOWPOWER-F3-SDK |

## Device → CCS Type Mapping

| Device | CCS device_type | Default LaunchPad ccxml |
|---|---|---|
| F28P55 | f28p55x | TMS320F28P550SJ9_LaunchPad.ccxml |
| F28P65 | f28p65x | TMS320F28P650DH9.ccxml |
| F28004 | f28004x | TMS320F280049C_LaunchPad.ccxml |
| F2837 | f2837x | TMS320F28379D.ccxml |
| MSPM0G3507 | mspm0g3507 | MSPM0G3507.ccxml |
| MSPM0G5187 | mspm0g5187 | MSPM0G5187.ccxml |
| AM263 | am263 | AM263.ccxml |
| CC2755 | cc2755 | CC2755.ccxml |
| CC1352 | cc1352 | CC1352R.ccxml |

## SDK Installation Paths (Search Order)

### C2000Ware
```
~/ti/c2000ware_*
~/ti/C2000Ware_*
/opt/ti/c2000ware_*
/opt/ti/C2000Ware_*
~/ti/ccs*/c2000/C2000Ware_*
```

### MSPM0 SDK
```
~/ti/mspm0_sdk_*
/opt/ti/mspm0_sdk_*
```

### MSPM33 SDK
```
~/ti/mspm33_sdk_*
/opt/ti/mspm33_sdk_*
```

### MCU+ SDK (AM13 / AM26x)
```
~/ti/mcu_plus_sdk_am263x_*
~/ti/mcu_plus_sdk_*
/opt/ti/mcu_plus_sdk_*
```

### SimpleLink LP SDK
```
~/ti/simplelink_cc13xx_cc26xx_sdk_*
~/ti/simplelink_lowpower_f3_sdk_*
/opt/ti/simplelink_*
```

## AI Examples Path in SDK

Unified subpath structure across families:

| Family | AI Examples Path |
|---|---|
| c2000 | `{SDK_ROOT}/libraries/ai/examples` |
| mspm0 | `{SDK_ROOT}/examples` |
| mspm33 | `{SDK_ROOT}/examples` |
| am13 | `{SDK_ROOT}/examples` |
| am26x | `{SDK_ROOT}/examples` |
| simplelink | `{SDK_ROOT}/examples` |
