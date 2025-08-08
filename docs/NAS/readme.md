# Neural Architecture Search (NAS) in ModelMaker

ModelMaker supports Neural Architecture Search (NAS) for timeseries classification tasks, allowing users to automatically search for optimal neural network architectures based on their requirements. The NAS configuration is managed via YAML config files named `config.yaml`.

## Code Flow Overview

1. **Config Parsing**: The YAML config file is parsed at runtime. NAS-related parameters are read from the `training` section.
2. **NAS Activation**: If `nas_enabled: True`, the NAS module is invoked instead of using a static model architecture.
3. **NAS Search**: The NAS engine runs for a specified number of epochs (`nas_epochs`), optimizing either for memory or compute, as per `nas_optimization_mode`.
4. **Model Selection**: The best architecture found during the search is selected and trained according to the rest of the training configuration.
5. **Model Export**: The trained model can be tested and optionally compiled for deployment.

## NAS Configuration Arguments

> **Warning:** NAS is very compute intensive. A GPU is a must for practical use. Each NAS epoch can take several minutes to complete. Choose the number of epochs very wisely to avoid excessive runtimes.

All NAS-related arguments are specified under the `training` section in the YAML config file.

| Argument                    | Type    | Description                                                                                   | Example Values         |
|-----------------------------|---------|-----------------------------------------------------------------------------------------------|-----------------------|
| `nas_enabled`               | bool    | Enable or disable NAS.                                                                        | `True`, `False`       |
| `nas_epochs`                | int     | Number of epochs for which the NAS search is executed.                                        | `10`, `20`            |
| `nas_optimization_mode`     | str     | Optimization target for NAS.<br/>"Memory" optimises for parameters (read-only data on MCU).<br/>"Compute" optimises for MACs/FLOPS (peak SRAM usage). | `'Memory'`, `'Compute'` |
| `nas_model_size`            | str     | Preset model size for NAS. Determines search space complexity.                                | `'s'`, `'m'`, `'l'`, `'xl'`, `'xxl'` |

### Customization Arguments
Use these arguments if you want to control model size manually, instead of using presets

| Argument                    | Type    | Description                                                                                   | Example Values         |
|-----------------------------|---------|-----------------------------------------------------------------------------------------------|-----------------------|
| `nas_nodes_per_layer`       | int     | Number of nodes per layer for DAG construction.                                               | `4`                   |
| `nas_layers`                | int     | Number of layers in the architecture. Minimum is 3.                                           | `3`, `5`              |
| `nas_init_channels`         | int     | Initial feature map channels for the first conv layer.                                        | `1`, `8`              |
| `nas_init_channel_multiplier`| int    | Channel multiplier for the first layer.                                                       | `3`                   |
| `nas_fanout_concat`         | int     | Number of nodes per layer to concatenate for output.                                          | `4`                   |

**Note:** Only `nas_enabled`, `nas_epochs`, `nas_optimization_mode`, and `nas_model_size` are required for preset mode. The customization mode parameters are optional and allow advanced users to define the NAS search space in detail.

## NAS Model Size Presets: Detailed Configurations

When using NAS in preset mode, the `nas_model_size` parameter selects a predefined search space configuration. Each preset controls the complexity and size of the architectures explored by NAS. The detailed configurations are as follows:

| Preset | Layers (`nas_layers`) | Nodes per Layer (`nas_nodes_per_layer`) | Initial Channels (`nas_init_channels`) | Channel Multiplier (`nas_init_channel_multiplier`) | Fanout Concat (`nas_fanout_concat`) |
|--------|----------------------|-----------------------------------------|---------------------------------------|---------------------------------------------------|-------------------------------------|
| `s`    | 3                    | 4                                       | 1                                     | 3                                                 | 4                                   |
| `m`    | 10                   | 4                                       | 1                                     | 3                                                 | 4                                   |
| `l`    | 12                   | 4                                       | 4                                     | 3                                                 | 4                                   |
| `xl`   | 20                   | 4                                       | 4                                     | 3                                                 | 4                                   |
| `xxl`  | 20                   | 6                                       | 8                                     | 3                                                 | 4                                   |

- **Layers**: Number of layers in the architecture search space.
- **Nodes per Layer**: Number of nodes (operations) per layer in the DAG.
- **Initial Channels**: Number of feature map channels in the first convolutional layer.
- **Channel Multiplier**: Factor by which channels are increased in subsequent layers.
- **Fanout Concat**: Number of nodes per layer whose outputs are concatenated for the next layer.

These values are set automatically when you specify the preset via `nas_model_size` in the YAML config. For more control, use customization mode and set these parameters manually.

## How to Use NAS via YAML Config

1. **Enable NAS**: Set `nas_enabled: True` in the `training` section.
2. **Choose Preset or Customization Mode**:
    - **Preset Mode**: Set `nas_model_size` to one of `'s'`, `'m'`, `'l'`, or `'xl'`. This controls the search space size and model complexity.
    - **Customization Mode**: Uncomment and set values for `nas_nodes_per_layer`, `nas_layers`, `nas_init_channels`, `nas_init_channel_multiplier`, and `nas_fanout_concat` to define a custom model size.
3. **Set Optimization Target**: Use `nas_optimization_mode` to optimize for either memory or compute.
4. **Set Search Duration**: Adjust `nas_epochs` to control how long the NAS search runs.
5. **Other Training Parameters**: All other training parameters (batch size, learning rate, etc.) are compatible with NAS.

### Example: Preset Mode

```yaml
training:
    nas_enabled: True
    nas_epochs: 10
    nas_optimization_mode: 'Memory'
    nas_model_size: 'm'
```

### Example: Customization Mode

```yaml
training:
    nas_enabled: True
    nas_epochs: 20
    nas_optimization_mode: 'Compute'
    # Customization mode parameters
    nas_nodes_per_layer: 4
    nas_layers: 5
    nas_init_channels: 8
    nas_init_channel_multiplier: 2
    nas_fanout_concat: 3
```

## Tips for Using NAS

- **Preset mode** is recommended for most users and provides a balance between search space and ease of use.
- **Customization mode** is for advanced users who want fine-grained control over the architecture search space.
- Increasing `nas_epochs` can improve search results but will increase runtime.
- The `nas_optimization_mode` should be chosen based on deployment constraints (e.g., use `'Memory'` for devices with limited RAM).
- All NAS parameters can be adjusted in the YAML config file without modifying code.

## Full List of NAS Configs

- `nas_enabled`
- `nas_epochs`
- `nas_optimization_mode`
- `nas_model_size`
- `nas_nodes_per_layer`
- `nas_layers`
- `nas_init_channels`
- `nas_init_channel_multiplier`
- `nas_fanout_concat`

Refer to the example YAML config (`config.yaml`) for usage patterns and default values.

