# Input report: TinyCNN

- input: `model/TinyCNN/model.py`
- input type: `.py`
- execution: `CPU only`
- parameters: `{"calibration_samples": 2, "input_shape": [1, 16, 16, 3]}`
- optimization: `default`
- CMSIS-NN required: `yes`
- methods: `forward`

## CPU fallback operators

- `cortex_m::dequantize_per_tensor.out`
- `cortex_m::quantize_per_tensor.out`
- `cortex_m::quantized_conv2d_nhwc.out`
- `cortex_m::quantized_linear.out`
- `cortex_m::quantized_max_pool2d_nhwc.out`
- `cortex_m::transpose.out`
- `dim_order_ops::_clone_dim_order.out`
