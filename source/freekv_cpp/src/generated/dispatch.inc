#define _DISPATCH_CASES_head_dim(case_var, ...)         \
  _DISPATCH_CASE(128, case_var, __VA_ARGS__) \
// EOL

#define _DISPATCH_CASES_group_size(case_var, ...)         \
  _DISPATCH_CASE(1, case_var, __VA_ARGS__) \
  _DISPATCH_CASE(4, case_var, __VA_ARGS__) \
  _DISPATCH_CASE(5, case_var, __VA_ARGS__) \
  _DISPATCH_CASE(7, case_var, __VA_ARGS__) \
  _DISPATCH_CASE(8, case_var, __VA_ARGS__) \
// EOL

#define _DISPATCH_CASES_logits_post_hook(case_var, ...)         \
  _DISPATCH_CASE(LogitsPostHook::kNone, case_var, __VA_ARGS__) \
// EOL

#define _DISPATCH_CASES_kv_layout(case_var, ...)         \
  _DISPATCH_CASE(QKVLayout::kNHD, case_var, __VA_ARGS__) \
// EOL

#define _DISPATCH_CASES_pos_encoding_mode(case_var, ...)         \
  _DISPATCH_CASE(PosEncodingMode::kNone, case_var, __VA_ARGS__) \
// EOL

#define _DISPATCH_CASES_allow_fp16_qk_reduction(case_var, ...)         \
  _DISPATCH_CASE(false, case_var, __VA_ARGS__) \
// EOL

#define _DISPATCH_CASES_mask_mode(case_var, ...)         \
  _DISPATCH_CASE(MaskMode::kCausal, case_var, __VA_ARGS__) \
// EOL
