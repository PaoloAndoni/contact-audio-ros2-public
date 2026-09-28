# AST CLS embedding for one audio window.
# Convention: header.stamp = START time of the source window (NOT inference completion time).
std_msgs/Header header
string model_id
uint64 start_sample_index   # ties this embedding back to its source window
float32 window_duration_sec
float32[] cls               # 768-D CLS token
