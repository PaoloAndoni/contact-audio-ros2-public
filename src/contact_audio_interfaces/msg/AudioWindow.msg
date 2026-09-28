# Fixed-length analysis window assembled from raw chunks.
# Convention: header.stamp = time of the FIRST frame in the window.
std_msgs/Header header
uint32 sample_rate
uint8 channels
uint64 start_sample_index   # per-channel frame index of the first frame; seconds = index / sample_rate
float32 duration_sec        # convenience; must equal (len(data)/channels)/sample_rate
int16[] data                # interleaved (L R L R ...) if channels > 1
