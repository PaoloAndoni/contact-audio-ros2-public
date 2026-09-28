# Raw audio chunk from a capture source (live mic or wav_player_node).
# Convention: header.stamp = time of the FIRST frame in this chunk.
std_msgs/Header header
uint32 sample_rate
uint8 channels
uint64 start_sample_index   # per-channel frame index; seconds = index / sample_rate
int16[] data                # interleaved (L R L R ...) if channels > 1
