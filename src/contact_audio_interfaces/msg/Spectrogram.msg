# Mel filterbank features for one analysis buffer.
#
# Convention: header.stamp = time of the FIRST sample in the buffer (never the
# time the features were computed), so a spectrogram stays aligned with the
# robot state that produced the sound.
#
# LAYOUT: data is row-major TIME-MAJOR, i.e. data[f * n_mels + m] is mel bin m
# of frame f. This matches torchaudio.compliance.kaldi.fbank, which returns
# (n_frames, n_mels) -- do not transpose it on the wire.
#
# The features are UNPADDED and UNNORMALISED. Each encoder applies its own
# padding and normalisation (AST pads to 1024 frames and normalises with its
# own mean/std), so this message carries only real measured frames.
std_msgs/Header header
uint64 start_sample_index    # per-channel frame index, counted at source_sample_rate

uint32 source_sample_rate    # what the microphone or WAV actually delivered
uint32 analysis_sample_rate  # what the filterbank was computed at (16000 for AST)

float32 window_duration_sec  # length of the analysed buffer, e.g. 1.0
float32 frame_length_ms      # filterbank window, e.g. 25.0
float32 frame_shift_ms       # filterbank step, e.g. 10.0  (NOT the publish rate)
float32 peak_gain            # divisor applied by peak normalisation; multiply back
                             # by this to recover absolute amplitude

uint16 n_frames              # 98 for a 1.0 s buffer at 16 kHz with 25/10 ms framing
uint16 n_mels                # 128 for AST
string feature_type          # 'kaldi_fbank'
string preprocessing_id      # digest of the full feature config; a consumer MUST
                             # refuse features whose id its model was not trained on
float32[] data               # length == n_frames * n_mels
