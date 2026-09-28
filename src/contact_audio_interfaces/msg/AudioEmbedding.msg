# A fixed-length embedding of one audio analysis buffer, from ANY encoder.
#
# This is deliberately not AstEmbedding. That message names an AST CLS token
# ('cls', 'model_id') and is still the contract of the retired four-node chain,
# which the parity gate in verify_sprint.sh replays to prove the two-node
# rebuild reproduces it. Renaming its fields to generalise it would break that
# evidence to save one file.
#
# Convention: header.stamp is COPIED UNCHANGED from the source Spectrogram, so
# it is the capture time of the FIRST sample of the analysed buffer -- not the
# time the encoder finished. A consumer aligning this against robot state must
# rely on that.
std_msgs/Header header
uint64 start_sample_index    # per-channel frame index of the source buffer,
                             # counted at the source sample rate; run-independent,
                             # so two runs can be compared window by window even
                             # when each drops a different subset under load

string encoder_id            # '<type>:<weights_source>:<16 hex>' -- the full
                             # identity (architecture, config, pinned revision,
                             # weight digest), not just a checkpoint name
string encoder_type          # 'ast' | 'ast_random' | 'cnn' | 'melstats'
uint32 embedding_dim         # length of embedding; carried explicitly so a
                             # consumer can reject a mismatch before reading

string preprocessing_id      # digest of the feature config of the SOURCE
                             # spectrogram, copied through. Two embeddings are
                             # comparable only if encoder_id AND preprocessing_id
                             # both match.
float32[] embedding          # length == embedding_dim
