# Contact-class prediction for one audio window.
# Convention: header.stamp = START time of the source window.
std_msgs/Header header
uint64 start_sample_index   # ties this prediction back to its source window,
                            # exactly as AstEmbedding does. Run-independent, so
                            # two pipelines can be compared window by window even
                            # when each drops a different subset under load.
string label
float32 confidence
string[] class_names        # parallel to probabilities
float32[] probabilities
