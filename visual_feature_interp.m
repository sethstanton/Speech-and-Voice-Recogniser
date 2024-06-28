function visual_feat = visual_feature_interp(visual_feat, audio_feat)
    %   Return visual features matching the number of frames of the supplied audio
    %   feature. The time dimension must be the first dim of each feature
    %   matrix. Using the spline interpolation method.
    %
    %   Args:
    %   "visual_feat": the input visual features size: (visual_num_frames, visual_feature_len)
    %   "audio_feat": the input audio features size: (audio_num_frames, audio_feature_len)
    %
    %   Returns:
    %   "visual_feat_interp": visual features that match the time of the audio features.

video_timesteps = length(visual_feat);
audio_timesteps = length(audio_feat);

visual_x = 1:video_timesteps;
query_x = linspace(1, video_timesteps, audio_timesteps);

visual_feat = interp1(visual_x, visual_feat, query_x, 'spline');
end