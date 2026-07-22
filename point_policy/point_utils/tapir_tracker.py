"""Causal TAPIR point tracker.

A thin online/streaming wrapper around DeepMind's TAPIR (pure-PyTorch, causal
variant) that replaces the non-commercial CoTracker used previously. One
instance tracks a fixed set of query points for a single camera view, advancing
an explicit causal state one frame at a time.

Frames are resized to a fixed square resolution before tracking (TAPIR, like
its reference demos, operates in a square space; feeding a non-square frame
mis-scales the x axis). Query points are scaled into that square space and the
output tracks are scaled per-axis back to the frame's native resolution, so
callers see [x, y] in native pixel coordinates -- matching what the rest of
PointsClass consumes. TAPIR's query convention is (t, y, x), so the incoming
(t, x, y) points are swapped.
"""

import sys

import cv2
import numpy as np
import torch
import tree


class TapirTracker:
    def __init__(self, checkpoint, device, tapnet_path, resize=512):
        """
        Parameters
        ----------
        checkpoint : str
            Path to the causal BootsTAPIR ``.pt`` state dict.
        device : str
            Torch device, e.g. ``"cuda"`` or ``"cpu"``.
        tapnet_path : str
            Path to the tapnet repo root (so ``tapnet.torch`` is importable).
        resize : int
            Square side length (multiple of 8) at which TAPIR tracks. 512 keeps
            more detail than TAPIR's 256 training resolution (and exceeds
            CoTracker's 384x512), trading some latency for accuracy.
        """
        if tapnet_path not in sys.path:
            sys.path.append(tapnet_path)
        from tapnet.torch import tapir_model

        self.device = device
        self.resize = resize
        # NOTE: ``use_casual_conv`` is misspelled in the upstream API; pass it
        # verbatim. The causal conv is what enables one-frame-at-a-time tracking.
        self.model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model = self.model.to(device).eval()
        torch.set_grad_enabled(False)

        self._reset_state()

    def _reset_state(self):
        self.query_features = None
        self.causal_state = None
        self.num_points = 0
        self.height = None
        self.width = None

    def reset(self):
        """Drop the query features and causal state for a new episode."""
        self._reset_state()

    def _preprocess(self, frame_rgb):
        """RGB uint8 HWC frame -> [1, 1, S, S, 3] float in [-1, 1] on device.

        The frame is resized to the fixed square working resolution.
        """
        if isinstance(frame_rgb, torch.Tensor):
            frame = frame_rgb.detach().cpu().numpy()
        else:
            frame = np.asarray(frame_rgb)
        frame = np.ascontiguousarray(frame)
        frame = cv2.resize(
            frame, (self.resize, self.resize), interpolation=cv2.INTER_LINEAR
        )
        frame = torch.from_numpy(frame).to(self.device)
        frame = frame.float() / 255.0 * 2 - 1  # [-1, 1]
        return frame[None, None]  # [1, 1, S, S, 3]

    def set_queries(self, frame_rgb, queries_xy):
        """Initialize query features + causal state from the query frame.

        Parameters
        ----------
        frame_rgb : np.ndarray
            RGB uint8 HWC image containing the query points (frame 0).
        queries_xy : array-like
            ``(N, 3)`` rows of ``(t, x, y)`` or ``(N, 2)`` rows of ``(x, y)`` in
            native pixel coordinates (the CoTracker/DIFT convention).
        """
        H, W = np.asarray(frame_rgb).shape[:2]
        self.height, self.width = H, W

        q = np.asarray(queries_xy, dtype=np.float32)
        if q.shape[-1] == 3:
            t, x, y = q[:, 0], q[:, 1], q[:, 2]
        else:
            x, y = q[:, 0], q[:, 1]
            t = np.zeros(len(q), dtype=np.float32)
        # Scale native (x, y) into the square working resolution, then emit the
        # (t, y, x) order TAPIR expects.
        x = x * self.resize / W
        y = y * self.resize / H
        query_points = np.stack([t, y, x], axis=-1).astype(np.float32)
        self.num_points = len(query_points)
        query_points = torch.from_numpy(query_points)[None].to(self.device)

        frame = self._preprocess(frame_rgb)
        feature_grids = self.model.get_feature_grids(frame, is_training=False)
        self.query_features = self.model.get_query_features(
            frame,
            is_training=False,
            query_points=query_points,
            feature_grids=feature_grids,
        )
        causal_state = self.model.construct_initial_causal_state(
            self.num_points, len(self.query_features.resolutions) - 1
        )
        self.causal_state = tree.map_structure(
            lambda t: t.to(self.device), causal_state
        )

    def predict(self, frame_rgb):
        """Track one frame, advancing the causal state.

        Returns
        -------
        coords : torch.Tensor
            ``(N, 2)`` of ``[x, y]`` in native pixel coordinates, clamped to the
            image bounds.
        visible : torch.Tensor
            ``(N,)`` boolean visibility.
        """
        if self.query_features is None:
            raise RuntimeError("set_queries() must be called before predict().")

        frame = self._preprocess(frame_rgb)
        feature_grids = self.model.get_feature_grids(frame, is_training=False)
        trajectories = self.model.estimate_trajectories(
            frame.shape[-3:-1],
            is_training=False,
            feature_grids=feature_grids,
            query_features=self.query_features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=self.causal_state,
            get_causal_context=True,
        )
        self.causal_state = trajectories["causal_context"]

        # [-1] selects the final (highest) refinement resolution.
        tracks = trajectories["tracks"][-1][0, :, 0]  # [N, 2] as [x, y]
        occlusion = trajectories["occlusion"][-1][0, :, 0]  # [N]
        expected_dist = trajectories["expected_dist"][-1][0, :, 0]  # [N]
        visible = (1 - torch.sigmoid(occlusion)) * (
            1 - torch.sigmoid(expected_dist)
        ) > 0.5

        # Scale from square working resolution back to native, per axis.
        coords = tracks.clone()
        coords[:, 0] = coords[:, 0] * self.width / self.resize
        coords[:, 1] = coords[:, 1] * self.height / self.resize
        # Clamp so points that drift off-frame (TAPIR still emits a coordinate
        # when occluded) can't crash downstream depth sampling.
        coords[:, 0] = coords[:, 0].clamp(0, self.width - 1)
        coords[:, 1] = coords[:, 1].clamp(0, self.height - 1)
        return coords.cpu(), visible.cpu()
