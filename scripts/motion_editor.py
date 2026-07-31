#!/usr/bin/env python3
"""
Motion Editor - mjviser-based humanoid motion viewer & editor.

Features:
  - Load MuJoCo model + pkl motion data
  - Play / pause / scrub animation in web browser
  - Select a frame and edit joint angles / root pose
  - Select two keyframes and re-interpolate (slerp for quaternions, linear for positions)

Usage:
  python motion_editor.py \
      --mjcf robot_assets/ths_t2_29dof/urdf/ths_t2_29dof.xml \
      --pkl  robot_assets/ths_t2_29dof/motion_data/walk_run/B9_-__Walk_turn_left_90_stageii.pkl \
      --port 8080
"""

import argparse
import threading
import time
from pathlib import Path

import joblib
import mjviser
import mujoco
import numpy as np
import trimesh.visual
import viser
import viser.transforms as vtf
from scipy.linalg import solve_banded

# ─────────────────────────── helpers ────────────────────────────────────────


def _enable_mjviser_mesh_transparency() -> None:
    """Patch mjviser's merge_geoms so exported GLB meshes render transparent.

    mjviser bakes geom colors into vertex colors (ColorVisuals), which export to
    GLB without any material, so three.js renders them opaque and ignores the
    vertex alpha channel. This wraps merge_geoms to convert each merged mesh's
    visual into a TextureVisuals carrying an alphaMode="BLEND" material while
    preserving the vertex colors, so the alpha channel actually takes effect.
    """
    import mjviser.scene as _mj_scene

    if getattr(_mj_scene, "_transparent_patched", False):
        return
    _orig_merge_geoms = _mj_scene.merge_geoms

    def _merge_geoms_transparent(mj_model, geom_ids):
        mesh = _orig_merge_geoms(mj_model, geom_ids)
        try:
            if mesh.visual.kind in ("vertex", "face"):
                vc = mesh.visual.vertex_colors.copy()
                mat = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                    alphaMode="BLEND",
                )
                tv = trimesh.visual.TextureVisuals(material=mat)
                tv.vertex_attributes = {"color": vc}
                mesh.visual = tv
        except Exception:
            pass
        return mesh

    _mj_scene.merge_geoms = _merge_geoms_transparent
    _mj_scene._transparent_patched = True


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions (wxyz)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = np.sin((1 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return a * q0 + b * q1


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + t * (b - a)


# ─────────────────────────── MotionEditor ───────────────────────────────────────


class MotionEditor:
    def __init__(self, mjcf_path: str, pkl_path: str, port: int = 8080):
        # ── paths ──
        self.mjcf_path = Path(mjcf_path).resolve()
        self.pkl_path = Path(pkl_path).resolve()

        # ── load MuJoCo model ──
        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)

        # ── load motion data ──
        raw = joblib.load(str(self.pkl_path))
        self.fps: float = float(raw["fps"])
        self.dt: float = 1.0 / self.fps
        self.num_frames: int = len(raw["root_pos_w"])
        self.motion_joint_names: list = list(raw["joint_names"])

        self.root_pos_w = np.array(raw["root_pos_w"], dtype=np.float64)  # (F, 3)
        self.root_quat_w = np.array(raw["root_quat_w"], dtype=np.float64)  # (F, 4)
        self.joint_pos = np.array(raw["joint_pos"], dtype=np.float64)  # (F, nj)

        # ── validate that all motion joints exist in the MuJoCo model ──
        for jname in self.motion_joint_names:
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname) < 0:
                raise ValueError(f"Joint '{jname}' not found in MuJoCo model")

        # ── working copy of motion data (editable) ──
        self.edit_root_pos = self.root_pos_w.copy()
        self.edit_root_quat = self.root_quat_w.copy()  # wxyz
        self.edit_joint_pos = self.joint_pos.copy()

        # ── state ──
        self.current_frame = 0
        self.playing = False
        self.play_speed = 1.0
        self.keyframe_a: int | None = None
        self.keyframe_b: int | None = None
        self._segment_play = False  # play only keyframe_a..keyframe_b
        self._pre_interp_state = None  # for undo
        self._mj_lock = threading.Lock()  # protect MuJoCo data access
        self._playback_path = []  # list of root_pos for green trajectory line
        self._green_line_created = False
        self._prev_frame = -1  # previous frame index for trajectory tracking
        self._suppress_edit_callbacks = False  # guard: ignore edit callbacks while syncing GUI

        # ── viser server + mjviser scene ──
        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+z")

        # ── add ground grid; use viser's built-in world axes ──
        self.server.scene.add_grid(
            name="/ground_grid",
            width=20.0,
            height=20.0,
            position=(0, 0, 0),
        )
        # Show viser's built-in world axes by default
        self.server.scene.world_axes.visible = True

        # ── robot transparency: bake 30% opacity into geom/material RGBA ──
        # (must be set BEFORE ViserMujocoScene builds the meshes)
        self.model.geom_rgba[:, 3] = 0.3
        self.model.mat_rgba[:, 3] = 0.3
        # Ensure the baked alpha actually renders (GLB alphaMode=BLEND)
        _enable_mjviser_mesh_transparency()

        # mjviser handles all mesh loading, body visualization, etc.
        self.mj_scene = mjviser.ViserMujocoScene(
            server=self.server,
            mj_model=self.model,
            num_envs=1,
        )
        # Disable camera tracking by default (Track camera unchecked)
        self.mj_scene.camera_tracking_enabled = False
        # Add visualization GUI tabs from mjviser
        self.mj_scene.create_visualization_gui()

        # ── trajectory visualization (point clouds, no flashing) ──
        # Red points: original root_pos trajectory
        orig_pts = np.array(self.root_pos_w, dtype=np.float32)
        self.server.scene.add_point_cloud(
            name="/traj_original",
            points=orig_pts,
            colors=(255, 0, 0),
            point_size=0.005,
            point_shape="circle",
        )
        # ── build custom GUI ──
        self._build_gui()

        # ── initial update ──
        self._apply_frame_to_mjdata(0)
        self.mj_scene.update_from_mjdata(self.data)

    # ─────────────────── FK & visualization ───────────────────────────────────

    def _apply_frame_to_mjdata(self, frame_idx: int):
        """Set MuJoCo data qpos from the edited motion data for a given frame.

        Follows play_motion_dataset.py:
            data.qpos[:3] = root_pos_w
            data.qpos[3:7] = root_quat_w
            data.qpos[7:] = joint_pos
        """
        self.data.qpos[:3] = self.edit_root_pos[frame_idx]
        self.data.qpos[3:7] = self.edit_root_quat[frame_idx]
        self.data.qpos[7:] = self.edit_joint_pos[frame_idx]

        # Run forward kinematics
        mujoco.mj_forward(self.model, self.data)

    def _update_visualization(self, frame_idx: int):
        """Update the viser visualization for a given frame (thread-safe)."""
        with self._mj_lock:
            self._apply_frame_to_mjdata(frame_idx)
            self.mj_scene.update_from_mjdata(self.data)
            root_pos = self.data.xpos[1].copy()  # base_link

        # Update green trajectory based on frame changes (works for play/scrub/step)
        if frame_idx != self._prev_frame:
            if frame_idx == self._prev_frame + 1:
                # Consecutive frame: append to path
                self._playback_path.append(root_pos.copy())
            else:
                # Frame jumped/scrubbed/looped: clear and restart from current frame
                self._playback_path = [root_pos.copy()]
                if self._green_line_created:
                    self.server.scene.remove_by_name("/traj_played")
                    self._green_line_created = False
            self._prev_frame = frame_idx

            # Draw green points whenever frame changes (need >= 2 points)
            if len(self._playback_path) > 1:
                pts = np.array(self._playback_path, dtype=np.float32)
                if self._green_line_created:
                    self.server.scene.remove_by_name("/traj_played")
                self.server.scene.add_point_cloud(
                    name="/traj_played",
                    points=pts,
                    colors=(0, 255, 0),
                    point_size=0.0055,
                    point_shape="circle",
                )
                self._green_line_created = True

    # ─────────────────── GUI ──────────────────────────────────────────────────

    def _build_gui(self):
        """Build the viser GUI controls."""
        gui = self.server.gui

        # ── Playback controls ──
        with gui.add_folder("Playback"):
            self.play_btn = gui.add_button("▶ Play / Pause")
            self.play_btn.on_click(self._on_play_toggle)

            self.speed_slider = gui.add_slider("Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0)
            self.speed_slider.on_update(self._on_speed_change)

            self.frame_slider = gui.add_slider(
                "Frame",
                min=0,
                max=self.num_frames - 1,
                step=1,
                initial_value=0,
            )
            self.frame_slider.on_update(self._on_frame_change)

            self.frame_label = gui.add_markdown(f"**Frame:** 0 / {self.num_frames - 1}")

            # Frame step buttons
            self.frame_minus_btn = gui.add_button("◀ -1")
            self.frame_minus_btn.on_click(self._on_frame_minus)
            self.frame_plus_btn = gui.add_button("+1 ▶")
            self.frame_plus_btn.on_click(self._on_frame_plus)

            self.time_label = gui.add_markdown("**Time:** 0.000 s")

        # ── Frame editing (uses current playback frame) ──
        with gui.add_folder("Edit Frame"):
            self.edit_info = gui.add_markdown("**Editing frame:** 0")

            # Joint selection dropdown
            self.joint_dropdown = gui.add_dropdown(
                "Joint",
                options=self.motion_joint_names,
                initial_value=self.motion_joint_names[0],
            )
            self.joint_dropdown.on_update(self._on_joint_select)

            # Joint value slider
            self.joint_value_slider = gui.add_slider(
                "Joint Value (rad)",
                min=-3.14,
                max=3.14,
                step=0.01,
                initial_value=0.0,
            )
            self.joint_value_slider.on_update(self._on_joint_value_change)

            # Root position editing
            with gui.add_folder("Root Position"):
                self.root_px = gui.add_slider("X (m)", min=-10, max=10, step=0.01, initial_value=0)
                self.root_py = gui.add_slider("Y (m)", min=-10, max=10, step=0.01, initial_value=0)
                self.root_pz = gui.add_slider("Z (m)", min=-10, max=10, step=0.01, initial_value=0)

            # Root orientation editing
            with gui.add_folder("Root Orientation (RPY rad)"):
                self.root_roll = gui.add_slider("Roll", min=-3.14, max=3.14, step=0.01, initial_value=0)
                self.root_pitch = gui.add_slider("Pitch", min=-3.14, max=3.14, step=0.01, initial_value=0)
                self.root_yaw = gui.add_slider("Yaw", min=-3.14, max=3.14, step=0.01, initial_value=0)

            # Reset the current frame's data back to the original motion
            self.reset_frame_btn = gui.add_button("↩ Reset This Frame")
            self.reset_frame_btn.on_click(self._on_reset_frame)

        # ── Keyframe interpolation ──
        with gui.add_folder("Keyframe Interpolation"):
            self.kf_label = gui.add_markdown("**Keyframe A:** —  **Keyframe B:** —")
            self.kf_warn = gui.add_markdown("")

            self.set_a_btn = gui.add_button("📌 Set Current Frame as A")
            self.set_a_btn.on_click(self._on_set_keyframe_a)
            self.set_b_btn = gui.add_button("📌 Set Current Frame as B")
            self.set_b_btn.on_click(self._on_set_keyframe_b)

            self.interpolate_btn = gui.add_button("🔄 Interpolate A→B")
            self.interpolate_btn.on_click(self._on_interpolate)

            # ── Smoothing ──
            gui.add_markdown("---")
            gui.add_markdown("**平滑优化 (A→B)**")
            self.smooth_alpha = gui.add_number("α 平滑强度", min=0.001, max=100.0, step=0.1, initial_value=1.0)
            with gui.add_folder("平滑目标"):
                self.smooth_root = gui.add_checkbox("Root Pos/Quat", initial_value=True)
                self.smooth_joints = gui.add_checkbox("Joints", initial_value=True)
            self.smooth_btn = gui.add_button("✨ Smooth A→B")
            self.smooth_btn.on_click(self._on_smooth)
            self.smooth_label = gui.add_markdown("")

            # Undo and segment play (hidden until interpolation is done)
            self.undo_interp_btn = gui.add_button("↩ Undo Interpolation")
            self.undo_interp_btn.on_click(self._on_undo_interpolate)
            self.undo_interp_btn.visible = False

            self.seg_play_btn = gui.add_button("▶ Play Segment A→B")
            self.seg_play_btn.on_click(self._on_seg_play_toggle)
            self.seg_play_btn.visible = False

            # ── Anchor frame insertion ──
            gui.add_markdown("---")
            gui.add_markdown("**Anchor 帧插入**")
            self.anchor_info = gui.add_markdown("当前帧: —")
            self.anchor_transition = gui.add_number("过渡帧数", min=1, max=500, step=1, initial_value=30)
            with gui.add_folder("插入方向"):
                self.anchor_before = gui.add_checkbox("在轨迹开始前插入", initial_value=True)
                self.anchor_after = gui.add_checkbox("在轨迹结束后插入", initial_value=False)
            self.anchor_btn = gui.add_button("📌 插入 Anchor 帧")
            self.anchor_btn.on_click(self._on_anchor_insert)
            self.anchor_label = gui.add_markdown("")

        # ── Save / Load ──
        with gui.add_folder("Save / Load"):
            self.save_btn = gui.add_button("💾 Save Edited Motion")
            self.save_btn.on_click(self._on_save)
            self.save_label = gui.add_markdown("")

            self.reset_btn = gui.add_button("↩ Reset to Original")
            self.reset_btn.on_click(self._on_reset)

        # Initialize sliders to current frame values BEFORE registering callbacks
        self._update_edit_frame_gui(0)
        self._on_joint_select(None)

        # NOW register callbacks (after values are set, so no accidental overwrite)
        for s in [self.root_px, self.root_py, self.root_pz]:
            s.on_update(self._on_root_pos_change)
        for s in [self.root_roll, self.root_pitch, self.root_yaw]:
            s.on_update(self._on_root_orient_change)

    # ─────────────────── callbacks ────────────────────────────────────────────

    def _on_play_toggle(self, _):
        self.playing = not self.playing
        if self.playing:
            self.play_btn.label = "⏸ Pause"
            self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self._play_thread.start()
        else:
            self.play_btn.label = "▶ Play / Pause"

    def _play_loop(self):
        while self.playing:
            # Render current frame FIRST (avoid flash after scrubbing)
            self._update_visualization(self.current_frame)
            with self.server.atomic():
                self.frame_slider.value = self.current_frame
                self._update_frame_info()
                self._update_edit_frame_gui(self.current_frame)
            time.sleep(self.dt / self.play_speed)
            if not self.playing:
                break
            # Then advance for next iteration
            if self._segment_play and self.keyframe_a is not None and self.keyframe_b is not None:
                if self.current_frame >= self.keyframe_b:
                    self.current_frame = self.keyframe_a
                else:
                    self.current_frame += 1
            else:
                total = len(self.edit_root_pos)
                self.current_frame = (self.current_frame + 1) % total

    def _on_speed_change(self, _):
        self.play_speed = self.speed_slider.value

    def _on_frame_change(self, _):
        if self.playing:
            return
        self.current_frame = int(self.frame_slider.value)
        self._update_visualization(self.current_frame)
        self._update_frame_info()
        self._update_edit_frame_gui(self.current_frame)

    def _on_frame_minus(self, _):
        """Step frame backward by 1."""
        if self.playing:
            return
        self.current_frame = max(0, self.current_frame - 1)
        self.frame_slider.value = self.current_frame
        self._update_visualization(self.current_frame)
        self._update_frame_info()
        self._update_edit_frame_gui(self.current_frame)

    def _on_frame_plus(self, _):
        """Step frame forward by 1."""
        if self.playing:
            return
        total = len(self.edit_root_pos)
        self.current_frame = min(total - 1, self.current_frame + 1)
        self.frame_slider.value = self.current_frame
        self._update_visualization(self.current_frame)
        self._update_frame_info()
        self._update_edit_frame_gui(self.current_frame)

    def _update_frame_info(self):
        """Update frame/time display labels."""
        total = len(self.edit_root_pos)
        self.frame_label.content = f"**Frame:** {self.current_frame} / {total - 1}"
        self.time_label.content = f"**Time:** {self.current_frame * self.dt:.3f} s"
        self.edit_info.content = f"**Editing frame:** {self.current_frame}"
        self.anchor_info.content = f"当前帧: **{self.current_frame}**"

    def _update_edit_frame_gui(self, frame: int):
        """Update the edit GUI to reflect the selected frame's values."""
        # Suppress edit callbacks while programmatically setting slider values,
        # otherwise they would write back and corrupt data (e.g. quat RPY round-trip).
        self._suppress_edit_callbacks = True
        try:
            # Update root position sliders
            self.root_px.value = float(self.edit_root_pos[frame, 0])
            self.root_py.value = float(self.edit_root_pos[frame, 1])
            self.root_pz.value = float(self.edit_root_pos[frame, 2])

            # Update root orientation (convert quat to euler)
            so3 = vtf.SO3.from_quaternion_xyzw(self.edit_root_quat[frame])
            rpy = so3.as_rpy_radians()
            self.root_roll.value = float(rpy[0])
            self.root_pitch.value = float(rpy[1])
            self.root_yaw.value = float(rpy[2])

            # Update joint slider
            self._on_joint_select(None)
        finally:
            self._suppress_edit_callbacks = False

        self.edit_info.content = f"**Editing frame:** {frame}"

    def _on_joint_select(self, _):
        """Update joint value slider when a new joint is selected."""
        frame = self.current_frame
        jname = self.joint_dropdown.value
        if jname in self.motion_joint_names:
            idx = self.motion_joint_names.index(jname)
            val = float(self.edit_joint_pos[frame, idx])
            self.joint_value_slider.value = float(np.clip(val, -3.14, 3.14))

    def _on_joint_value_change(self, _):
        """Apply joint value change to the current frame."""
        if self._suppress_edit_callbacks:
            return
        frame = self.current_frame
        jname = self.joint_dropdown.value
        if jname in self.motion_joint_names:
            idx = self.motion_joint_names.index(jname)
            self.edit_joint_pos[frame, idx] = self.joint_value_slider.value
            self._update_visualization(frame)

    def _on_root_pos_change(self, _):
        """Apply root position change to the current frame."""
        if self._suppress_edit_callbacks:
            return
        frame = self.current_frame
        self.edit_root_pos[frame, 0] = self.root_px.value
        self.edit_root_pos[frame, 1] = self.root_py.value
        self.edit_root_pos[frame, 2] = self.root_pz.value
        self._update_visualization(frame)

    def _on_root_orient_change(self, _):
        """Apply root orientation change (from RPY) to the current frame."""
        if self._suppress_edit_callbacks:
            return
        frame = self.current_frame
        roll = self.root_roll.value
        pitch = self.root_pitch.value
        yaw = self.root_yaw.value
        so3 = vtf.SO3.from_rpy_radians(roll, pitch, yaw)
        self.edit_root_quat[frame] = so3.as_quaternion_xyzw()
        self._update_visualization(frame)

    def _on_reset_frame(self, _):
        """Reset the current frame's data back to the original motion values."""
        frame = self.current_frame
        self.edit_root_pos[frame] = self.root_pos_w[frame].copy()
        self.edit_root_quat[frame] = self.root_quat_w[frame].copy()
        self.edit_joint_pos[frame] = self.joint_pos[frame].copy()
        self._update_visualization(frame)
        self._update_edit_frame_gui(frame)

    def _on_set_keyframe_a(self, _):
        """Set current frame as keyframe A (start)."""
        self.keyframe_a = self.current_frame
        self._update_kf_label()

    def _on_set_keyframe_b(self, _):
        """Set current frame as keyframe B (end)."""
        self.keyframe_b = self.current_frame
        self._update_kf_label()

    def _update_kf_label(self):
        """Update keyframe display and validation."""
        a_str = str(self.keyframe_a) if self.keyframe_a is not None else "—"
        b_str = str(self.keyframe_b) if self.keyframe_b is not None else "—"
        self.kf_label.content = f"**Keyframe A:** {a_str}  **Keyframe B:** {b_str}"
        # Validate
        if self.keyframe_a is not None and self.keyframe_b is not None:
            if self.keyframe_a >= self.keyframe_b:
                self.kf_warn.content = "⚠️ **结束帧必须大于开始帧，请重新设置！**"
            else:
                self.kf_warn.content = f"将插值 {self.keyframe_b - self.keyframe_a} 帧"

    def _on_interpolate(self, _):
        """Interpolate between keyframe A and keyframe B."""
        fa = self.keyframe_a
        fb = self.keyframe_b
        if fa is None or fb is None:
            self.kf_warn.content = "⚠️ **请先设置开始帧和结束帧！**"
            return
        if fa >= fb:
            self.kf_warn.content = "⚠️ **结束帧必须大于开始帧，请重新调整！**"
            return
        # Save pre-interpolation state for undo
        self._pre_interp_state = (
            self.edit_root_pos.copy(),
            self.edit_root_quat.copy(),
            self.edit_joint_pos.copy(),
        )
        n_interp = fb - fa
        for i in range(1, n_interp):
            t = i / n_interp
            idx = fa + i
            self.edit_root_pos[idx] = lerp(self.edit_root_pos[fa], self.edit_root_pos[fb], t)
            self.edit_root_quat[idx] = slerp(self.edit_root_quat[fa], self.edit_root_quat[fb], t)
            self.edit_joint_pos[idx] = lerp(self.edit_joint_pos[fa], self.edit_joint_pos[fb], t)
        # Update visualization (do NOT call _update_edit_frame_gui — it triggers
        # edit slider callbacks that flash other frames)
        self._update_visualization(self.current_frame)
        self.kf_warn.content = f"✅ 已插值帧 {fa} → {fb}"
        self.save_label.content = f"Interpolated frames {fa} → {fb}"
        # Show undo and segment play buttons
        self.undo_interp_btn.visible = True
        self.seg_play_btn.visible = True

    def _on_smooth(self, _):
        """Smooth trajectory between keyframe A and B.

        Minimizes: L = sum_t (x'_t - x_t)^2 + alpha * (x'_{t+1} - 2x'_t + x'_{t-1})^2
        This is a quadratic optimization with a pentadiagonal linear system.
        """
        fa = self.keyframe_a
        fb = self.keyframe_b
        if fa is None or fb is None:
            self.smooth_label.content = "⚠️ **请先设置 A/B 关键帧！**"
            return
        if fa >= fb:
            self.smooth_label.content = "⚠️ **结束帧必须大于开始帧！**"
            return
        if fb - fa < 3:
            self.smooth_label.content = "⚠️ **A/B 之间至少需要 3 帧！**"
            return

        smooth_root = self.smooth_root.value
        smooth_joints = self.smooth_joints.value
        if not smooth_root and not smooth_joints:
            self.smooth_label.content = "⚠️ **请至少选择一种平滑目标！**"
            return

        alpha = float(self.smooth_alpha.value)
        # Save state for undo
        self._pre_interp_state = (
            self.edit_root_pos.copy(),
            self.edit_root_quat.copy(),
            self.edit_joint_pos.copy(),
        )
        n = fb - fa - 1  # number of interior points
        # Build banded matrix for solve_banded (l=u=2)
        ab = np.zeros((5, n), dtype=np.float64)
        ab[2, :] = 1.0 + 6.0 * alpha  # main diagonal
        ab[1, 1:] = -2.0 * alpha  # +1 super-diagonal
        ab[3, :-1] = -2.0 * alpha  # -1 sub-diagonal
        ab[0, 2:] = alpha  # +2 super-diagonal
        ab[4, :-2] = alpha  # -2 sub-diagonal

        # Collect data channels based on selection
        channels = []
        channel_info = []  # (data_array, channel_idx, frame_range)

        if smooth_root:
            for c in range(3):  # root_pos
                channels.append(self.edit_root_pos[fa : fb + 1, c])
                channel_info.append(("root_pos", c))
            for c in range(4):  # root_quat
                channels.append(self.edit_root_quat[fa : fb + 1, c])
                channel_info.append(("root_quat", c))

        if smooth_joints:
            nj = self.edit_joint_pos.shape[1]
            for c in range(nj):  # joint_pos
                channels.append(self.edit_joint_pos[fa : fb + 1, c])
                channel_info.append(("joint_pos", c))

        data = np.array(channels)  # (C, fb-fa+1)

        # RHS = x_i + α*(x_{i+1} - 2x_i + x_{i-1}) for each interior point
        rhs = data[:, 1:-1].copy()  # (C, n)
        for i in range(n):
            col = i + 1  # column in data
            d2 = data[:, col + 1] - 2 * data[:, col] + data[:, col - 1]
            rhs[:, i] += alpha * d2

        # Boundary contributions (fixed endpoints)
        x_a = data[:, 0]
        x_b = data[:, -1]
        rhs[:, 0] += alpha * x_a
        if n > 1:
            rhs[:, 1] -= alpha * x_a
        if n > 2:
            rhs[:, n - 2] -= alpha * x_b
        rhs[:, n - 1] += alpha * x_b

        # Solve banded system for all channels
        result = solve_banded((2, 2), ab, rhs.T).T  # (C, n)

        # Write back smoothed values
        for idx, (dtype, c) in enumerate(channel_info):
            if dtype == "root_pos":
                self.edit_root_pos[fa + 1 : fb, c] = result[idx, :]
            elif dtype == "root_quat":
                self.edit_root_quat[fa + 1 : fb, c] = result[idx, :]
            elif dtype == "joint_pos":
                self.edit_joint_pos[fa + 1 : fb, c] = result[idx, :]

        # Re-normalize quaternions if root was smoothed
        if smooth_root:
            for i in range(fa + 1, fb):
                q = self.edit_root_quat[i]
                norm = np.linalg.norm(q)
                if norm > 1e-8:
                    self.edit_root_quat[i] = q / norm

        # Update visualization
        self._update_visualization(self.current_frame)

        # Build description
        targets = []
        if smooth_root:
            targets.append("Root")
        if smooth_joints:
            targets.append("Joints")
        target_str = " + ".join(targets)
        self.smooth_label.content = f"✅ 已平滑帧 {fa} → {fb} (α={alpha}, 目标={target_str})"
        self.undo_interp_btn.visible = True
        self.seg_play_btn.visible = True

    def _on_undo_interpolate(self, _):
        """Undo the last interpolation or anchor insertion."""
        if self._pre_interp_state is not None:
            # If segment play is active, stop it first
            if self._segment_play:
                self._segment_play = False
                self.playing = False
                self.play_btn.label = "▶ Play / Pause"
                self.seg_play_btn.label = "▶ Play Segment A→B"
            self.edit_root_pos, self.edit_root_quat, self.edit_joint_pos = self._pre_interp_state
            self._pre_interp_state = None
            # Restore frame_slider max
            total = len(self.edit_root_pos)
            self.frame_slider.max = total - 1
            # Return to keyframe B (or clamp)
            if self.keyframe_b is not None and self.keyframe_b < total:
                self.current_frame = self.keyframe_b
            elif self.current_frame >= total:
                self.current_frame = total - 1
            self.frame_slider.value = self.current_frame
            self._update_visualization(self.current_frame)
            self._update_frame_info()
            self.kf_warn.content = "↩️ 已撤销"
            self.anchor_label.content = "↩️ 已撤销"
            self.undo_interp_btn.visible = False
            self.seg_play_btn.visible = False

    def _on_seg_play_toggle(self, _):
        """Toggle segment playback between keyframe A and B."""
        self._segment_play = not self._segment_play
        if self._segment_play:
            if self.keyframe_a is None or self.keyframe_b is None or self.keyframe_a >= self.keyframe_b:
                self.kf_warn.content = "⚠️ **请先设置有效的开始帧和结束帧！**"
                self._segment_play = False
                return
            self.seg_play_btn.label = "⏹ Stop Segment Play"
            # Jump to keyframe A atomically to avoid flash
            self.current_frame = self.keyframe_a
            self._update_visualization(self.current_frame)
            with self.server.atomic():
                self.frame_slider.value = self.current_frame
                self._update_frame_info()
            # Auto-start playback if not already playing
            # (green trajectory is handled by frame-change detection in _update_visualization)
            if not self.playing:
                self.playing = True
                self.play_btn.label = "⏸ Pause"
                self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
                self._play_thread.start()
        else:
            self.seg_play_btn.label = "▶ Play Segment A→B"
            # Stop playback
            if self.playing:
                self.playing = False
                self.play_btn.label = "▶ Play / Pause"

    def _on_anchor_insert(self, _):
        """Insert current frame as anchor before trajectory start or after trajectory end.

        - If 'before': prepend N transition frames from anchor → original frame 0
        - If 'after': append N transition frames from original last frame → anchor
        Arrays are extended and frame_slider max is updated.
        """
        do_before = self.anchor_before.value
        do_after = self.anchor_after.value
        if not do_before and not do_after:
            self.anchor_label.content = "⚠️ **请至少选择一个插入方向！**"
            return
        n_trans = int(self.anchor_transition.value)
        # Save state for undo
        self._pre_interp_state = (
            self.edit_root_pos.copy(),
            self.edit_root_quat.copy(),
            self.edit_joint_pos.copy(),
        )
        # Anchor = current frame's JOINT angles only
        anchor_joint = self.edit_joint_pos[self.current_frame].copy()

        if do_before:
            # root_pos/root_quat come from frame 0 (trajectory start)
            start_pos = self.edit_root_pos[0].copy()
            start_quat = self.edit_root_quat[0].copy()
            # Build: [anchor] + [transition frames] + [original trajectory]
            # Anchor frame uses start's root + anchor's joints
            new_pos_list = [start_pos]
            new_quat_list = [start_quat]
            new_joint_list = [anchor_joint]
            for i in range(1, n_trans):
                t = i / n_trans
                new_pos_list.append(start_pos)  # root stays at start
                new_quat_list.append(start_quat)
                new_joint_list.append(lerp(anchor_joint, self.edit_joint_pos[0], t))
            # Prepend: new_frames + original
            new_pos = np.vstack([np.array(new_pos_list), self.edit_root_pos])
            new_quat = np.vstack([np.array(new_quat_list), self.edit_root_quat])
            new_joint = np.vstack([np.array(new_joint_list), self.edit_joint_pos])
            self.edit_root_pos = new_pos
            self.edit_root_quat = new_quat
            self.edit_joint_pos = new_joint
            # Adjust current_frame and keyframes
            offset = n_trans
            self.current_frame += offset
            if self.keyframe_a is not None:
                self.keyframe_a += offset
            if self.keyframe_b is not None:
                self.keyframe_b += offset
            self.anchor_label.content = f"✅ 在轨迹前插入 {n_trans} 帧过渡 + anchor"

        if do_after:
            # root_pos/root_quat come from last frame (trajectory end)
            last = len(self.edit_root_pos) - 1
            end_pos = self.edit_root_pos[last].copy()
            end_quat = self.edit_root_quat[last].copy()
            # Build: [original trajectory] + [transition frames] + [anchor]
            new_pos_list = []
            new_quat_list = []
            new_joint_list = []
            for i in range(1, n_trans):
                t = i / n_trans
                new_pos_list.append(end_pos)  # root stays at end
                new_quat_list.append(end_quat)
                new_joint_list.append(lerp(self.edit_joint_pos[last], anchor_joint, t))
            # Anchor frame uses end's root + anchor's joints
            new_pos_list.append(end_pos)
            new_quat_list.append(end_quat)
            new_joint_list.append(anchor_joint)
            # Append: original + new_frames
            new_pos = np.vstack([self.edit_root_pos, np.array(new_pos_list)])
            new_quat = np.vstack([self.edit_root_quat, np.array(new_quat_list)])
            new_joint = np.vstack([self.edit_joint_pos, np.array(new_joint_list)])
            self.edit_root_pos = new_pos
            self.edit_root_quat = new_quat
            self.edit_joint_pos = new_joint
            self.anchor_label.content = f"✅ 在轨迹后插入 {n_trans} 帧过渡 + anchor"

        # Update frame_slider max
        total = len(self.edit_root_pos)
        self.frame_slider.max = total - 1
        # Clamp current_frame
        if self.current_frame >= total:
            self.current_frame = total - 1
        self.frame_slider.value = self.current_frame
        self._update_visualization(self.current_frame)
        self._update_frame_info()
        # Show undo button
        self.undo_interp_btn.visible = True

    def _on_save(self, _):
        """Save the edited motion data to a pkl file."""
        save_path = self.pkl_path.parent / f"{self.pkl_path.stem}_edited.pkl"
        data = {
            "fps": self.fps,
            "root_pos_w": self.edit_root_pos.astype(np.float32),
            "root_quat_w": self.edit_root_quat.astype(np.float32),
            "joint_pos": self.edit_joint_pos.astype(np.float32),
            "joint_names": self.motion_joint_names,
        }
        # Preserve body data from original file
        raw = joblib.load(str(self.pkl_path))
        for key in ("body_pos_b", "body_quat_b", "body_names"):
            if key in raw:
                data[key] = raw[key]

        joblib.dump(data, str(save_path))
        self.save_label.content = f"Saved to: `{save_path.name}`"

    def _on_reset(self, _):
        """Reset all edits to original motion data."""
        self.edit_root_pos = self.root_pos_w.copy()
        self.edit_root_quat = self.root_quat_w.copy()
        self.edit_joint_pos = self.joint_pos.copy()
        self._update_visualization(self.current_frame)
        self._update_edit_frame_gui(self.current_frame)
        self.save_label.content = "Reset to original"

    # ─────────────────── run ──────────────────────────────────────────────────

    def run(self):
        """Keep the server running."""
        print(f"\n✅ Motion Editor running at http://localhost:{self.server.get_port()}")
        print(f"   Model: {self.mjcf_path}")
        print(f"   Motion: {self.pkl_path.name}")
        print(f"   Frames: {self.num_frames} @ {self.fps} fps")
        print(f"   Joints: {len(self.motion_joint_names)}")
        print(f"   Root pos range X: [{self.edit_root_pos[:, 0].min():.3f}, {self.edit_root_pos[:, 0].max():.3f}]")
        print(f"   Root pos range Y: [{self.edit_root_pos[:, 1].min():.3f}, {self.edit_root_pos[:, 1].max():.3f}]")
        print(f"   Root pos range Z: [{self.edit_root_pos[:, 2].min():.3f}, {self.edit_root_pos[:, 2].max():.3f}]")
        print("   Press Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nShutting down.")


# ─────────────────────────── main ───────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Motion Editor - mjviser-based motion viewer & editor")
    parser.add_argument(
        "--mjcf",
        type=str,
        default="robot_assets/ths_t2_29dof/urdf/ths_t2_29dof.xml",
        help="Path to MuJoCo MJCF XML file",
    )
    parser.add_argument(
        "--pkl",
        type=str,
        default="robot_assets/ths_t2_29dof/motion_data/walk_run/B9_-__Walk_turn_left_90_stageii.pkl",
        help="Path to pkl motion data file",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    editor = MotionEditor(args.mjcf, args.pkl, args.port)
    editor.run()


if __name__ == "__main__":
    main()
