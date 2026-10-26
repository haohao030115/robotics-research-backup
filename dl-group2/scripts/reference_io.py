"""Shared CSV/NPY export for free-base references and validated episodes."""
import numpy as np
from expert_trajectory import save_arrays, write_json


def write_csv(path, array, header):
    np.savetxt(
        path,
        np.asarray(array),
        delimiter=",",
        fmt="%.9g",
        header=",".join(header),
        comments="",
    )


def export_reference(out, arrays, body_names, hand_names, metadata):
    out.mkdir(parents=True, exist_ok=True)
    save_arrays(out, arrays)
    t = arrays["timestamps"]
    velocities = metadata.pop("root_velocities")
    values = {"joint_pos": arrays["body_ref_q"], "joint_vel": arrays["body_ref_dq"],
              "body_pos": arrays["root_ref_pos"], "body_quat": arrays["root_ref_quat"],
              "body_lin_vel": velocities[0], "body_ang_vel": velocities[1]}
    for key, a in values.items():
        if key.startswith("joint"):
            prefix = "joint" if key == "joint_pos" else "joint_vel"
            header = [f"{prefix}_{j}" for j in range(29)]
        else:
            axes = "wxyz" if key == "body_quat" else "xyz"
            suffix = "vel_" if key in ("body_lin_vel", "body_ang_vel") else ""
            header = [f"body_0_{suffix}{axis}" for axis in axes]
        write_csv(out / f"{key}.csv", a, header)
    write_csv(out / "hand_ref_q.csv", arrays["hand_ref_q"], hand_names)
    write_csv(out / "hand_ref_dq.csv", arrays["hand_ref_dq"], hand_names)
    write_csv(out / "timestamps.csv", t[:, None], ["timestamp"])
    (out / "metadata.txt").write_text(f"Metadata for: {out.name}\nBody part indexes:\n[0]\n\nTotal timesteps: {len(t)}\n")
    (out / "info.txt").write_text(
        f"Motion Information: {out.name}\nsource: {metadata['candidate_type']}\n"
        "sample_rate_hz: 50\nquaternion_order: wxyz\nroot_velocities_frame: world\n"
        "expert_valid: false (requires physical SONIC+hand validation)\n" +
        "".join(f"{key}: {a.shape}\n" for key, a in arrays.items()))
    write_json(out / "metadata.json", metadata)
