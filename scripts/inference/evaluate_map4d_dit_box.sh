#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/../../../.." && pwd)"

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-${PROJECT_ROOT}/codes/CoppeliaSim}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH:-${COPPELIASIM_ROOT}}"
CONDA_ENV_LIB="${CONDA_ENV_LIB:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/miniconda3/envs/4dmap/lib}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${CONDA_ENV_LIB}:${LD_LIBRARY_PATH:-}"
XVFB_CMD=(xvfb-run -a -s "-screen 0 1280x1024x24 +extension GLX +render -noreset")

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/../../dataset/rlbench2}"
DEFAULT_DEMO_PATH="${DATA_ROOT}/bimanual_push_box/test"
SQUASHFS_ROOT_DEMO_PATH="${DATA_ROOT}/squashfs-root"
if [[ -z "${RLBENCH2_TEST_DEMO_PATH:-}" ]]; then
  if [[ -d "${DEFAULT_DEMO_PATH}/all_variations/episodes" ]]; then
    export RLBENCH2_TEST_DEMO_PATH="${DEFAULT_DEMO_PATH}"
  elif [[ -d "${SQUASHFS_ROOT_DEMO_PATH}/all_variations/episodes" ]]; then
    export RLBENCH2_TEST_DEMO_PATH="${SQUASHFS_ROOT_DEMO_PATH}"
  else
    export RLBENCH2_TEST_DEMO_PATH="${DEFAULT_DEMO_PATH}"
  fi
fi

CFG_ONLY=0
for arg in "$@"; do
  if [[ "${arg}" == "--cfg" || "${arg}" == --cfg=* ]]; then
    CFG_ONLY=1
  fi
done

if [[ "${CFG_ONLY}" == "0" && -z "${MAP4D_DIT_CKPT:-}" ]]; then
  echo "MAP4D_DIT_CKPT must point to a Map4D DiT .pth.tar checkpoint" >&2
  exit 2
fi

if [[ ! -d "${RLBENCH2_TEST_DEMO_PATH}" ]]; then
  echo "RLBench2 test demo directory not found: ${RLBENCH2_TEST_DEMO_PATH}" >&2
  echo "Mount or extract ${DATA_ROOT}/bimanual_push_box.test.squashfs and set RLBENCH2_TEST_DEMO_PATH." >&2
  exit 2
fi
if [[ ! -d "${RLBENCH2_TEST_DEMO_PATH}/all_variations/episodes" ]]; then
  echo "RLBench2 demo directory does not contain all_variations/episodes: ${RLBENCH2_TEST_DEMO_PATH}" >&2
  exit 2
fi

cd "${ROOT_DIR}"
"${XVFB_CMD[@]}" conda run --no-capture-output -n 4dmap python eval_map4d_dit.py \
  framework.eval_from_eps_number="${EVAL_FROM_EPS_NUMBER:-0}" \
  framework.eval_episodes="${EVAL_EPISODES:-100}" \
  framework.eval_type="${EVAL_TYPE:-0}" \
  framework.gpu="${GPU:-0}" \
  framework.logdir="${LOGDIR:-outputs/rlbench2_map4d_dit_eval}" \
  rlbench.headless="${RLBENCH_HEADLESS:-true}" \
  rlbench.episode_length="${EPISODE_LENGTH:-300}" \
  rlbench.task_name=box \
  'rlbench.tasks=[bimanual_push_box]' \
  rlbench.demo_path="${RLBENCH2_TEST_DEMO_PATH}" \
  rlbench.query_freq="${QUERY_FREQ:-20}" \
  method.semantic_feature_source="${SEMANTIC_FEATURE_SOURCE:-fusion}" \
  method.map_pose_source="${MAP_POSE_SOURCE:-simulator}" \
  method.map_object_name="${MAP_OBJECT_NAME:-cube}" \
  method.debug_map_video_path="${MAP4D_DEBUG_VIDEO_PATH:-}" \
  method.debug_map_video_camera="${MAP4D_DEBUG_VIDEO_CAMERA:-front}" \
  method.debug_map_video_fps="${MAP4D_DEBUG_VIDEO_FPS:-10}" \
  cinematic_recorder.enabled="${SAVE_VIDEO:-false}" \
  cinematic_recorder.save_path="${VIDEO_DIR:-outputs/rlbench2_map4d_dit_eval/box/videos}" \
  "$@"
