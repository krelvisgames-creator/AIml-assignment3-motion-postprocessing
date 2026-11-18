using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Tycoon.Demo
{
    /// <summary>
    /// [데모 전용 — Company Tycoon_Demo 복사본에만 둘 것]
    /// 시티즈 스카이라인 · 심즈 같은 풀 3D 카메라. 플레이하면 기본으로 켜지고, 쿼터뷰 조작기(QuarterViewCameraController)는 꺼둔다.
    /// 땅 위의 '중심점(pivot)'을 기준으로 돌고 · 기울이고 · 당긴다. 씬에 둘 필요 없음(자동 생성).
    ///
    ///   WASD / 방향키   이동(보는 방향 기준, 멀수록 빠름)      Shift   빠르게
    ///   Q / E           좌우 회전                              R / F, 휠   줌 인 / 아웃
    ///   휠 클릭 드래그   중심점(땅) 기준 돌기 + 기울이기(바닥 높이 ~ 수직 탑뷰)
    ///   우클릭 드래그    카메라 위치는 그대로, 고개만 돌려 둘러보기(하늘까지)
    ///   F3   인부 따라가기 켜기/끄기 (Tab 으로 다음 인부 — DemoCrew)
    ///   F2   원래 쿼터뷰로 돌아가기 / 다시 3D 카메라
    ///   H    게임 UI 숨기기/보이기
    /// </summary>
    public class DemoFreeCamera : MonoBehaviour
    {
        public float rotateSpeed = 0.25f;     // 마우스 드래그 감도 (도/픽셀)
        public float keyRotateSpeed = 90f;    // Q/E 도/초
        public float minPitch = 3f, maxPitch = 89f;
        public float minLookPitch = -60f;     // 우클릭 둘러보기는 위(하늘)까지
        public float minDistance = 1.5f, maxDistance = 1500f;
        public float panSpeedFactor = 1.2f;   // 거리 대비 이동 속도
        public float smooth = 12f;

        private bool _on, _follow, _uiHidden, _initialized;
        private Behaviour _quarterView;
        private Camera _cam;
        private Vector3 _pivot, _pivotGoal;
        private float _yaw, _pitch = 35f, _dist = 25f;
        private float _yawGoal, _pitchGoal = 35f, _distGoal = 25f;
        private readonly List<Canvas> _hidden = new List<Canvas>();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Boot()
        {
            if (FindFirstObjectByType<DemoFreeCamera>() != null) return;
            var go = new GameObject("[DemoFreeCamera]");
            DontDestroyOnLoad(go);
            go.AddComponent<DemoFreeCamera>();
        }

        private void Update()
        {
            // 플레이 시작 후 카메라 · 쿼터뷰 조작기가 준비되면 자동으로 켬 (기본 = 3D 카메라)
            if (!_initialized && Camera.main != null) { _initialized = true; SetOn(true); }

            var kb = Keyboard.current;
            if (kb == null || Tycoon.UI.Kit.UiInput.IsTyping) return;
            if (kb.f2Key.wasPressedThisFrame) SetOn(!_on);
            if (kb.f3Key.wasPressedThisFrame) { if (!_on) SetOn(true); _follow = !_follow; }
            if (kb.hKey.wasPressedThisFrame) ToggleUi();
        }

        private void LateUpdate()
        {
            if (!_on) return;
            if (_cam == null) { _cam = Camera.main; if (_cam == null) return; }
            if (_quarterView != null && _quarterView.enabled) _quarterView.enabled = false;   // 게임이 다시 켜도 데모 중엔 끔
            var kb = Keyboard.current; var ms = Mouse.current;
            if (kb == null || ms == null) return;
            float dt = Time.unscaledDeltaTime;
            bool typing = Tycoon.UI.Kit.UiInput.IsTyping;

            // ---- 휠 클릭 드래그: 중심점(땅) 기준으로 돌기 + 기울이기 (시티즈식)
            Vector2 d = ms.delta.ReadValue();
            if (ms.middleButton.wasPressedThisFrame) RepivotToGround();
            // (인부 따라가기 중엔 우클릭도 인부 주위 돌기)
            if (ms.middleButton.isPressed || (_follow && ms.rightButton.isPressed))
            {
                _yawGoal += d.x * rotateSpeed;
                _pitchGoal = Mathf.Clamp(_pitchGoal - d.y * rotateSpeed, minPitch, maxPitch);
            }
            // ---- 우클릭 드래그: 카메라 위치는 그대로, 고개만 돌려 둘러보기 (위·아래·하늘까지)
            else if (ms.rightButton.isPressed)
            {
                Vector3 camPos = _cam.transform.position;
                _yaw = _yawGoal = _yawGoal + d.x * rotateSpeed;
                _pitch = _pitchGoal = Mathf.Clamp(_pitchGoal - d.y * rotateSpeed, minLookPitch, maxPitch);
                var look = Quaternion.Euler(_pitch, _yaw, 0f);
                _pivot = _pivotGoal = camPos + look * Vector3.forward * _dist;   // 중심점을 시선 방향으로 옮겨서 카메라는 제자리
                _distGoal = _dist;
            }
            if (!typing)
            {
                if (kb.qKey.isPressed) _yawGoal += keyRotateSpeed * dt;
                if (kb.eKey.isPressed) _yawGoal -= keyRotateSpeed * dt;
            }

            // ---- 줌 (휠 · R/F) — 거리에 비례해서 당김
            float scroll = ms.scroll.ReadValue().y;
            if (scroll != 0) _distGoal *= scroll > 0 ? 0.88f : 1.0f / 0.88f;
            if (!typing)
            {
                if (kb.rKey.isPressed) _distGoal *= 1f - 1.5f * dt;
                if (kb.fKey.isPressed) _distGoal *= 1f + 1.5f * dt;
            }
            _distGoal = Mathf.Clamp(_distGoal, minDistance, maxDistance);

            // ---- 이동 (WASD · 방향키) — 보는 방향 기준 수평 이동
            Transform target = _follow ? FindTarget() : null;
            if (target != null)
            {
                _pivotGoal = target.position;
            }
            else if (!typing)
            {
                Vector3 fwd = Quaternion.Euler(0, _yaw, 0) * Vector3.forward;
                Vector3 right = Quaternion.Euler(0, _yaw, 0) * Vector3.right;
                Vector3 mv = Vector3.zero;
                if (kb.wKey.isPressed || kb.upArrowKey.isPressed) mv += fwd;
                if (kb.sKey.isPressed || kb.downArrowKey.isPressed) mv -= fwd;
                if (kb.dKey.isPressed || kb.rightArrowKey.isPressed) mv += right;
                if (kb.aKey.isPressed || kb.leftArrowKey.isPressed) mv -= right;
                float sp = Mathf.Max(3f, _dist * panSpeedFactor) * (kb.leftShiftKey.isPressed ? 3f : 1f);
                _pivotGoal += mv * sp * dt;
            }

            // ---- 부드럽게 따라가기
            float k = 1f - Mathf.Exp(-smooth * dt);
            _yaw = Mathf.LerpAngle(_yaw, _yawGoal, k);
            _pitch = Mathf.Lerp(_pitch, _pitchGoal, k);
            _dist = Mathf.Lerp(_dist, _distGoal, k);
            _pivot = Vector3.Lerp(_pivot, _pivotGoal, target != null ? 1f - Mathf.Exp(-20f * dt) : k);

            var rot = Quaternion.Euler(_pitch, _yaw, 0f);
            Vector3 pos = _pivot - rot * Vector3.forward * _dist;
            float minY = GroundY(pos) + 0.3f;            // 땅 밑으로 안 들어가게
            if (pos.y < minY) pos.y = minY;
            _cam.transform.SetPositionAndRotation(pos, Quaternion.LookRotation(_pivot - pos));
        }

        /// <summary>지금 시선이 땅에 닿는 점을 새 중심점으로 (카메라 위치는 유지).</summary>
        private void RepivotToGround()
        {
            var t = _cam.transform;
            float gy = GroundY(t.position);
            Vector3 f = t.forward;
            if (f.y > -0.05f) return;                          // 하늘을 보고 있으면 그대로
            float dist = Mathf.Clamp((t.position.y - gy) / -f.y, minDistance, maxDistance);
            _pivot = _pivotGoal = t.position + f * dist;
            _dist = _distGoal = dist;
            _pitch = _pitchGoal = Mathf.Clamp(_pitch, minPitch, maxPitch);
        }

        private static float GroundY(Vector3 p)
        {
            var g = Tycoon.Grid.GridManager.Instance;
            return g != null ? g.GroundSurfaceY : 0f;
        }

        private void SetOn(bool on)
        {
            _cam = Camera.main;
            if (_cam == null) return;
            if (_quarterView == null)
                _quarterView = _cam.GetComponentInParent<Tycoon.CameraSystem.QuarterViewCameraController>();
            if (_quarterView == null)
                _quarterView = FindFirstObjectByType<Tycoon.CameraSystem.QuarterViewCameraController>();
            if (_quarterView != null) _quarterView.enabled = !on;
            _on = on;
            if (on)
            {
                // 지금 화면이 보던 땅 지점을 중심점으로 이어받기
                var t = _cam.transform;
                float gy = GroundY(t.position);
                Vector3 f = t.forward;
                float dist = f.y < -0.05f ? (t.position.y - gy) / -f.y : 30f;
                _pivot = _pivotGoal = t.position + f * Mathf.Clamp(dist, minDistance, maxDistance);
                _pivot.y = _pivotGoal.y = gy;
                _dist = _distGoal = Mathf.Clamp(Vector3.Distance(t.position, _pivot), minDistance, maxDistance);
                var e = Quaternion.LookRotation(_pivot - t.position).eulerAngles;
                _yaw = _yawGoal = e.y;
                _pitch = _pitchGoal = Mathf.Clamp(e.x > 180f ? e.x - 360f : e.x, minPitch, maxPitch);
            }
            else _follow = false;
            Debug.Log($"[DemoFreeCamera] 3D camera {(on ? "ON" : "OFF (quarter view)")}");
        }

        private Transform FindTarget()
        {
            if (DemoCrew.Instance != null && DemoCrew.Instance.FollowTarget != null) return DemoCrew.Instance.FollowTarget;
            var p = FindFirstObjectByType<DemoMotionPlayer>();
            if (p == null) return null;
            return p.Hips != null ? p.Hips : p.transform;
        }

        /// <summary>H 로 UI 를 숨긴 상태 — 데모 이름표 · 안내 글(DemoCrew.OnGUI)도 같이 숨긴다.</summary>
        public static bool UiHidden { get; private set; }

        private void ToggleUi()
        {
            _uiHidden = !_uiHidden;
            UiHidden = _uiHidden;
            if (_uiHidden)
            {
                _hidden.Clear();
                foreach (var c in FindObjectsByType<Canvas>(FindObjectsSortMode.None))
                    if (c.enabled && c.isRootCanvas) { c.enabled = false; _hidden.Add(c); }
            }
            else
            {
                foreach (var c in _hidden) if (c != null) c.enabled = true;
                _hidden.Clear();
            }
        }
    }
}
