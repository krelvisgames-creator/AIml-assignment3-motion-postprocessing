using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Tycoon.Demo
{
    /// <summary>
    /// [데모 전용 — Company Tycoon_Demo 복사본에만 둘 것]
    /// AI&ML 과제 3: BlazePose 모션 CSV(frame, landmark, x, y, z, visibility)를 캐릭터(인부)에 입혀 반복 재생한다.
    /// 교수님 Unity 코드(BlazeDataExtraction)와 같은 '관절 방향 리타게팅' 원리 — 부모→자식 방향만 맞추고,
    /// 비틀림은 부모 본을 따라가게(체인 방식) 해서 팔·다리 꼬임을 줄였다. 골반은 엉덩이 기준, 상체는 어깨 기준 프레임.
    /// 본 찾기: Humanoid Animator 가 있으면 그걸, 없으면 Mixamo 이름(mixamorig:*)으로 직접 찾는다.
    ///
    /// 보통은 DemoCrew 가 공사장에 자동으로 만들어 쓴다. 씬에 직접 붙여 쓸 수도 있다(csvPath 지정).
    /// 공통 키: Space 전체 재생/정지 · Backspace 전체 처음부터
    /// </summary>
    public class DemoMotionPlayer : MonoBehaviour
    {
        [Tooltip("재생할 CSV 전체 경로 (비우면 csvFolder 의 첫 *_refined.csv)")]
        public string csvPath = "";
        public string csvFolder = DemoPaths.ProcessedFolder;
        public string label = "";
        public float dataFps = 30f;
        public float speed = 1f;
        [Tooltip("0이면 캐릭터 다리 길이 ÷ 데이터 다리 길이로 자동")]
        public float positionScale = 0f;
        public bool applyRootMotion = true;
        [Tooltip("좌우 뒤집어 재생(같은 클립을 여러 명이 쓸 때 다르게 보이게) — Init 전에 정할 것")]
        public bool mirror = false;
        [Range(0f, 1f)] public float spineFollowChest = 0.5f;

        public enum ToolKind { None, Hammer, Shovel }
        [Tooltip("손에 쥘 도구 (CSV 이름에 hammer/shovel 이 있으면 자동)")]
        public ToolKind tool = ToolKind.None;
        [Tooltip("CSV 이름으로 도구를 자동으로 쥐어 줄지")]
        public bool autoTool = true;

        public static readonly List<DemoMotionPlayer> All = new List<DemoMotionPlayer>();
        public static bool Paused;

        public string ClipName { get; private set; } = "";
        public int FrameCount => _frames.Count;
        public int CurrentFrame { get; private set; }
        public Transform Hips => _hips;

        private List<Vector3[]> _frames = new List<Vector3[]>();
        private float _time;
        private Transform _root, _hips, _spine, _chest, _upperChest, _head;
        private readonly List<Limb> _limbs = new List<Limb>();
        private readonly Dictionary<Transform, Quaternion> _restRot = new Dictionary<Transform, Quaternion>();
        private readonly Dictionary<Transform, Quaternion> _restAny = new Dictionary<Transform, Quaternion>();
        private Vector3 _anchorPos;
        private Quaternion _anchorRot = Quaternion.identity;
        private float _scale = 1f, _charLeg = 0.85f;
        private bool _ready;

        private class Limb { public Transform bone; public Vector3 restDir; public int a, b; public bool useHead; }

        // Mixamo 이름 (Humanoid 아바타가 없을 때)
        private static readonly Dictionary<HumanBodyBones, string> MixamoNames = new Dictionary<HumanBodyBones, string>
        {
            { HumanBodyBones.Hips, "Hips" }, { HumanBodyBones.Spine, "Spine" }, { HumanBodyBones.Chest, "Spine1" },
            { HumanBodyBones.UpperChest, "Spine2" }, { HumanBodyBones.Head, "Head" },
            { HumanBodyBones.LeftUpperLeg, "LeftUpLeg" }, { HumanBodyBones.LeftLowerLeg, "LeftLeg" },
            { HumanBodyBones.LeftFoot, "LeftFoot" }, { HumanBodyBones.LeftToes, "LeftToeBase" },
            { HumanBodyBones.RightUpperLeg, "RightUpLeg" }, { HumanBodyBones.RightLowerLeg, "RightLeg" },
            { HumanBodyBones.RightFoot, "RightFoot" }, { HumanBodyBones.RightToes, "RightToeBase" },
            { HumanBodyBones.LeftUpperArm, "LeftArm" }, { HumanBodyBones.LeftLowerArm, "LeftForeArm" }, { HumanBodyBones.LeftHand, "LeftHand" },
            { HumanBodyBones.RightUpperArm, "RightArm" }, { HumanBodyBones.RightLowerArm, "RightForeArm" }, { HumanBodyBones.RightHand, "RightHand" },
        };
        private Animator _anim;
        private Dictionary<string, Transform> _byName;

        // =========================================================== setup
        private void OnEnable() { if (!All.Contains(this)) All.Add(this); }
        private void OnDisable() { All.Remove(this); }

        private void Start()
        {
            if (!_ready) Setup();
            if (_ready && _frames.Count == 0)
            {
                if (string.IsNullOrEmpty(csvPath) && Directory.Exists(csvFolder))
                {
                    var f = Directory.GetFiles(csvFolder, "*_refined.csv");
                    if (f.Length > 0) { System.Array.Sort(f); csvPath = f[0]; }
                }
                if (!string.IsNullOrEmpty(csvPath)) Load(csvPath, 0f);
            }
        }

        /// <summary>DemoCrew 가 만든 직후 호출: 위치·방향·클립·시작 시점.</summary>
        public void Init(string path, Vector3 anchorPos, Quaternion anchorRot, float startTime)
        {
            if (!_ready) Setup();
            _anchorPos = anchorPos; _anchorRot = anchorRot;
            Load(path, startTime);
        }

        public void MoveTo(Vector3 anchorPos, Quaternion anchorRot) { _anchorPos = anchorPos; _anchorRot = anchorRot; }

        private void Setup()
        {
            _root = transform;
            _anim = GetComponentInChildren<Animator>();
            if (_anim != null && !_anim.isHuman) _anim = null;
            if (_anim == null)
            {
                _byName = new Dictionary<string, Transform>();
                foreach (var t in GetComponentsInChildren<Transform>(true))
                {
                    string n = t.name; int c = n.LastIndexOf(':');
                    if (c >= 0) n = n.Substring(c + 1);
                    if (!_byName.ContainsKey(n)) _byName[n] = t;
                }
            }
            _hips = B(HumanBodyBones.Hips);
            if (_hips == null) { Debug.LogError($"[DemoMotionPlayer] {name}: Hips 본을 못 찾음 (Humanoid 또는 mixamorig 이름 필요)"); enabled = false; return; }
            _anchorPos = _root.position; _anchorRot = _root.rotation;
            _spine = B(HumanBodyBones.Spine); _chest = B(HumanBodyBones.Chest);
            _upperChest = B(HumanBodyBones.UpperChest); _head = B(HumanBodyBones.Head);

            AddLimb(HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, 23, 25);
            AddLimb(HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot, 25, 27);
            AddLimb(HumanBodyBones.LeftFoot, HumanBodyBones.LeftToes, 27, 31);
            AddLimb(HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, 24, 26);
            AddLimb(HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot, 26, 28);
            AddLimb(HumanBodyBones.RightFoot, HumanBodyBones.RightToes, 28, 32);
            AddLimb(HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, 11, 13);
            AddLimb(HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand, 13, 15);
            AddLimb(HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, 12, 14);
            AddLimb(HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand, 14, 16);
            if (_head != null && _head.childCount > 0)
                _limbs.Add(new Limb { bone = _head, restDir = RootDir(_head.position, _head.GetChild(0).position), useHead = true });

            foreach (var t in new[] { _hips, _spine, _chest, _upperChest, _head })
                if (t != null) _restRot[t] = Quaternion.Inverse(_root.rotation) * t.rotation;
            foreach (var l in _limbs) _restRot[l.bone] = Quaternion.Inverse(_root.rotation) * l.bone.rotation;
            SetupHands();

            var ul = B(HumanBodyBones.LeftUpperLeg); var ll = B(HumanBodyBones.LeftLowerLeg); var ft = B(HumanBodyBones.LeftFoot);
            if (ul != null && ll != null && ft != null)
                _charLeg = Vector3.Distance(ul.position, ll.position) + Vector3.Distance(ll.position, ft.position);
            SetupGround();
            var anyAnim = GetComponentInChildren<Animator>();
            if (anyAnim != null) anyAnim.enabled = false;   // Animator 가 포즈를 덮어쓰지 않게
            _ready = true;
        }

        private Transform B(HumanBodyBones b)
        {
            if (_anim != null) return _anim.GetBoneTransform(b);
            return MixamoNames.TryGetValue(b, out var n) && _byName.TryGetValue(n, out var t) ? t : null;
        }
        private Vector3 RootDir(Vector3 from, Vector3 to) => Quaternion.Inverse(_root.rotation) * (to - from).normalized;
        private void AddLimb(HumanBodyBones bone, HumanBodyBones child, int a, int b)
        {
            var t = B(bone); var c = B(child);
            if (t == null || c == null) { Debug.LogWarning($"[DemoMotionPlayer] {bone}/{child} 없음 — 건너뜀"); return; }
            _limbs.Add(new Limb { bone = t, restDir = RootDir(t.position, c.position), a = a, b = b });
        }

        // =========================================================== data
        public void Load(string path, float startTime)
        {
            if (!File.Exists(path)) { Debug.LogError($"[DemoMotionPlayer] CSV 없음: {path}"); return; }
            csvPath = path;
            _frames = ReadCsv(path);
            if (mirror) Mirror(_frames);
            ClipName = Path.GetFileNameWithoutExtension(path);
            if (tool == ToolKind.None && autoTool)
            {
                string cn = ClipName.ToLowerInvariant();
                tool = cn.Contains("shovel") ? ToolKind.Shovel : cn.Contains("hammer") ? ToolKind.Hammer : ToolKind.None;
            }
            if (string.IsNullOrEmpty(label)) label = ClipName;
            float sum = 0; int n = 0;
            for (int i = 0; i < _frames.Count; i += 5)
            {
                var p = _frames[i];
                sum += Vector3.Distance(p[23], p[25]) + Vector3.Distance(p[25], p[27]); n++;
            }
            float dataLeg = n > 0 ? sum / n : 0.76f;
            ComputeHeadBaseline();
            ComputeRearHand();
            _hasToolJoints = false;
            for (int i = 0; i < _frames.Count && !_hasToolJoints; i += 10) _hasToolJoints = _frames[i][34] != Vector3.zero;
            _scale = positionScale > 0 ? positionScale : _charLeg / Mathf.Max(0.01f, dataLeg);
            _time = startTime;
        }

        /// <summary>좌우 뒤집기: x 부호를 바꾸고 BlazePose 왼쪽 · 오른쪽 점을 맞바꿈(도구 관절 33 · 34 는 x 만).</summary>
        private static void Mirror(List<Vector3[]> frames)
        {
            int[,] pairs = { { 1, 4 }, { 2, 5 }, { 3, 6 }, { 7, 8 }, { 9, 10 }, { 11, 12 }, { 13, 14 }, { 15, 16 }, { 17, 18 }, { 19, 20 },
                             { 21, 22 }, { 23, 24 }, { 25, 26 }, { 27, 28 }, { 29, 30 }, { 31, 32 } };
            foreach (var f in frames)
            {
                for (int j = 0; j < f.Length; j++) f[j].x = -f[j].x;
                for (int k = 0; k < pairs.GetLength(0); k++)
                {
                    int a = pairs[k, 0], b = pairs[k, 1];
                    var t = f[a]; f[a] = f[b]; f[b] = t;
                }
            }
        }

        private static List<Vector3[]> ReadCsv(string path)
        {
            var frames = new List<Vector3[]>();
            var inv = CultureInfo.InvariantCulture;
            var lines = File.ReadAllLines(path);
            for (int i = 1; i < lines.Length; i++)
            {
                var v = lines[i].Split(',');
                if (v.Length < 5) continue;
                int f = int.Parse(v[0], inv), j = int.Parse(v[1], inv);
                while (frames.Count <= f) frames.Add(new Vector3[35]);   // 0~32 BlazePose + 33 자루 끝 · 34 도구 머리(p1_5 강체 도구 후처리)
                frames[f][j] = new Vector3(-float.Parse(v[2], inv), float.Parse(v[3], inv), -float.Parse(v[4], inv)); // 교수님 코드와 같은 (-x, y, -z)
            }
            return frames;
        }

        // =========================================================== playback
        private void Update()
        {
            if (!_ready) return;
            var kb = Keyboard.current;
            if (All.Count > 0 && All[0] == this && kb != null && !Tycoon.UI.Kit.UiInput.IsTyping)
            {
                if (kb.spaceKey.wasPressedThisFrame) Paused = !Paused;
                if (kb.backspaceKey.wasPressedThisFrame) foreach (var p in All) p._time = 0;
            }
            if (!Paused) _time += Time.unscaledDeltaTime * speed;   // 게임 일시정지와 상관없이 재생
        }

        private void LateUpdate()
        {
            if (!_ready || _frames.Count == 0) return;
            CurrentFrame = Mathf.FloorToInt(_time * dataFps) % _frames.Count;
            var fr = _frames[CurrentFrame];
            ApplyPose(fr);
            GroundFeet();                 // 발이 땅에 묻히거나 뜨지 않게 골반 높이 보정
            GripIK(fr);                   // 도구가 기준: 두 손을 자루의 정해진 두 지점에 IK 로 붙임
            ApplyHands(fr);
            UpdateTool(fr);
        }

        // =========================================================== retarget
        private static Quaternion Frame(Vector3 l, Vector3 r, Vector3 up)
        {
            Vector3 right = r - l; right.y = 0; right.Normalize();
            Vector3 fwd = Vector3.Cross(right, up.normalized).normalized;
            Vector3 u = Vector3.Cross(fwd, right).normalized;
            return Quaternion.LookRotation(fwd, u);
        }

        private void ApplyPose(Vector3[] p)
        {
            Vector3 hipC = (p[23] + p[24]) * 0.5f, shC = (p[11] + p[12]) * 0.5f;
            Vector3 up = shC - hipC;
            Quaternion pelvis = Frame(p[23], p[24], up);
            Quaternion chest = Frame(p[11], p[12], up);

            Vector3 hipLocal = applyRootMotion ? hipC * _scale : new Vector3(0, hipC.y * _scale, 0);
            _hips.SetPositionAndRotation(_anchorPos + _anchorRot * hipLocal, _anchorRot * pelvis * _restRot[_hips]);

            if (_spine != null) _spine.rotation = _anchorRot * Quaternion.Slerp(pelvis, chest, spineFollowChest) * _restRot[_spine];
            var top = _upperChest != null ? _upperChest : _chest;
            if (_chest != null && _chest != top) _chest.rotation = _anchorRot * Quaternion.Slerp(pelvis, chest, 0.8f) * _restRot[_chest];
            if (top != null) top.rotation = _anchorRot * chest * _restRot[top];

            foreach (var l in _limbs)
            {
                Vector3 target = l.useHead ? HeadTarget(p, chest, shC) : (p[l.b] - p[l.a]);
                if (target.sqrMagnitude < 1e-8f) continue;
                target = _anchorRot * target.normalized;
                Quaternion carry = l.bone.parent.rotation * Quaternion.Inverse(RestWorldOf(l.bone.parent));
                Vector3 dirNow = carry * (_anchorRot * l.restDir);
                Quaternion rotNow = carry * (_anchorRot * _restRot[l.bone]);
                l.bone.rotation = Quaternion.FromToRotation(dirNow, target) * rotNow;
            }
        }

        // =========================================================== feet (발 묻힘 보정)
        [Tooltip("발 · 발가락 중 가장 낮은 곳을 땅(앵커 높이)에 맞춘다")] public bool groundFeet = true;
        private Transform _lf, _rf, _lt, _rt;
        private float _footH, _toeH;   // 기본 자세에서 발목 · 발가락 본이 바닥(루트)보다 높은 양

        private void SetupGround()
        {
            _lf = B(HumanBodyBones.LeftFoot); _rf = B(HumanBodyBones.RightFoot);
            _lt = B(HumanBodyBones.LeftToes); _rt = B(HumanBodyBones.RightToes);
            float baseY = _root.position.y;
            _footH = _lf != null ? _lf.position.y - baseY : 0.08f;
            _toeH = _lt != null ? _lt.position.y - baseY : 0.02f;
        }

        /// <summary>
        /// 데이터 다리 비율과 캐릭터 다리 비율이 달라 방향만 옮기면 발이 바닥 아래로 들어가거나 뜬다.
        /// 매 프레임 두 발(발목 · 발가락)의 '밑바닥' 중 가장 낮은 곳을 땅 높이에 맞춰 골반을 올리고 내린다(작업 동작은 늘 한 발 이상 디딤).
        /// </summary>
        private void GroundFeet()
        {
            if (!groundFeet || _lf == null || _rf == null) return;
            float bottom = Mathf.Min(_lf.position.y - _footH, _rf.position.y - _footH);
            if (_lt != null && _rt != null) bottom = Mathf.Min(bottom, Mathf.Min(_lt.position.y - _toeH, _rt.position.y - _toeH));
            float shift = _anchorPos.y - bottom;
            _hips.position += Vector3.up * shift;
        }

        // =========================================================== grip IK (도구 기준 손 고정)
        [Tooltip("도구를 기준으로 두 손을 자루에 IK 로 붙임(끄면 팔은 데이터 방향 그대로)")] public bool gripIK = true;
        [Tooltip("손목 → 손바닥(자루를 쥐는 곳) 거리(m)")] public float palmOffset = 0.07f;

        /// <summary>
        /// 망치 · 삽은 자루의 정해진 두 지점을 쥔다. 도구 자세(뒷손 위치 + 자루 방향 + 두 손 간격)를 데이터에서 가져와 캐릭터 어깨 기준으로 놓고,
        /// 두 손목을 그 두 지점에 2-bone IK(위팔 · 아래팔 길이 그대로, 지금 팔꿈치 굽힘 방향 유지)로 붙인다 → 도구가 손에서 꼬이지 않음.
        /// </summary>
        private void GripIK(Vector3[] p)
        {
            if (!gripIK || tool == ToolKind.None || !_hasToolJoints || p[34] == p[33]) return;
            var lu = B(HumanBodyBones.LeftUpperArm); var ru = B(HumanBodyBones.RightUpperArm);
            var lf = B(HumanBodyBones.LeftLowerArm); var rfo = B(HumanBodyBones.RightLowerArm);
            var lh = B(HumanBodyBones.LeftHand); var rh = B(HumanBodyBones.RightHand);
            if (lu == null || ru == null || lf == null || rfo == null || lh == null || rh == null) return;

            Vector3 shData = (p[11] + p[12]) * 0.5f, shMesh = (lu.position + ru.position) * 0.5f;
            int rearI = _leftRear ? 15 : 16, leadI = _leftRear ? 16 : 15;
            Vector3 axis = (_anchorRot * (p[34] - p[33])).normalized;
            float gap = Vector3.Distance(p[rearI], p[leadI]) * _scale;
            // 뒷손 쥠점 = 캐릭터 어깨 + 데이터에서 어깨 → 뒷손(캐릭터 크기로)
            Vector3 rearGrip = shMesh + _anchorRot * ((p[rearI] - shData) * _scale);
            Vector3 leadGrip = rearGrip + axis * gap;
            _gripRear = rearGrip; _gripAxis = axis; _gripValid = true;

            Transform ru_ = _leftRear ? lu : ru, rf_ = _leftRear ? lf : rfo, rh_ = _leftRear ? lh : rh;
            Transform lu_ = _leftRear ? ru : lu, lf_ = _leftRear ? rfo : lf, lh_ = _leftRear ? rh : lh;
            TwoBoneIK(ru_, rf_, rh_, rearGrip - (rh_.position - rf_.position).normalized * palmOffset);
            TwoBoneIK(lu_, lf_, lh_, leadGrip - (lh_.position - lf_.position).normalized * palmOffset);
        }

        private Vector3 _gripRear, _gripAxis;
        private bool _gripValid;

        /// <summary>해석적 2-bone IK (Holden 방식): 위팔 · 아래팔 길이 고정, 굽힘 축 = 지금 팔 평면.</summary>
        private static void TwoBoneIK(Transform a, Transform b, Transform c, Vector3 t)
        {
            const float eps = 0.001f;
            Vector3 pa = a.position, pb = b.position, pc = c.position;
            float lab = (pb - pa).magnitude, lcb = (pb - pc).magnitude;
            float lat = Mathf.Clamp((t - pa).magnitude, eps, lab + lcb - eps);
            float ac_ab_0 = Mathf.Acos(Mathf.Clamp(Vector3.Dot((pc - pa).normalized, (pb - pa).normalized), -1f, 1f));
            float ba_bc_0 = Mathf.Acos(Mathf.Clamp(Vector3.Dot((pa - pb).normalized, (pc - pb).normalized), -1f, 1f));
            float ac_at_0 = Mathf.Acos(Mathf.Clamp(Vector3.Dot((pc - pa).normalized, (t - pa).normalized), -1f, 1f));
            float ac_ab_1 = Mathf.Acos(Mathf.Clamp((lcb * lcb - lab * lab - lat * lat) / (-2f * lab * lat), -1f, 1f));
            float ba_bc_1 = Mathf.Acos(Mathf.Clamp((lat * lat - lab * lab - lcb * lcb) / (-2f * lab * lcb), -1f, 1f));
            Vector3 axis0 = Vector3.Cross(pc - pa, pb - pa);
            if (axis0.sqrMagnitude < 1e-8f) axis0 = Vector3.Cross(pc - pa, Vector3.up);
            axis0.Normalize();
            Vector3 axis1 = Vector3.Cross(pc - pa, t - pa);
            Quaternion r2 = axis1.sqrMagnitude > 1e-10f
                ? Quaternion.AngleAxis(ac_at_0 * Mathf.Rad2Deg, Quaternion.Inverse(a.rotation) * axis1.normalized)
                : Quaternion.identity;
            Quaternion r0 = Quaternion.AngleAxis((ac_ab_1 - ac_ab_0) * Mathf.Rad2Deg, Quaternion.Inverse(a.rotation) * axis0);
            Quaternion r1 = Quaternion.AngleAxis((ba_bc_1 - ba_bc_0) * Mathf.Rad2Deg, Quaternion.Inverse(b.rotation) * axis0);
            a.localRotation = a.localRotation * (r0 * r2);
            b.localRotation = b.localRotation * r1;
        }

        // =========================================================== hands (손 점으로 손목 회전)
        [Tooltip("손 방향을 데이터 손 점(검지 · 새끼)으로 맞출 정도(0 = 팔뚝 따라감)")] [Range(0, 1)] public float handFollow = 0.85f;
        [Tooltip("리그 기본 자세에서 손바닥 방향 보정(도) — 손등이 뒤집혀 보이면 180")] public float handRollOffset = 0f;
        private Transform _lh, _rh, _lfa, _rfa;
        private Vector3 _lhRestDir, _rhRestDir;

        private void SetupHands()
        {
            _lh = B(HumanBodyBones.LeftHand); _rh = B(HumanBodyBones.RightHand);
            _lfa = B(HumanBodyBones.LeftLowerArm); _rfa = B(HumanBodyBones.RightLowerArm);
            if (_lh != null && _lfa != null) { _lhRestDir = RootDir(_lfa.position, _lh.position); _restRot[_lh] = Quaternion.Inverse(_root.rotation) * _lh.rotation; }
            if (_rh != null && _rfa != null) { _rhRestDir = RootDir(_rfa.position, _rh.position); _restRot[_rh] = Quaternion.Inverse(_root.rotation) * _rh.rotation; }
        }

        /// <summary>
        /// 손 방향 = 손목 → (검지 + 새끼)/2, 손 옆 = 새끼 → 검지(엄지 쪽). 리그 기본 자세(T/A, 손바닥 아래)에선
        /// 손 방향 = 팔 연장, 엄지 쪽 = 캐릭터 앞(+Z). 두 틀 사이 회전을 기본 회전에 얹는다(좌우 대칭이라 손마다 따로 부호가 필요 없음).
        /// </summary>
        private void ApplyHands(Vector3[] p)
        {
            if (handFollow <= 0f) return;
            ApplyHand(_lh, _lhRestDir, p[15], p[19], p[17]);
            ApplyHand(_rh, _rhRestDir, p[16], p[20], p[18]);
        }

        private void ApplyHand(Transform hand, Vector3 restDir, Vector3 wr, Vector3 index, Vector3 pinky)
        {
            if (hand == null || !_restRot.ContainsKey(hand)) return;
            Vector3 dir = (index + pinky) * 0.5f - wr, side = index - pinky;
            if (dir.sqrMagnitude < 1e-6f || side.sqrMagnitude < 1e-6f) return;
            Quaternion target = Quaternion.LookRotation(_anchorRot * dir.normalized, _anchorRot * side.normalized);
            Quaternion rest = Quaternion.LookRotation(_anchorRot * restDir, _anchorRot * (Quaternion.AngleAxis(handRollOffset, restDir) * Vector3.forward));
            Quaternion want = target * Quaternion.Inverse(rest) * (_anchorRot * _restRot[hand]);
            hand.rotation = Quaternion.Slerp(hand.rotation, want, handFollow);
        }

        // =========================================================== head (목 꺾임 보정)
        [Tooltip("머리가 기본 자세에서 벗어날 수 있는 최대 각도")] public float headMaxAngle = 25f;
        [Tooltip("데이터 머리 움직임을 얼마나 따를지(0 = 가슴 따라 고정)")] [Range(0, 1)] public float headFollow = 0.5f;
        private Vector3 _headMean = Vector3.up;     // 이 클립의 평소 머리 방향(가슴 기준)
        private Quaternion _headSmooth = Quaternion.identity;
        private bool _headInit;

        /// <summary>
        /// BlazePose 귀 두 점은 보통 어깨보다 앞에 있어서(귀-어깨 벡터가 늘 20~30° 앞으로 기움) 그대로 쓰면 목이 늘 꺾여 보이고,
        /// 등을 돌리면 귀가 튀어서 목이 휙휙 돌아간다. 클립 평균 방향을 '똑바로'로 보고 그 차이만, 줄여서 · 각도를 막고 · 부드럽게 쓴다.
        /// </summary>
        private void ComputeHeadBaseline()
        {
            Vector3 acc = Vector3.zero;
            for (int i = 0; i < _frames.Count; i += 3)
            {
                var p = _frames[i];
                Vector3 shC = (p[11] + p[12]) * 0.5f, hipC = (p[23] + p[24]) * 0.5f;
                Vector3 d = (p[7] + p[8]) * 0.5f - shC;
                if (d.sqrMagnitude < 1e-8f) continue;
                acc += Quaternion.Inverse(Frame(p[11], p[12], shC - hipC)) * d.normalized;
            }
            _headMean = acc.sqrMagnitude > 1e-8f ? acc.normalized : Vector3.up;
            _headInit = false;
        }

        /// <summary>머리 목표 방향(데이터 좌표) — 가슴 '위'를 기준으로, 평소 방향에서 벗어난 만큼만 줄여서.</summary>
        private Vector3 HeadTarget(Vector3[] p, Quaternion chest, Vector3 shC)
        {
            Vector3 d = (p[7] + p[8]) * 0.5f - shC;
            Quaternion delta = Quaternion.identity;
            if (d.sqrMagnitude > 1e-8f)
            {
                Vector3 local = Quaternion.Inverse(chest) * d.normalized;
                delta = Quaternion.FromToRotation(_headMean, local);
                delta.ToAngleAxis(out float ang, out Vector3 ax);
                if (ang > 180f) ang -= 360f;
                ang = Mathf.Clamp(ang * headFollow, -headMaxAngle, headMaxAngle);
                delta = Quaternion.AngleAxis(ang, ax);
            }
            _headSmooth = _headInit ? Quaternion.Slerp(_headSmooth, delta, 0.25f) : delta;
            _headInit = true;
            return chest * (_headSmooth * Vector3.up);
        }

        // =========================================================== tool (해머 · 삽)
        private bool _hasToolJoints;
        private bool _leftRear;   // 클립 전체에서 손잡이 끝(뒷손)을 쥐는 손 — 프레임마다 바꾸면 도구가 두 손 사이를 오가며 튄다

        private void ComputeRearHand()
        {
            int leftCloser = 0, total = 0;
            for (int i = 0; i < _frames.Count; i += 3)
            {
                var p = _frames[i];
                Vector3 sh = (p[11] + p[12]) * 0.5f;
                float dl = (p[15] - sh).sqrMagnitude, dr = (p[16] - sh).sqrMagnitude;
                if (Mathf.Abs(dl - dr) < 0.0025f) continue;
                total++; if (dl < dr) leftCloser++;
            }
            _leftRear = total == 0 || leftCloser * 2 >= total;
            // 도구 관절이 있으면 자루 끝(33)에 가까운 손이 뒷손 — 파이썬 후처리와 같은 손
            int lc = 0, tc = 0;
            for (int i = 0; i < _frames.Count; i += 3)
            {
                var p = _frames[i]; if (p[34] == Vector3.zero) continue;
                tc++; if ((p[15] - p[33]).sqrMagnitude < (p[16] - p[33]).sqrMagnitude) lc++;
            }
            if (tc > 0) _leftRear = lc * 2 >= tc;
        }

        private Transform _tool, _lHand, _rHand, _lFore, _rFore;
        private bool _toolIsModel;
        private Vector3 _lastAxis = Vector3.up;
        private static Material _woodMat, _steelMat;

        /// <summary>
        /// 두 손을 잇는 선 = 손잡이 축. 머리(쇠뭉치 · 삽날)는 어깨에서 먼 쪽 끝.
        /// 손이 거의 붙으면(한 손 쥠) 직전 축을 유지. 데이터에 도구 좌표가 없으니 손 위치에서 추정한다.
        /// </summary>
        private void UpdateTool(Vector3[] p)
        {
            if (tool == ToolKind.None) return;
            if (_tool == null) BuildTool();
            if (_lHand == null || _rHand == null) return;

            Vector3 gl = Grip(_lHand, _lFore), gr = Grip(_rHand, _rFore);
            Vector3 shoulders = (B(HumanBodyBones.LeftUpperArm).position + B(HumanBodyBones.RightUpperArm).position) * 0.5f;
            // 뒷손(어깨에 가까운 손) = 손잡이 끝 쪽, 앞손 = 머리 쪽
            Vector3 rear = _leftRear ? gl : gr, lead = _leftRear ? gr : gl;
            Vector3 mid = (gl + gr) * 0.5f;
            float sep = Vector3.Distance(gl, gr);
            // 두 손이 붙어 있으면 손 사이 방향은 잡음뿐 → 팔 연장선(어깨 → 손)으로. 손이 벌어질수록 손 사이 방향을 믿는다.
            Vector3 armDir = (mid - shoulders).sqrMagnitude > 1e-4f ? (mid - shoulders).normalized : _lastAxis;
            Vector3 handDir = sep > 0.02f ? (lead - rear) / sep : armDir;
            float w = Mathf.Clamp01((sep - 0.08f) / 0.25f);
            Vector3 axis = Vector3.Slerp(armDir, handDir, w).normalized;
            if (_hasToolJoints && p[34] != p[33])
                axis = (_anchorRot * (p[34] - p[33])).normalized;   // 강체 제약 · 평활된 데이터 자루 방향(이미 부드러움)
            _lastAxis = Vector3.Slerp(_lastAxis, axis, _hasToolJoints ? 1f : 0.85f).normalized;   // 지연이 크면 손과 따로 노는 것처럼 보여서 거의 바로 따라감
            axis = _lastAxis;

            if (_gripValid) { rear = _gripRear; axis = _gripAxis; _gripValid = false; }   // IK 가 손을 붙인 그 자루 선 그대로
            Vector3 fwd = _hips.forward; fwd -= Vector3.Dot(fwd, axis) * axis;
            if (fwd.sqrMagnitude < 1e-4f) fwd = Vector3.Cross(axis, Vector3.right);
            float len = tool == ToolKind.Hammer ? 0.85f : 1.05f;
            // 손잡이 끝(꽁무니)은 뒷손에서 0.1m 뒤 — 손잡이가 늘 뒷손을 지나고, 머리는 앞손 너머
            Vector3 back = rear - axis * 0.10f;
            _tool.SetPositionAndRotation(back, Quaternion.LookRotation(fwd.normalized, axis));
            if (_toolIsModel) return;   // 모델: 원점 = 자루 끝, +Y = 자루, +Z = 머리 긴 방향(Blender 에서 그렇게 만듦)
            _tool.localScale = Vector3.one;
            _tool.GetChild(0).localPosition = new Vector3(0, len * 0.5f, 0);
            _tool.GetChild(0).localScale = new Vector3(0.035f, len * 0.5f, 0.035f);
            _tool.GetChild(1).localPosition = new Vector3(0, tool == ToolKind.Shovel ? len + 0.15f : len, 0);
        }

        /// <summary>도구 바꾸기(데모 T 키) — 지금 도구를 지우고 다음 프레임에 새로 만든다.</summary>
        public void SetTool(ToolKind k)
        {
            tool = k;
            if (_tool != null) { Destroy(_tool.gameObject); _tool = null; }
        }

        private static Vector3 Grip(Transform hand, Transform fore)
        {
            Vector3 dir = fore != null ? (hand.position - fore.position).normalized : Vector3.down;
            return hand.position + dir * 0.08f;   // 손목에서 손바닥 가운데쯤
        }

        private void BuildTool()
        {
            _lHand = B(HumanBodyBones.LeftHand); _rHand = B(HumanBodyBones.RightHand);
            _lFore = B(HumanBodyBones.LeftLowerArm); _rFore = B(HumanBodyBones.RightLowerArm);
            if (_woodMat == null)
            {
                _woodMat = MakeMat(new Color(0.62f, 0.45f, 0.28f), 0.25f, 0f);   // 나뭇결(회색) × 나무색
                var grain = Resources.Load<Texture2D>("DemoWorker/T_Wood_BC");    // 게임 나무(트리 베이커) 껍질 텍스처 복사본
                if (grain != null) { _woodMat.SetTexture("_BaseMap", grain); _woodMat.SetTexture("_MainTex", grain); }
            }
            if (_steelMat == null) _steelMat = MakeMat(new Color(0.45f, 0.46f, 0.48f), 0.55f, 0.9f);

            var model = Resources.Load<GameObject>(tool == ToolKind.Hammer ? "DemoWorker/SM_Tool_Hammer" : "DemoWorker/SM_Tool_Shovel");
            if (model != null)
            {
                _tool = new GameObject(tool + "Tool").transform;
                _tool.SetParent(transform, false);
                var go = Instantiate(model, _tool, false);
                go.transform.localPosition = Vector3.zero; go.transform.localRotation = Quaternion.identity;
                foreach (var col in go.GetComponentsInChildren<Collider>()) Destroy(col);
                foreach (var r in go.GetComponentsInChildren<Renderer>())
                {
                    var mats = r.sharedMaterials;
                    for (int i = 0; i < mats.Length; i++)
                    {
                        string n = mats[i] != null ? mats[i].name : "";
                        mats[i] = n.Contains("Steel") ? _steelMat : n.Contains("Wood") ? _woodMat : (i == 0 ? _woodMat : _steelMat);
                    }
                    r.sharedMaterials = mats;
                }
                _toolIsModel = true;
                return;
            }

            _tool = new GameObject(tool + "Tool").transform;
            _tool.SetParent(transform, false);
            var handle = Prim(PrimitiveType.Cylinder, _tool, _woodMat);
            Transform head;
            if (tool == ToolKind.Hammer)
            {
                head = Prim(PrimitiveType.Cube, _tool, _steelMat);
                head.localScale = new Vector3(0.11f, 0.11f, 0.24f);       // 쇠뭉치: 손잡이에 수직(앞뒤로 긴)
            }
            else
            {
                head = Prim(PrimitiveType.Cube, _tool, _steelMat);
                head.localScale = new Vector3(0.28f, 0.34f, 0.025f);      // 삽날: 넓적한 판
                head.localRotation = Quaternion.Euler(20f, 0f, 0f);       // 손잡이에서 살짝 꺾인 날
                var grip = Prim(PrimitiveType.Cube, _tool, _woodMat);      // 손잡이 끝 가로 손잡이(D그립 대신)
                grip.localScale = new Vector3(0.16f, 0.03f, 0.03f);
                grip.localPosition = new Vector3(0f, -0.02f, 0f);
            }
            handle.SetSiblingIndex(0); head.SetSiblingIndex(1);
        }

        private static Transform Prim(PrimitiveType t, Transform parent, Material m)
        {
            var go = GameObject.CreatePrimitive(t);
            var col = go.GetComponent<Collider>(); if (col != null) Destroy(col);   // 클릭 · 카메라에 안 걸리게
            go.GetComponent<Renderer>().sharedMaterial = m;
            go.transform.SetParent(parent, false);
            return go.transform;
        }

        private static Material MakeMat(Color c, float smooth, float metal)
        {
            var sh = Shader.Find("Universal Render Pipeline/Lit");
            if (sh == null) sh = Shader.Find("Standard");
            var m = new Material(sh);
            m.SetColor("_BaseColor", c); m.SetColor("_Color", c);
            m.SetFloat("_Smoothness", smooth); m.SetFloat("_Glossiness", smooth); m.SetFloat("_Metallic", metal);
            return m;
        }

        /// <summary>부모 본의 '기본 자세 월드 회전'(지금 앵커 기준). 안 움직이는 중간 본(어깨뼈 · 목 등)은 처음 로컬 회전을 기억.</summary>
        private Quaternion RestWorldOf(Transform t)
        {
            if (t == null || t == _root) return _anchorRot;
            if (_restRot.TryGetValue(t, out var r)) return _anchorRot * r;
            if (!_restAny.TryGetValue(t, out var a)) { a = t.localRotation; _restAny[t] = a; }
            return RestWorldOf(t.parent) * a;
        }
    }

    public static class DemoPaths
    {
        public const string ProcessedFolder = @"C:\Users\junha\OneDrive\바탕 화면\인머러\AI_ML_UnityProject\AI_ML_Python_Final\final_project\data\processed";
    }
}
