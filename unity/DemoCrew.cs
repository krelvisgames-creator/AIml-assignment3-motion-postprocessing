using System.Collections.Generic;
using System.IO;
using System.Reflection;
using UnityEngine;
using UnityEngine.InputSystem;
using Tycoon.Core;
using Tycoon.Grid;
using Tycoon.Visualization;

namespace Tycoon.Demo
{
    /// <summary>
    /// [데모 전용 — Company Tycoon_Demo 복사본에만 둘 것]
    /// 도시(도로망) 한가운데 직선 도로 한 구간을 '도로 공사 중'으로 막고, 그 안에서 인부 5명이 각자 다른 작업 클립(해머 · 삽질)을 반복한다.
    ///   · 공사 구간 = 도시 중심에 가장 가까운 직선 도로 타일(앞뒤로 이어진 타일까지). 소품은 게임 아트만: 물통 · 고깔 줄 + 모서리 화분.
    ///   · 차는 이 구간에 안 들어온다: 구간(+ 앞뒤 여유) 안에 들어선 차는 게임 자체의 사라지기(TrafficAgent.BeginFadeOut)로 지운다.
    ///     — 게임 코드는 안 고치고 데모에서 리플렉션으로 부름(스폰 수 카운트도 정상으로 줄어듦).
    ///   · 인부는 제자리 모드(골반 수평 이동 끔).
    ///
    ///   G    공사 구간 · 인부 다시 배치     Tab  카메라가 따라갈 인부 바꾸기(F3 따라가기)
    ///   L    이름표 켜기/끄기              Space 전체 재생/정지 · Backspace 처음부터
    /// </summary>
    public class DemoCrew : MonoBehaviour
    {
        public static DemoCrew Instance { get; private set; }
        public int crewSize = 5;
        [Tooltip("_work.csv = 작업 구간만(p1_5_make_work_clips.py) · 없으면 _refined.csv")]
        public string suffix = "_work.csv";
        public string fallbackSuffix = "_refined.csv";
        [Tooltip("공사 구간 앞뒤로 차를 지울 여유(m)")]
        public float trafficMargin = 10f;
        [Tooltip("소품을 도로 노면보다 띄울 높이(m) — 깜빡임 방지")]
        public float propLift = 0.03f;
        [Tooltip("망치 인부가 쓸 클립(가장 정확했던 것)")] public string hammerClip = "hammer__s8LRGALu_h8";
        [Tooltip("삽 인부가 쓸 클립")] public string shovelClip = "shovel__HmyvWOXlSt0";

        private readonly List<DemoMotionPlayer> _crew = new List<DemoMotionPlayer>();
        private int _follow;
        private bool _labels = true;
        private float _nextCheck, _nextTraffic;
        private Material _mat;
        private GameObject _prefab;
        private int _placedForBlocks = -1;

        // 공사 구간 (월드)
        private bool _haveSite;
        private Vector3 _siteC, _along, _across;
        private float _halfLen, _halfWid;
        private Transform _props;

        public Transform FollowTarget => _crew.Count > 0 && _crew[_follow % _crew.Count] != null ? _crew[_follow % _crew.Count].Hips : null;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Boot()
        {
            if (Instance != null) return;
            var go = new GameObject("[DemoCrew]");
            DontDestroyOnLoad(go);
            Instance = go.AddComponent<DemoCrew>();
        }

        private void Update()
        {
            if (Time.unscaledTime >= _nextCheck)
            {
                _nextCheck = Time.unscaledTime + 0.5f;
                var roads = RoadManager.Instance;
                if (GridManager.Instance != null && roads != null)
                {
                    int blocks = 0; foreach (var _ in roads.GetAllBlockOrigins()) blocks++;
                    if (blocks > 0 && blocks != _placedForBlocks) Place();   // 새 게임 · 불러오기로 도시가 바뀌면 다시
                }
            }
            if (_haveSite && Time.unscaledTime >= _nextTraffic) { _nextTraffic = Time.unscaledTime + 0.15f; BlockTraffic(); }

            var kb = Keyboard.current;
            if (kb == null || Tycoon.UI.Kit.UiInput.IsTyping) return;
            if (kb.gKey.wasPressedThisFrame) { _placedForBlocks = -1; Place(); }
            if (kb.tabKey.wasPressedThisFrame && _crew.Count > 0) _follow = (_follow + 1) % _crew.Count;
            if (kb.lKey.wasPressedThisFrame) _labels = !_labels;
        }

        // ------------------------------------------------------------- 차 막기
        private static MethodInfo _fadeOut;
        private static FieldInfo _stateField;

        /// <summary>공사 구간(+ 앞뒤 여유) 안의 차를 게임의 사라지기 연출로 지운다.</summary>
        private void BlockTraffic()
        {
            if (_fadeOut == null)
            {
                var t = typeof(TrafficAgent);
                _fadeOut = t.GetMethod("BeginFadeOut", BindingFlags.Instance | BindingFlags.NonPublic);
                _stateField = t.GetField("_state", BindingFlags.Instance | BindingFlags.NonPublic);
                if (_fadeOut == null) return;
            }
            float mu = WorldScale.MetersToUnits;
            float lim = _halfLen + trafficMargin * mu, wid = _halfWid + 1f * mu;
            foreach (var a in FindObjectsByType<TrafficAgent>(FindObjectsSortMode.None))
            {
                Vector3 d = a.transform.position - _siteC;
                if (Mathf.Abs(Vector3.Dot(d, _along)) > lim || Mathf.Abs(Vector3.Dot(d, _across)) > wid) continue;
                if (_stateField != null)
                {
                    string st = _stateField.GetValue(a).ToString();
                    if (st == "FadingOut" || st == "Done") continue;
                }
                _fadeOut.Invoke(a, null);
            }
        }

        // ------------------------------------------------------------- 배치
        private void Place()
        {
            var grid = GridManager.Instance; var roads = RoadManager.Instance;
            var blocks = new HashSet<Vector2Int>(roads.GetAllBlockOrigins());
            if (blocks.Count == 0) return;
            _placedForBlocks = blocks.Count;

            int span = RoadManager.TileSpan;
            float cell = grid.CellSize, half = span * cell * 0.5f, mu = WorldScale.MetersToUnits;
            Vector3 o = grid.Origin; float gy = grid.GroundSurfaceY;
            System.Func<Vector2Int, Vector3> Center = b => new Vector3(o.x + b.x * cell + half, gy, o.z + b.y * cell + half);

            // 도시 중심 = 도로 타일 중심 평균
            Vector3 c = Vector3.zero;
            foreach (var b in blocks) c += Center(b);
            c /= blocks.Count;

            // 중심에 가장 가까운 '직선' 도로 타일(남북 or 동서로만 이어짐) — 앞뒤 타일도 직선이면 점수 우대(공사 구간을 길게)
            Vector2Int best = default; bool found = false, bestNS = false; float bestD = float.MaxValue;
            foreach (var b in blocks)
            {
                bool ns = IsStraight(blocks, b, span, out bool isNS);
                if (!ns) continue;
                Vector2Int step = isNS ? new Vector2Int(0, span) : new Vector2Int(span, 0);
                int bonus = (IsStraight(blocks, b + step, span, out bool n1) && n1 == isNS ? 1 : 0)
                          + (IsStraight(blocks, b - step, span, out bool n2) && n2 == isNS ? 1 : 0);
                float d = (Center(b) - c).sqrMagnitude * (bonus == 2 ? 0.6f : 1f);
                if (d < bestD) { bestD = d; best = b; found = true; bestNS = isNS; }
            }
            if (!found) { Debug.LogWarning("[DemoCrew] 도시 중심 근처에 직선 도로가 없음"); return; }

            _along = bestNS ? Vector3.forward : Vector3.right;
            _across = bestNS ? Vector3.right : Vector3.forward;
            Vector2Int st = bestNS ? new Vector2Int(0, span) : new Vector2Int(span, 0);
            bool front = IsStraight(blocks, best + st, span, out bool f1) && f1 == bestNS;
            bool back = IsStraight(blocks, best - st, span, out bool f2) && f2 == bestNS;
            // 구간 = 고른 타일 + 이어진 직선 타일 절반씩 (최대 2타일 길이)
            float plus = front ? half : 0f, minus = back ? half : 0f;
            _siteC = Center(best) + _along * ((plus - minus) * 0.5f);
            _halfLen = half + (plus + minus) * 0.5f;
            _halfWid = VehicleManager.RoadHalfWidth;
            _haveSite = true;

            BuildSiteProps(gy, mu);
            PlaceCrew(gy, mu);
            Debug.Log($"[DemoCrew] 도시 중심({c.x:F0}, {c.z:F0}) 도로 공사 구간 {_siteC} · 길이 {_halfLen * 2f:F1} · 폭 {_halfWid * 2f:F1}");
        }

        private static bool IsStraight(HashSet<Vector2Int> blocks, Vector2Int b, int span, out bool ns)
        {
            ns = false;
            if (!blocks.Contains(b)) return false;
            bool n = blocks.Contains(b + new Vector2Int(0, span)), s = blocks.Contains(b + new Vector2Int(0, -span));
            bool e = blocks.Contains(b + new Vector2Int(span, 0)), w = blocks.Contains(b + new Vector2Int(-span, 0));
            if (n && s && !e && !w) { ns = true; return true; }
            if (e && w && !n && !s) { ns = false; return true; }
            return false;
        }

        // ------------------------------------------------------------- 공사 소품 (게임에 있는 아트만)
        /// <summary>
        /// 새 도형은 안 쓰고 게임 아트(Resources/Art/Props)만 — 물통 · 고깔(PROP_WATER_CONE)로 양 끝을 가로로 막고 양옆에 줄 세움,
        /// 화분(PROP_PLANTERS)은 네 모서리에. 아트가 없으면 그 소품은 건너뜀.
        /// </summary>
        private void BuildSiteProps(float gy, float mu)
        {
            if (_props != null) Destroy(_props.gameObject);
            _props = new GameObject("[DemoRoadWork]").transform;
            var cone = Resources.Load<GameObject>(Tycoon.Placement.ArtResourceRegistry.ArtRoot + "/Props/PROP_WATER_CONE");
            var plant = Resources.Load<GameObject>(Tycoon.Placement.ArtResourceRegistry.ArtRoot + "/Props/PROP_PLANTERS");
            if (cone == null && plant == null) { Debug.LogWarning("[DemoCrew] Art/Props 의 PROP_WATER_CONE · PROP_PLANTERS 없음"); return; }

            float step = 1.3f * mu;
            if (cone != null)
            {
                // 양 끝: 도로를 가로로 막는 줄
                for (int end = -1; end <= 1; end += 2)
                    for (float c = -_halfWid + 0.4f * mu; c <= _halfWid - 0.4f * mu + 1e-3f; c += step)
                        Spawn(cone, _siteC + _along * (end * _halfLen) + _across * c, Quaternion.LookRotation(_along * end), mu, gy);
                // 양옆: 길 따라 줄
                for (int side = -1; side <= 1; side += 2)
                    for (float a = -_halfLen + step; a <= _halfLen - step + 1e-3f; a += step * 1.6f)
                        Spawn(cone, _siteC + _across * (side * (_halfWid - 0.4f * mu)) + _along * a, Quaternion.LookRotation(_across * side), mu, gy);
            }
            if (plant != null)
                for (int end = -1; end <= 1; end += 2)
                    for (int side = -1; side <= 1; side += 2)
                        Spawn(plant, _siteC + _along * (end * (_halfLen + 0.9f * mu)) + _across * (side * (_halfWid - 0.6f * mu)),
                              Quaternion.LookRotation(_along * end), mu, gy);
        }

        private void Spawn(GameObject prefab, Vector3 pos, Quaternion rot, float mu, float gy)
        {
            pos.y = gy + propLift * mu;   // 도로 노면과 같은 높이면 바닥면이 깜빡임(z-fighting) → 살짝 띄움
            var go = Instantiate(prefab, pos, rot, _props);
            go.transform.localScale = prefab.transform.localScale * mu;   // 아트는 1:1(m) — 월드 배율만
            foreach (var col in go.GetComponentsInChildren<Collider>()) Destroy(col);   // 클릭 · 배치 판정에 안 걸리게
        }

        // ------------------------------------------------------------- 인부
        /// <summary>구간 안에 지그재그로 5명 — 서로 2.4m 이상, 각자 파헤친 노면 가운데 쪽을 보고.</summary>
        private void PlaceCrew(float gy, float mu)
        {
            var clips = PickClips();
            if (clips.Count == 0) { Debug.LogError($"[DemoCrew] CSV 없음: {DemoPaths.ProcessedFolder} — p1_2 · p1_5 스크립트 먼저"); return; }
            EnsureAssets();
            if (_prefab == null) return;

            float usable = Mathf.Max(_halfLen - 1.4f * mu, 0.5f * mu);
            float side = Mathf.Min(_halfWid * 0.45f, 1.5f * mu);
            for (int i = 0; i < crewSize; i++)
            {
                float t = crewSize == 1 ? 0f : Mathf.Lerp(-1f, 1f, i / (float)(crewSize - 1));
                Vector3 pos = _siteC + _along * (t * usable) + _across * ((i % 2 == 0 ? -1f : 1f) * side);
                pos.y = gy;
                Vector3 look = _siteC + _along * (t * usable * 0.7f) - pos; look.y = 0;
                var rot = Quaternion.LookRotation(look.sqrMagnitude > 1e-4f ? look.normalized : _along);

                DemoMotionPlayer p = i < _crew.Count ? _crew[i] : null;
                if (p == null)
                {
                    var go = Instantiate(_prefab, pos, Quaternion.identity);
                    go.name = $"DemoWorker_{i}";
                    foreach (var r in go.GetComponentsInChildren<Renderer>()) r.sharedMaterial = _mat;
                    p = go.AddComponent<DemoMotionPlayer>();
                    p.applyRootMotion = false;                       // 제자리 모드
                    p.mirror = (i / 2) % 2 == 1;                     // 같은 클립 2번째 사용자는 좌우 반전
                    p.Init(clips[i % clips.Count], pos, rot, i * 3.7f);   // 같은 클립이라도 시작 시점을 엇갈리게
                    if (i < _crew.Count) _crew[i] = p; else _crew.Add(p);
                }
                else { p.gameObject.SetActive(true); p.MoveTo(pos, rot); }
            }
            _follow = 0;
        }

        /// <summary>
        /// 망치 = 가장 정확했던 s8LR, 삽 = HmyvW — 망치 · 삽 · 망치 · 삽 · 망치 순서. 같은 클립은 시작 시점 · 좌우 반전으로 다르게.
        /// 지정 클립이 없으면 폴더의 다른 hammer / shovel 클립으로.
        /// </summary>
        private List<string> PickClips()
        {
            var res = new List<string>();
            string dir = DemoPaths.ProcessedFolder;
            if (!Directory.Exists(dir)) return res;
            string suf = Directory.GetFiles(dir, "*" + suffix).Length > 0 ? suffix : fallbackSuffix;
            string h = FindClip(dir, suf, hammerClip, "hammer"), sh = FindClip(dir, suf, shovelClip, "shovel");
            for (int i = 0; i < crewSize; i++)
            {
                string c = i % 2 == 0 ? (h ?? sh) : (sh ?? h);
                if (c != null) res.Add(c);
            }
            return res;
        }

        private static string FindClip(string dir, string suf, string preferred, string kind)
        {
            string f = Path.Combine(dir, preferred + suf);
            if (File.Exists(f)) return f;
            var any = Directory.GetFiles(dir, kind + "*" + suf);
            return any.Length > 0 ? any[0] : null;
        }

        private void EnsureAssets()
        {
            if (_prefab == null)
            {
                _prefab = Resources.Load<GameObject>("DemoWorker/SM_Worker");
                if (_prefab == null) { Debug.LogError("[DemoCrew] Resources/DemoWorker/SM_Worker 없음"); return; }
            }
            if (_mat != null) return;
            var sh = Shader.Find("Universal Render Pipeline/Lit");
            if (sh == null) sh = Shader.Find("Standard");
            _mat = new Material(sh) { name = "M_DemoWorker" };
            var bc = Resources.Load<Texture2D>("DemoWorker/T_Worker_BaseColor");
            var nm = Resources.Load<Texture2D>("DemoWorker/T_Worker_Normal");
            var ms = Resources.Load<Texture2D>("DemoWorker/T_Worker_MetallicSmoothness");
            if (bc != null) { _mat.SetTexture("_BaseMap", bc); _mat.SetTexture("_MainTex", bc); }
            _mat.SetColor("_BaseColor", Color.white);
            if (nm != null) { _mat.SetTexture("_BumpMap", nm); _mat.EnableKeyword("_NORMALMAP"); }
            if (ms != null)
            {
                _mat.SetTexture("_MetallicGlossMap", ms); _mat.EnableKeyword("_METALLICSPECGLOSSMAP");
                _mat.SetFloat("_SmoothnessTextureChannel", 0f); _mat.SetFloat("_Smoothness", 1f);
            }
            else _mat.SetFloat("_Smoothness", 0.35f);
        }

        // ------------------------------------------------------------- labels
        private void OnGUI()
        {
            if (DemoFreeCamera.UiHidden) return;   // H: 이름표 · 프레임 · 안내 글 전부 숨김(영상 녹화용)
            var cam = Camera.main;
            if (cam == null || _crew.Count == 0) return;
            var style = new GUIStyle(GUI.skin.label) { fontSize = 14, alignment = TextAnchor.MiddleCenter };
            for (int i = 0; i < _crew.Count; i++)
            {
                var p = _crew[i]; if (p == null || p.Hips == null) continue;
                bool sel = i == _follow % _crew.Count;
                if (!_labels && !sel) continue;
                Vector3 sp = cam.WorldToScreenPoint(p.Hips.position + Vector3.up * 1.2f);
                if (sp.z < 0) continue;
                style.normal.textColor = sel ? Color.yellow : Color.white;
                string text = $"{(sel ? "> " : "")}{p.ClipName.Replace("_work", "").Replace("_refined", "")} [{p.tool}]  {p.CurrentFrame}/{p.FrameCount}";
                GUI.Label(new Rect(sp.x - 160, Screen.height - sp.y - 12, 320, 24), text, style);
            }
            var hs = new GUIStyle(GUI.skin.label) { fontSize = 14 };
            hs.normal.textColor = Color.white;
            GUI.Label(new Rect(16, 16, 1400, 24),
                $"Demo: road works, {_crew.Count} workers{(DemoMotionPlayer.Paused ? "  [PAUSED]" : "")}   |   G re-place   Tab next worker   L labels   Space play/pause   Backspace restart   F2 quarter view   F3 follow   H hide UI", hs);
        }
    }
}
