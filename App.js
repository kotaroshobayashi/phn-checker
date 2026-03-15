import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  SafeAreaView,
  Alert,
  Image,
  Platform,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import * as ImagePicker from 'expo-image-picker';

// ─────────────────────────────────────────────────────────
// AI API 設定
// ─────────────────────────────────────────────────────────
// ローカル開発: http://localhost:8000
// Android実機: http://<PCのIPアドレス>:8000
// iOSシミュレータ: http://localhost:8000
const AI_API_URL = 'http://localhost:8000';

// KapoorPaper モデルの8クラス 日本語ラベル
const CLASS_LABELS_JA = {
  'BA-cellulitis':              '蜂窩織炎（細菌性）',
  'BA-impetigo':                'とびひ（細菌性）',
  'FU-athlete-foot':            '水虫（真菌性）',
  'FU-nail-fungus':             '爪白癬（真菌性）',
  'FU-ringworm':                '白癬・たむし（真菌性）',
  'PA-cutaneous-larva-migrans': '皮膚幼虫移行症（寄生虫）',
  'VI-chickenpox':              '水痘・水ぼうそう（ウイルス性）',
  'VI-shingles':                '帯状疱疹（ウイルス性）',
};
import Svg, {
  Circle,
  Ellipse,
  Path,
  Rect,
  G,
  Line,
  Text as SvgText,
} from 'react-native-svg';

// ─────────────────────────────────────────────────────────
// データ定義
// ─────────────────────────────────────────────────────────
const STEPS = ['患者情報', '症状チェック', '皮疹写真', '痛みエリア', '判定結果'];

const riskFactors = [
  { id: 'immune',     label: '免疫低下状態（糖尿病・ステロイド使用・がん治療中など）', score: 2 },
  { id: 'chickenpox', label: '水痘（みずぼうそう）の既往あり', score: 1 },
  { id: 'stress',     label: '最近、強いストレスや疲労がある', score: 1 },
];

const symptoms = [
  { id: 'pain_one_side',    label: '片側だけの皮膚の痛み・違和感',         score: 3, tag: '主要症状' },
  { id: 'rash',             label: '赤い発疹・水ぶくれ（片側）',           score: 4, tag: '主要症状' },
  { id: 'tingling',         label: 'ピリピリ・ヒリヒリする感覚',           score: 3, tag: '主要症状' },
  { id: 'numbness',         label: '感覚が鈍くなる（しびれ・麻痺感）',     score: 2, tag: '主要症状' },
  { id: 'allodynia',        label: '触れると過剰に痛む・過敏（アロデニア）', score: 2, tag: '主要症状' },
  { id: 'fever',            label: '発熱（37.5℃以上）',                  score: 1, tag: '随伴症状' },
  { id: 'headache',         label: '頭痛',                                score: 1, tag: '随伴症状' },
  { id: 'fatigue',          label: '強い倦怠感・だるさ',                   score: 1, tag: '随伴症状' },
  { id: 'light_sensitivity',label: '光がまぶしい・目の痛み',               score: 1, tag: '随伴症状' },
];

const highRiskRegionIds = [
  'T3L','T4L','T5L','T3R','T4R','T5R','C2','C3','T6L','T6R','T7L','T7R',
];

const dermatomeRegions = [
  { id: 'C2',  label: 'C2 頭・後頭部',     cx: 100, cy: 22,  r: 14, color: '#FF6B6B' },
  { id: 'C3',  label: 'C3 頸部',           cx: 100, cy: 43,  r: 10, color: '#FF8C69' },
  { id: 'C4',  label: 'C4 肩',             cx: 100, cy: 58,  r: 10, color: '#FFA07A' },
  { id: 'T1L', label: 'T1 上胸部（左）',   cx: 78,  cy: 73,  r: 9,  color: '#FFB347' },
  { id: 'T2L', label: 'T2 胸部（左）',     cx: 75,  cy: 86,  r: 9,  color: '#FFD700' },
  { id: 'T3L', label: 'T3 胸部（左）',     cx: 73,  cy: 99,  r: 9,  color: '#ADFF2F' },
  { id: 'T4L', label: 'T4 胸部（左）',     cx: 71,  cy: 112, r: 9,  color: '#7FFF00' },
  { id: 'T5L', label: 'T5 胸部（左）',     cx: 70,  cy: 124, r: 9,  color: '#00FA9A' },
  { id: 'T6L', label: 'T6 中胸部（左）',   cx: 69,  cy: 136, r: 9,  color: '#00CED1' },
  { id: 'T7L', label: 'T7 中胸部（左）',   cx: 68,  cy: 148, r: 9,  color: '#1E90FF' },
  { id: 'T8L', label: 'T8 上腹部（左）',   cx: 68,  cy: 160, r: 9,  color: '#6495ED' },
  { id: 'T9L', label: 'T9 上腹部（左）',   cx: 68,  cy: 171, r: 9,  color: '#7B68EE' },
  { id: 'T10L',label: 'T10 臍（左）',      cx: 68,  cy: 182, r: 9,  color: '#9370DB' },
  { id: 'T11L',label: 'T11 下腹部（左）',  cx: 69,  cy: 193, r: 9,  color: '#BA55D3' },
  { id: 'T12L',label: 'T12 下腹部（左）',  cx: 70,  cy: 204, r: 9,  color: '#FF69B4' },
  { id: 'T1R', label: 'T1 上胸部（右）',   cx: 122, cy: 73,  r: 9,  color: '#FFB347' },
  { id: 'T2R', label: 'T2 胸部（右）',     cx: 125, cy: 86,  r: 9,  color: '#FFD700' },
  { id: 'T3R', label: 'T3 胸部（右）',     cx: 127, cy: 99,  r: 9,  color: '#ADFF2F' },
  { id: 'T4R', label: 'T4 胸部（右）',     cx: 129, cy: 112, r: 9,  color: '#7FFF00' },
  { id: 'T5R', label: 'T5 胸部（右）',     cx: 130, cy: 124, r: 9,  color: '#00FA9A' },
  { id: 'T6R', label: 'T6 中胸部（右）',   cx: 131, cy: 136, r: 9,  color: '#00CED1' },
  { id: 'T7R', label: 'T7 中胸部（右）',   cx: 132, cy: 148, r: 9,  color: '#1E90FF' },
  { id: 'T8R', label: 'T8 上腹部（右）',   cx: 132, cy: 160, r: 9,  color: '#6495ED' },
  { id: 'T9R', label: 'T9 上腹部（右）',   cx: 132, cy: 171, r: 9,  color: '#7B68EE' },
  { id: 'T10R',label: 'T10 臍（右）',      cx: 132, cy: 182, r: 9,  color: '#9370DB' },
  { id: 'T11R',label: 'T11 下腹部（右）',  cx: 131, cy: 193, r: 9,  color: '#BA55D3' },
  { id: 'T12R',label: 'T12 下腹部（右）',  cx: 130, cy: 204, r: 9,  color: '#FF69B4' },
  { id: 'L1L', label: 'L1 鼠径部（左）',   cx: 80,  cy: 220, r: 9,  color: '#FF1493' },
  { id: 'L2L', label: 'L2 大腿前面（左）', cx: 78,  cy: 248, r: 9,  color: '#DC143C' },
  { id: 'L3L', label: 'L3 膝（左）',       cx: 75,  cy: 272, r: 9,  color: '#B22222' },
  { id: 'L4L', label: 'L4 下腿内側（左）', cx: 72,  cy: 296, r: 9,  color: '#8B0000' },
  { id: 'L5L', label: 'L5 足背（左）',     cx: 70,  cy: 320, r: 9,  color: '#4B0082' },
  { id: 'L1R', label: 'L1 鼠径部（右）',   cx: 120, cy: 220, r: 9,  color: '#FF1493' },
  { id: 'L2R', label: 'L2 大腿前面（右）', cx: 122, cy: 248, r: 9,  color: '#DC143C' },
  { id: 'L3R', label: 'L3 膝（右）',       cx: 125, cy: 272, r: 9,  color: '#B22222' },
  { id: 'L4R', label: 'L4 下腿内側（右）', cx: 128, cy: 296, r: 9,  color: '#8B0000' },
  { id: 'L5R', label: 'L5 足背（右）',     cx: 130, cy: 320, r: 9,  color: '#4B0082' },
  { id: 'S1L', label: 'S1 足底（左）',     cx: 68,  cy: 345, r: 9,  color: '#2F0047' },
  { id: 'S1R', label: 'S1 足底（右）',     cx: 132, cy: 345, r: 9,  color: '#2F0047' },
  { id: 'C5L', label: 'C5 上腕（左）',     cx: 47,  cy: 100, r: 9,  color: '#FF8C69' },
  { id: 'C6L', label: 'C6 前腕（左）',     cx: 40,  cy: 130, r: 9,  color: '#FFA07A' },
  { id: 'C7L', label: 'C7 中指（左）',     cx: 35,  cy: 158, r: 9,  color: '#FFB347' },
  { id: 'C8L', label: 'C8 小指側（左）',   cx: 32,  cy: 182, r: 9,  color: '#FFD700' },
  { id: 'C5R', label: 'C5 上腕（右）',     cx: 153, cy: 100, r: 9,  color: '#FF8C69' },
  { id: 'C6R', label: 'C6 前腕（右）',     cx: 160, cy: 130, r: 9,  color: '#FFA07A' },
  { id: 'C7R', label: 'C7 中指（右）',     cx: 165, cy: 158, r: 9,  color: '#FFB347' },
  { id: 'C8R', label: 'C8 小指側（右）',   cx: 168, cy: 182, r: 9,  color: '#FFD700' },
];

// ─────────────────────────────────────────────────────────
// ヘルパー関数
// ─────────────────────────────────────────────────────────
function getAgeScore(age) {
  const n = parseInt(age, 10);
  if (isNaN(n) || n < 0) return 0;
  if (n >= 80) return 3;
  if (n >= 70) return 2;
  if (n >= 60) return 1;
  return 0;
}

function getDaysSince(dateStr) {
  if (!dateStr) return null;
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return null;
  const diff = Math.floor((new Date() - d) / (1000 * 60 * 60 * 24));
  return diff >= 0 ? diff : 0;
}

function getRiskLevel(score) {
  if (score >= 12) return { level: '超高', bg: '#fef2f2', border: '#dc2626', text: '#991b1b', badge: '#b91c1c' };
  if (score >= 8)  return { level: '高',   bg: '#fef2f2', border: '#f87171', text: '#b91c1c', badge: '#ef4444' };
  if (score >= 5)  return { level: '中高', bg: '#fff7ed', border: '#fb923c', text: '#c2410c', badge: '#f97316' };
  if (score >= 3)  return { level: '中',   bg: '#fefce8', border: '#facc15', text: '#a16207', badge: '#eab308' };
  return            { level: '低',   bg: '#f0fdf4', border: '#4ade80', text: '#15803d', badge: '#22c55e' };
}

function getRecommendation(score, rashStartDate, checkedSymptoms, selectedRegions) {
  const hasRash = checkedSymptoms['rash'];
  const hasEye  = checkedSymptoms['light_sensitivity'];
  const hasOphthalmic = selectedRegions.some(id => ['C2', 'C3'].includes(id));
  const daysSince = getDaysSince(rashStartDate);
  const inWindow  = hasRash && daysSince !== null && daysSince < 3;
  const hoursLeft = inWindow ? Math.max(0, 72 - daysSince * 24) : null;

  if (hasOphthalmic && (hasRash || hasEye) && score >= 5) {
    return {
      urgency: '眼科緊急', urgencyColor: '#9333ea',
      deadlineDays: '本日中（今すぐ）',
      action: '眼科・救急外来への今すぐ受診',
      detail: '顔面・目周辺の帯状疱疹（眼部帯状疱疹）の可能性があります。角膜炎・ぶどう膜炎による視力障害のリスクがあるため、今すぐ眼科または救急外来を受診してください。',
      department: '眼科・救急外来', mustGo: true,
    };
  }
  if (score >= 12) {
    return {
      urgency: '超緊急', urgencyColor: '#b91c1c',
      deadlineDays: '本日中',
      action: '救急外来または皮膚科への今すぐ受診',
      detail: inWindow
        ? `発疹から約${daysSince}日（抗ウイルス薬最適投与期間まで残り約${hoursLeft}時間）。今すぐ受診してください。PHN予防のためにも早期治療が不可欠です。`
        : '重篤な帯状疱疹の可能性が非常に高いです。本日中に皮膚科または救急外来を受診してください。',
      department: '皮膚科・救急外来', mustGo: true,
    };
  }
  if (score >= 8) {
    return {
      urgency: '緊急', urgencyColor: '#ef4444',
      deadlineDays: inWindow ? `本日〜明日中（残り約${hoursLeft}時間）` : '24時間以内',
      action: '皮膚科・内科への早急な受診',
      detail: inWindow
        ? `発疹から約${daysSince}日です。抗ウイルス薬の効果が最大となる72時間以内です。本日または明日中に皮膚科または内科を受診してください。PHNは早期治療で予防できます。`
        : '高リスクです。24時間以内に皮膚科または内科を受診してください。PHN（帯状疱疹後神経痛）予防のためにも早期治療が重要です。',
      department: '皮膚科・内科', mustGo: true,
    };
  }
  if (score >= 5) {
    return {
      urgency: '要注意', urgencyColor: '#f97316',
      deadlineDays: '3日以内',
      action: '皮膚科・内科への受診を推奨',
      detail: '帯状疱疹の初期症状の可能性があります。3日以内に皮膚科または内科を受診することを推奨します。症状が急激に悪化した場合はすぐに受診してください。',
      department: '皮膚科・内科', mustGo: false,
    };
  }
  if (score >= 3) {
    return {
      urgency: '要観察', urgencyColor: '#eab308',
      deadlineDays: '1週間以内（悪化時は即時）',
      action: '経過観察＋必要に応じて受診',
      detail: '現時点での帯状疱疹リスクはやや低めです。症状が続く場合や悪化した場合は1週間以内に受診してください。',
      department: '内科・皮膚科', mustGo: false,
    };
  }
  return {
    urgency: '経過観察', urgencyColor: '#22c55e',
    deadlineDays: '受診不要（症状変化時は再チェック）',
    action: '継続的な経過観察',
    detail: '現時点での帯状疱疹リスクは低めです。新たな症状が出た場合は再チェックを行ってください。',
    department: '—', mustGo: false,
  };
}

// ─────────────────────────────────────────────────────────
// Checkbox コンポーネント
// ─────────────────────────────────────────────────────────
function Checkbox({ checked, label, onToggle, accentColor = '#2563eb', bgColor, borderColor }) {
  return (
    <TouchableOpacity
      onPress={onToggle}
      activeOpacity={0.7}
      style={[
        s.checkRow,
        {
          backgroundColor: bgColor || (checked ? '#eff6ff' : '#fff'),
          borderColor: borderColor || (checked ? '#93c5fd' : '#e5e7eb'),
        },
      ]}
    >
      <View style={[s.checkBox, { borderColor: checked ? accentColor : '#d1d5db', backgroundColor: checked ? accentColor : '#fff' }]}>
        {checked && <Text style={{ color: '#fff', fontSize: 11, lineHeight: 15, fontWeight: '700' }}>✓</Text>}
      </View>
      <Text style={s.checkLabel}>{label}</Text>
    </TouchableOpacity>
  );
}

// ─────────────────────────────────────────────────────────
// CameraCapture（expo-image-picker / Web対応）
// ─────────────────────────────────────────────────────────
function CameraCapture({ onCapture }) {
  const [errMsg, setErrMsg] = useState(null);
  const isWeb = Platform.OS === 'web';

  const handleCamera = async () => {
    setErrMsg(null);
    try {
      if (isWeb) { await handleGallery(); return; }
      const perm = await ImagePicker.requestCameraPermissionsAsync();
      if (perm.status !== 'granted') {
        Alert.alert('カメラへのアクセスが必要です', '設定からカメラへのアクセスを許可してください。');
        return;
      }
      const result = await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 0.85 });
      if (!result.canceled && result.assets?.[0]) onCapture(result.assets[0].uri);
    } catch (e) {
      setErrMsg(e?.message ?? 'カメラを起動できませんでした');
    }
  };

  const handleGallery = async () => {
    setErrMsg(null);
    try {
      if (!isWeb) {
        const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (perm.status !== 'granted') {
          Alert.alert('フォトライブラリへのアクセスが必要です', '設定からフォトライブラリへのアクセスを許可してください。');
          return;
        }
      }
      const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.85 });
      if (!result.canceled && result.assets?.[0]) onCapture(result.assets[0].uri);
    } catch (e) {
      setErrMsg(e?.message ?? 'ギャラリーを開けませんでした');
    }
  };

  return (
    <View>
      {errMsg ? (
        <View style={{ backgroundColor: '#fef2f2', borderRadius: 10, padding: 12, marginBottom: 12 }}>
          <Text style={{ color: '#dc2626', fontSize: 13, fontWeight: '600' }}>⚠️ {errMsg}</Text>
          <Text style={{ color: '#ef4444', fontSize: 11, marginTop: 4 }}>
            設定 › プライバシーとセキュリティ でカメラ・写真の許可を確認してください
          </Text>
        </View>
      ) : null}

      {!isWeb ? (
        <View>
          <TouchableOpacity onPress={handleCamera} style={s.primaryBtn}>
            <Text style={s.primaryBtnText}>📹  カメラで撮影</Text>
          </TouchableOpacity>
          <View style={{ height: 12 }} />
          <TouchableOpacity onPress={handleGallery} style={s.outlineBtn}>
            <Text style={s.outlineBtnText}>📷  ギャラリーから選択</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <TouchableOpacity onPress={handleGallery} style={s.primaryBtn}>
          <Text style={s.primaryBtnText}>📷  カメラ / ギャラリーから選択</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

// ─────────────────────────────────────────────────────────
// DermatomeBody（react-native-svg）
// ─────────────────────────────────────────────────────────
function DermatomeBody({ selectedRegions, onToggle }) {
  return (
    <View style={{ alignItems: 'center' }}>
      <Svg width={160} height={300} viewBox="0 0 200 375">
        {/* 体のシルエット */}
        <Ellipse cx="100" cy="18" rx="18" ry="20" fill="#F5DEB3" stroke="#C8A882" strokeWidth="1.5" />
        <Rect x="90" y="35" width="20" height="16" rx="4" fill="#F5DEB3" stroke="#C8A882" strokeWidth="1" />
        <Path d="M58,50 Q100,46 142,50 L148,230 Q130,238 100,240 Q70,238 52,230Z" fill="#F5DEB3" stroke="#C8A882" strokeWidth="1.5" />
        <Path d="M30,58 Q50,52 58,55 L62,200 Q48,205 30,198 Q18,190 20,175Z" fill="#F5DEB3" stroke="#C8A882" strokeWidth="1.5" />
        <Path d="M142,55 Q150,52 170,58 Q182,175 180,198 Q162,205 138,200Z" fill="#F5DEB3" stroke="#C8A882" strokeWidth="1.5" />
        <Path d="M52,230 Q75,240 100,240 L98,370 Q82,372 65,368 Q48,364 46,350Z" fill="#F5DEB3" stroke="#C8A882" strokeWidth="1.5" />
        <Path d="M100,240 Q125,240 148,230 L154,350 Q152,364 135,368 Q118,372 102,370Z" fill="#F5DEB3" stroke="#C8A882" strokeWidth="1.5" />
        {/* 顔 */}
        <Ellipse cx="93" cy="15" rx="2.5" ry="3" fill="#8B7355" opacity="0.5" />
        <Ellipse cx="107" cy="15" rx="2.5" ry="3" fill="#8B7355" opacity="0.5" />
        {/* 中央線 */}
        <Line x1="100" y1="50" x2="100" y2="240" stroke="#C8A882" strokeWidth="0.5" strokeDasharray="3,2" opacity="0.5" />
        {/* 左右ラベル */}
        <SvgText x="15" y="72" fontSize="9" fill="#999" fontWeight="bold">左</SvgText>
        <SvgText x="178" y="72" fontSize="9" fill="#999" fontWeight="bold">右</SvgText>

        {/* デルマトーム円 */}
        {dermatomeRegions.map((region) => {
          const isSel = selectedRegions.includes(region.id);
          const isHR  = highRiskRegionIds.includes(region.id);
          return (
            <G key={region.id} onPress={() => onToggle(region.id)}>
              {/* タッチ領域を広げる透明円 */}
              <Circle cx={region.cx} cy={region.cy} r={region.r + 5} fill="transparent" />
              <Circle
                cx={region.cx}
                cy={region.cy}
                r={isSel ? region.r : region.r - 2}
                fill={isSel ? region.color : region.color + '44'}
                stroke={isSel ? region.color : region.color + '88'}
                strokeWidth={isSel ? 2.5 : 1}
              />
              {/* 好発部位マーカー */}
              {isHR && isSel && (
                <Circle cx={region.cx + region.r - 2} cy={region.cy - region.r + 2} r={3} fill="#FF0000" />
              )}
            </G>
          );
        })}
      </Svg>

      {selectedRegions.length > 0 ? (
        <Text style={{ fontSize: 11, color: '#2563eb', marginTop: 4, fontWeight: '700' }}>
          {selectedRegions.length}箇所選択中
        </Text>
      ) : (
        <Text style={{ fontSize: 10, color: '#9ca3af', marginTop: 4 }}>タップして選択</Text>
      )}
    </View>
  );
}

// ─────────────────────────────────────────────────────────
// メインアプリ
// ─────────────────────────────────────────────────────────
export default function App() {
  const [step, setStep]                       = useState(0);
  const [patientName, setPatientName]         = useState('');
  const [roomNumber, setRoomNumber]           = useState('');
  const [staffName, setStaffName]             = useState('');
  const [age, setAge]                         = useState('');
  const [rashStartDate, setRashStartDate]     = useState('');
  const [painDurVal, setPainDurVal]           = useState('');
  const [painDurUnit, setPainDurUnit]         = useState('日');
  const [checkedRisks, setCheckedRisks]       = useState({});
  const [checkedSymptoms, setCheckedSymptoms] = useState({});
  const [otherSymptoms, setOtherSymptoms]     = useState('');
  const [rashPhoto, setRashPhoto]             = useState(null);
  const [aiResult, setAiResult]               = useState(null);   // AI診断結果
  const [aiLoading, setAiLoading]             = useState(false);  // 解析中フラグ
  const [aiError, setAiError]                 = useState(null);   // 解析エラー
  const [selectedRegions, setSelectedRegions] = useState([]);
  const [painSide, setPainSide]               = useState(null);
  const [timestamp]                           = useState(new Date().toLocaleString('ja-JP'));

  const toggleRisk    = (id) => setCheckedRisks(p => ({ ...p, [id]: !p[id] }));
  const toggleSymptom = (id) => setCheckedSymptoms(p => ({ ...p, [id]: !p[id] }));
  const toggleRegion  = (id) => setSelectedRegions(p =>
    p.includes(id) ? p.filter(r => r !== id) : [...p, id]
  );

  // ── AI画像解析 ─────────────────────────────────────────
  const analyzePhoto = async (uri) => {
    setAiLoading(true);
    setAiError(null);
    setAiResult(null);
    try {
      const formData = new FormData();
      formData.append('file', {
        uri:  uri,
        type: 'image/jpeg',
        name: 'rash.jpg',
      });
      const res = await fetch(`${AI_API_URL}/predict`, {
        method:  'POST',
        body:    formData,
      });
      if (!res.ok) throw new Error(`サーバーエラー (${res.status})`);
      const data = await res.json();
      setAiResult(data);
    } catch (e) {
      setAiError(e.message || 'AI解析に失敗しました');
    } finally {
      setAiLoading(false);
    }
  };

  const handlePhotoCapture = (uri) => {
    setRashPhoto(uri);
    analyzePhoto(uri);
  };

  const dermBonus =
    (painSide === 'left' || painSide === 'right' ? 2 : 0) +
    (selectedRegions.some(id => highRiskRegionIds.includes(id)) ? 1 : 0);

  // AI診断スコア加算:
  //   帯状疱疹を70%以上の確信度で検出 → +3
  //   帯状疱疹を50〜70%の確信度で検出 → +2
  //   水痘（帯状疱疹と密接な関連）を検出 → +1
  const aiScore = (() => {
    if (!aiResult) return 0;
    if (aiResult.predicted_class === 'VI-shingles') {
      return aiResult.confidence >= 70 ? 3 : aiResult.confidence >= 50 ? 2 : 1;
    }
    if (aiResult.predicted_class === 'VI-chickenpox') return 1;
    return 0;
  })();

  const totalScore =
    getAgeScore(age) +
    riskFactors.reduce((acc, r) => acc + (checkedRisks[r.id] ? r.score : 0), 0) +
    symptoms.reduce((acc, r) => acc + (checkedSymptoms[r.id] ? r.score : 0), 0) +
    (selectedRegions.length > 0 ? dermBonus : 0) +
    aiScore;

  const risk = getRiskLevel(totalScore);
  const rec  = getRecommendation(totalScore, rashStartDate, checkedSymptoms, selectedRegions);
  const daysSince = getDaysSince(rashStartDate);

  const handleReset = () => {
    setStep(0); setPatientName(''); setRoomNumber(''); setStaffName(''); setAge('');
    setRashStartDate(''); setPainDurVal(''); setPainDurUnit('日');
    setCheckedRisks({}); setCheckedSymptoms({}); setOtherSymptoms('');
    setRashPhoto(null); setAiResult(null); setAiError(null); setAiLoading(false);
    setSelectedRegions([]); setPainSide(null);
  };

  return (
    <SafeAreaView style={s.safeArea}>
      <StatusBar style="light" backgroundColor="#2563eb" />

      {/* ── ヘッダー ── */}
      <View style={s.header}>
        <Text style={s.headerSub}>介護施設スタッフ向け</Text>
        <Text style={s.headerTitle}>🔍 帯状疱疹リスクチェッカー</Text>
      </View>

      {/* ── ステップバー ── */}
      <View style={s.stepBarContainer}>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={s.stepBarContent}
        >
          {STEPS.map((label, i) => (
            <View key={i} style={s.stepItem}>
              <View style={[s.stepCircle, i <= step ? s.stepCircleOn : s.stepCircleOff]}>
                <Text style={[s.stepNum, { color: i <= step ? '#fff' : '#9ca3af' }]}>
                  {i < step ? '✓' : i + 1}
                </Text>
              </View>
              <Text style={[s.stepLabel, i === step ? s.stepLabelOn : s.stepLabelOff]}>
                {label}
              </Text>
              {i < STEPS.length - 1 && (
                <View style={[s.stepConnector, i < step ? s.stepConnectorOn : s.stepConnectorOff]} />
              )}
            </View>
          ))}
        </ScrollView>
      </View>

      {/* ── コンテンツ ── */}
      <ScrollView style={s.scroll} contentContainerStyle={s.scrollContent} keyboardShouldPersistTaps="handled">

        {/* ════════════════════════════════════════
            STEP 0: 患者情報
        ════════════════════════════════════════ */}
        {step === 0 && (
          <View style={s.card}>
            <Text style={s.cardTitle}>📋 患者・スタッフ情報</Text>

            {/* 基本テキスト入力 */}
            {[
              { label: '患者名 *',        ph: '例：山田 花子', val: patientName, set: setPatientName },
              { label: '部屋番号',         ph: '例：201号室',  val: roomNumber,  set: setRoomNumber  },
              { label: 'チェック担当者名', ph: '例：鈴木 一郎', val: staffName,   set: setStaffName   },
            ].map(f => (
              <View key={f.label} style={{ marginBottom: 12 }}>
                <Text style={s.inputLabel}>{f.label}</Text>
                <TextInput
                  style={s.textInput}
                  placeholder={f.ph}
                  placeholderTextColor="#9ca3af"
                  value={f.val}
                  onChangeText={f.set}
                />
              </View>
            ))}

            {/* 年齢 */}
            <View style={{ marginBottom: 12 }}>
              <Text style={s.inputLabel}>年齢 *</Text>
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                <TextInput
                  style={[s.textInput, { width: 80 }]}
                  placeholder="例：78"
                  placeholderTextColor="#9ca3af"
                  keyboardType="numeric"
                  value={age}
                  onChangeText={setAge}
                />
                <Text style={{ color: '#6b7280', fontSize: 14 }}>歳</Text>
                {age !== '' && parseInt(age, 10) >= 60 && (
                  <View style={{
                    backgroundColor: parseInt(age, 10) >= 70 ? '#fef2f2' : '#fefce8',
                    borderRadius: 12, paddingHorizontal: 8, paddingVertical: 2,
                  }}>
                    <Text style={{
                      fontSize: 11, fontWeight: '700',
                      color: parseInt(age, 10) >= 70 ? '#b91c1c' : '#a16207',
                    }}>
                      {parseInt(age, 10) >= 80 ? '高リスク（+3）' : parseInt(age, 10) >= 70 ? '高リスク（+2）' : '注意（+1）'}
                    </Text>
                  </View>
                )}
              </View>
            </View>

            {/* 皮疹の開始日 */}
            <View style={{ marginBottom: 12 }}>
              <Text style={s.inputLabel}>皮疹はいつから？（わかる場合）</Text>
              <TextInput
                style={s.textInput}
                placeholder="例：2025-01-15（YYYY-MM-DD形式）"
                placeholderTextColor="#9ca3af"
                value={rashStartDate}
                onChangeText={setRashStartDate}
                keyboardType="numbers-and-punctuation"
              />
              {rashStartDate && daysSince !== null && (
                <Text style={{
                  fontSize: 12, marginTop: 4, fontWeight: '600',
                  color: daysSince < 3 ? '#dc2626' : daysSince < 7 ? '#ea580c' : '#6b7280',
                }}>
                  {daysSince < 3
                    ? `⚠️ 発疹から約${daysSince}日（72時間ウィンドウ内！）`
                    : `発疹から約${daysSince}日が経過`}
                </Text>
              )}
            </View>

            {/* 痛みの期間 */}
            <View style={{ marginBottom: 16 }}>
              <Text style={s.inputLabel}>痛みはいつから？</Text>
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                <TextInput
                  style={[s.textInput, { width: 70 }]}
                  placeholder="例：3"
                  placeholderTextColor="#9ca3af"
                  keyboardType="numeric"
                  value={painDurVal}
                  onChangeText={setPainDurVal}
                />
                {['日', '週', 'ヶ月'].map(u => (
                  <TouchableOpacity
                    key={u}
                    onPress={() => setPainDurUnit(u)}
                    style={[s.unitBtn, painDurUnit === u ? s.unitBtnOn : s.unitBtnOff]}
                  >
                    <Text style={[s.unitBtnTxt, { color: painDurUnit === u ? '#fff' : '#374151' }]}>{u}</Text>
                  </TouchableOpacity>
                ))}
                <Text style={{ color: '#6b7280', fontSize: 13 }}>前から</Text>
              </View>
            </View>

            {/* リスク背景 */}
            <View style={s.blueBox}>
              <Text style={[s.sectionLabel, { color: '#1d4ed8', marginBottom: 10 }]}>⚠️ リスク背景</Text>
              {riskFactors.map(r => (
                <Checkbox
                  key={r.id}
                  checked={!!checkedRisks[r.id]}
                  label={r.label}
                  onToggle={() => toggleRisk(r.id)}
                  accentColor="#2563eb"
                  bgColor={checkedRisks[r.id] ? '#dbeafe' : '#fff'}
                  borderColor={checkedRisks[r.id] ? '#93c5fd' : '#e5e7eb'}
                />
              ))}
            </View>

            <TouchableOpacity
              onPress={() => setStep(1)}
              disabled={!patientName || !age}
              style={[s.primaryBtn, (!patientName || !age) && { opacity: 0.4 }]}
            >
              <Text style={s.primaryBtnText}>次へ：症状チェック →</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* ════════════════════════════════════════
            STEP 1: 症状チェック
        ════════════════════════════════════════ */}
        {step === 1 && (
          <View style={s.card}>
            <Text style={s.cardTitle}>🩺 現在の症状</Text>
            <Text style={{ fontSize: 12, color: '#9ca3af', marginBottom: 12 }}>
              患者本人や他スタッフからの情報も含めて選択してください
            </Text>

            {['主要症状', '随伴症状'].map(tag => (
              <View key={tag} style={{ marginBottom: 16 }}>
                <View style={[s.tagBadge, { backgroundColor: tag === '主要症状' ? '#fee2e2' : '#f3f4f6' }]}>
                  <Text style={[s.tagTxt, { color: tag === '主要症状' ? '#b91c1c' : '#4b5563' }]}>{tag}</Text>
                </View>
                <View style={{ marginTop: 8, gap: 8 }}>
                  {symptoms.filter(sym => sym.tag === tag).map(sym => (
                    <Checkbox
                      key={sym.id}
                      checked={!!checkedSymptoms[sym.id]}
                      label={sym.label}
                      onToggle={() => toggleSymptom(sym.id)}
                      accentColor={tag === '主要症状' ? '#ef4444' : '#f97316'}
                      bgColor={checkedSymptoms[sym.id] ? (tag === '主要症状' ? '#fef2f2' : '#fff7ed') : '#fff'}
                      borderColor={checkedSymptoms[sym.id] ? (tag === '主要症状' ? '#fca5a5' : '#fdba74') : '#e5e7eb'}
                    />
                  ))}
                </View>
              </View>
            ))}

            {/* その他自由記入 */}
            <View style={{ marginBottom: 16 }}>
              <Text style={s.inputLabel}>その他の症状（自由記入）</Text>
              <TextInput
                style={[s.textInput, { height: 80, textAlignVertical: 'top', paddingTop: 8 }]}
                placeholder="例：右わき腹にじんじんする感覚、昨日から眠れないほどの痛み..."
                placeholderTextColor="#9ca3af"
                multiline
                value={otherSymptoms}
                onChangeText={setOtherSymptoms}
              />
            </View>

            <View style={s.rowBtns}>
              <TouchableOpacity onPress={() => setStep(0)} style={s.outlineBtn}>
                <Text style={s.outlineBtnText}>← 戻る</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => setStep(2)} style={[s.primaryBtn, { flex: 2 }]}>
                <Text style={s.primaryBtnText}>次へ：皮疹写真 →</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* ════════════════════════════════════════
            STEP 2: 皮疹写真
        ════════════════════════════════════════ */}
        {step === 2 && (
          <View style={s.card}>
            <Text style={s.cardTitle}>📷 皮疹の写真撮影</Text>
            <Text style={{ fontSize: 12, color: '#9ca3af', marginBottom: 16 }}>
              皮疹がある場合は写真を撮影してください。スキップも可能です。
            </Text>

            {rashPhoto ? (
              <View style={{ gap: 12 }}>
                <Image
                  source={{ uri: rashPhoto }}
                  style={s.photoPreview}
                  resizeMode="contain"
                />

                {/* ── AI 解析結果 ── */}
                {aiLoading && (
                  <View style={{ backgroundColor: '#eff6ff', borderRadius: 12, padding: 14, borderWidth: 1, borderColor: '#bfdbfe', alignItems: 'center' }}>
                    <Text style={{ color: '#2563eb', fontWeight: '700', fontSize: 14 }}>
                      🤖 AI が皮膚疾患を解析中...
                    </Text>
                    <Text style={{ color: '#3b82f6', fontSize: 12, marginTop: 4 }}>
                      EfficientNetB0 モデルで分類しています
                    </Text>
                  </View>
                )}

                {aiError && !aiLoading && (
                  <View style={{ backgroundColor: '#fef9c3', borderRadius: 12, padding: 12, borderWidth: 1, borderColor: '#fde047' }}>
                    <Text style={{ color: '#854d0e', fontWeight: '700', fontSize: 13 }}>
                      ⚠️ AI解析に接続できませんでした
                    </Text>
                    <Text style={{ color: '#92400e', fontSize: 11, marginTop: 4 }}>
                      {aiError}
                    </Text>
                    <Text style={{ color: '#92400e', fontSize: 11, marginTop: 4 }}>
                      APIサーバーが起動しているか確認してください。スコアは症状・年齢のみで算出されます。
                    </Text>
                    <TouchableOpacity
                      onPress={() => analyzePhoto(rashPhoto)}
                      style={{ marginTop: 8, backgroundColor: '#fbbf24', borderRadius: 8, padding: 8, alignItems: 'center' }}>
                      <Text style={{ color: '#1c1917', fontWeight: '700', fontSize: 12 }}>再解析する</Text>
                    </TouchableOpacity>
                  </View>
                )}

                {aiResult && !aiLoading && (
                  <View style={[
                    { borderRadius: 12, padding: 14, borderWidth: 2, gap: 8 },
                    aiResult.shingles_detected
                      ? { backgroundColor: '#fef2f2', borderColor: '#f87171' }
                      : { backgroundColor: '#f0fdf4', borderColor: '#86efac' },
                  ]}>
                    <Text style={{ fontWeight: '800', fontSize: 14, color: '#1f2937' }}>
                      🤖 AI 皮膚疾患解析結果
                    </Text>

                    {/* 最有力診断 */}
                    <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                      <View style={[
                        { borderRadius: 8, paddingHorizontal: 10, paddingVertical: 4 },
                        aiResult.shingles_detected
                          ? { backgroundColor: '#dc2626' }
                          : { backgroundColor: '#16a34a' },
                      ]}>
                        <Text style={{ color: '#fff', fontWeight: '700', fontSize: 12 }}>
                          {aiResult.confidence.toFixed(1)}%
                        </Text>
                      </View>
                      <View style={{ flex: 1 }}>
                        <Text style={{ fontWeight: '700', fontSize: 14, color: aiResult.shingles_detected ? '#b91c1c' : '#15803d' }}>
                          {CLASS_LABELS_JA[aiResult.predicted_class] || aiResult.predicted_class}
                        </Text>
                        <Text style={{ fontSize: 11, color: '#6b7280' }}>
                          {aiResult.predicted_class}
                        </Text>
                      </View>
                    </View>

                    {/* 帯状疱疹アラート */}
                    {aiResult.shingles_detected && (
                      <View style={{ backgroundColor: '#fee2e2', borderRadius: 8, padding: 10, borderWidth: 1, borderColor: '#fecaca' }}>
                        <Text style={{ color: '#b91c1c', fontWeight: '800', fontSize: 13 }}>
                          ⚠️ 帯状疱疹の可能性をAIが検出しました
                        </Text>
                        <Text style={{ color: '#dc2626', fontSize: 12, marginTop: 4, lineHeight: 18 }}>
                          確信度 {aiResult.confidence.toFixed(1)}% — リスクスコアに +{aiScore}点 加算されます。
                          医師・看護師への確認を推奨します。
                        </Text>
                      </View>
                    )}

                    {/* 確率上位3位 */}
                    <View style={{ gap: 4 }}>
                      <Text style={{ fontSize: 11, color: '#9ca3af', fontWeight: '600' }}>確率上位3疾患</Text>
                      {Object.entries(aiResult.all_probs).slice(0, 3).map(([cls, prob]) => (
                        <View key={cls} style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                          <View style={{ flex: 1, height: 6, backgroundColor: '#e5e7eb', borderRadius: 3 }}>
                            <View style={{
                              height: 6, borderRadius: 3,
                              width: `${Math.min(prob, 100)}%`,
                              backgroundColor: cls === 'VI-shingles' ? '#ef4444' : '#3b82f6',
                            }} />
                          </View>
                          <Text style={{ fontSize: 11, color: '#374151', width: 36, textAlign: 'right' }}>
                            {prob.toFixed(1)}%
                          </Text>
                          <Text style={{ fontSize: 10, color: '#6b7280', width: 120 }} numberOfLines={1}>
                            {CLASS_LABELS_JA[cls] || cls}
                          </Text>
                        </View>
                      ))}
                    </View>

                    <Text style={{ fontSize: 10, color: '#9ca3af', marginTop: 2 }}>
                      ※ AI解析は補助ツールです。最終判断は医療専門家が行ってください。
                    </Text>
                  </View>
                )}

                <TouchableOpacity
                  onPress={() => { setRashPhoto(null); setAiResult(null); setAiError(null); }}
                  style={s.dangerBtn}>
                  <Text style={{ color: '#ef4444', fontWeight: '700', fontSize: 14 }}>🗑  写真を削除</Text>
                </TouchableOpacity>
                <View style={s.warningBox}>
                  <Text style={{ fontSize: 11, color: '#a16207' }}>
                    ⚠️ 写真はこの端末内にのみ保存されます。外部送信はありません。
                  </Text>
                </View>
              </View>
            ) : (
              <CameraCapture onCapture={handlePhotoCapture} />
            )}

            <View style={[s.rowBtns, { marginTop: 16 }]}>
              <TouchableOpacity onPress={() => setStep(1)} style={s.outlineBtn}>
                <Text style={s.outlineBtnText}>← 戻る</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => setStep(3)} style={[s.primaryBtn, { flex: 2 }]}>
                <Text style={s.primaryBtnText}>{rashPhoto ? '次へ：痛みエリア →' : 'スキップ →'}</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* ════════════════════════════════════════
            STEP 3: 痛みエリア
        ════════════════════════════════════════ */}
        {step === 3 && (
          <View style={s.card}>
            <Text style={s.cardTitle}>🗺️ 痛み・違和感のエリアを選択</Text>
            <Text style={{ fontSize: 12, color: '#9ca3af', marginBottom: 12 }}>
              体の図をタップして痛みのある部位を選択してください（複数選択可）
            </Text>

            {/* 片側 / 両側 */}
            <View style={{ marginBottom: 16 }}>
              <Text style={s.sectionLabel}>痛みはどちら側ですか？</Text>
              <View style={{ flexDirection: 'row', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
                {[
                  { v: 'left',    l: '左側のみ' },
                  { v: 'right',   l: '右側のみ' },
                  { v: 'both',    l: '両側' },
                  { v: 'unknown', l: '不明' },
                ].map(o => (
                  <TouchableOpacity
                    key={o.v}
                    onPress={() => setPainSide(o.v)}
                    style={[s.sideBtn, painSide === o.v ? s.sideBtnOn : s.sideBtnOff]}
                  >
                    <Text style={[s.sideBtnTxt, { color: painSide === o.v ? '#fff' : '#374151' }]}>{o.l}</Text>
                  </TouchableOpacity>
                ))}
              </View>
              {(painSide === 'left' || painSide === 'right') && (
                <Text style={{ fontSize: 11, color: '#dc2626', fontWeight: '600', marginTop: 6 }}>
                  ⚠️ 片側の痛みは帯状疱疹の重要なサインです
                </Text>
              )}
            </View>

            {/* デルマトームマップ ＋ 凡例 */}
            <View style={{ flexDirection: 'row', gap: 12, alignItems: 'flex-start' }}>
              <DermatomeBody selectedRegions={selectedRegions} onToggle={toggleRegion} />

              <View style={{ flex: 1, paddingTop: 12 }}>
                <Text style={s.sectionLabel}>神経領域</Text>
                {[
                  { color: '#FF6B6B', label: '頭頸部 (C)' },
                  { color: '#00CED1', label: '胸部 (T)' },
                  { color: '#FF1493', label: '腰部 (L)' },
                  { color: '#4B0082', label: '仙骨 (S)' },
                ].map(l => (
                  <View key={l.label} style={{ flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                    <View style={{ width: 10, height: 10, borderRadius: 5, backgroundColor: l.color }} />
                    <Text style={{ fontSize: 11, color: '#6b7280' }}>{l.label}</Text>
                  </View>
                ))}

                <View style={{ marginTop: 10, backgroundColor: '#fef2f2', borderWidth: 1, borderColor: '#fecaca', borderRadius: 8, padding: 8 }}>
                  <Text style={{ fontSize: 11, fontWeight: '700', color: '#dc2626' }}>🔴 好発部位</Text>
                  <Text style={{ fontSize: 10, color: '#ef4444', marginTop: 2, lineHeight: 14 }}>
                    胸部T3〜T7と顔面・頭部（C2/C3）が特に多い
                  </Text>
                </View>

                {selectedRegions.some(id => highRiskRegionIds.includes(id)) && (
                  <View style={{ marginTop: 8, backgroundColor: '#eff6ff', borderRadius: 8, padding: 8 }}>
                    <Text style={{ fontSize: 11, color: '#dc2626', fontWeight: '600' }}>⚠️ 好発部位含む</Text>
                  </View>
                )}
              </View>
            </View>

            {/* 選択中の部位タグ */}
            {selectedRegions.length > 0 && (
              <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 4, marginTop: 12 }}>
                {selectedRegions.map(id => {
                  const region = dermatomeRegions.find(r => r.id === id);
                  return region ? (
                    <View key={id} style={{ backgroundColor: region.color, borderRadius: 10, paddingHorizontal: 8, paddingVertical: 2 }}>
                      <Text style={{ color: '#fff', fontSize: 10, fontWeight: '600' }}>{id}</Text>
                    </View>
                  ) : null;
                })}
              </View>
            )}

            <View style={[s.rowBtns, { marginTop: 16 }]}>
              <TouchableOpacity onPress={() => setStep(2)} style={s.outlineBtn}>
                <Text style={s.outlineBtnText}>← 戻る</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => setStep(4)} style={[s.primaryBtn, { flex: 2 }]}>
                <Text style={s.primaryBtnText}>判定する →</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* ════════════════════════════════════════
            STEP 4: 判定結果
        ════════════════════════════════════════ */}
        {step === 4 && (
          <View style={{ gap: 12 }}>

            {/* スコアカード */}
            <View style={[s.card, { backgroundColor: risk.bg, borderWidth: 2, borderColor: risk.border }]}>
              <View style={{ marginBottom: 12 }}>
                <Text style={{ fontSize: 13, color: '#6b7280' }}>
                  患者：
                  <Text style={{ fontWeight: '700', color: '#374151' }}>{patientName}</Text>
                  {roomNumber ? `  /  ${roomNumber}` : ''}
                  {age ? `  ${age}歳` : ''}
                </Text>
                <Text style={{ fontSize: 11, color: '#9ca3af', marginTop: 2 }}>
                  記録者：{staffName || '未入力'} | {timestamp}
                </Text>
              </View>
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 16 }}>
                <View style={{ width: 64, height: 64, borderRadius: 32, backgroundColor: risk.badge, alignItems: 'center', justifyContent: 'center' }}>
                  <Text style={{ color: '#fff', fontSize: 16, fontWeight: '900' }}>{risk.level}</Text>
                </View>
                <View>
                  <View style={{ backgroundColor: rec.urgencyColor, borderRadius: 10, paddingHorizontal: 8, paddingVertical: 2, alignSelf: 'flex-start', marginBottom: 4 }}>
                    <Text style={{ color: '#fff', fontSize: 11, fontWeight: '700' }}>{rec.urgency}</Text>
                  </View>
                  <Text style={{ fontSize: 22, fontWeight: '900', color: risk.text }}>{risk.level}リスク</Text>
                  <Text style={{ fontSize: 11, color: '#9ca3af' }}>
                    スコア：{totalScore}点{aiScore > 0 ? `（AI+${aiScore}含む）` : ''}
                  </Text>
                </View>
              </View>
            </View>

            {/* 受診期限カード */}
            <View style={[s.card, rec.mustGo && { borderWidth: 2, borderColor: '#f87171', backgroundColor: '#fef2f2' }]}>
              <Text style={{ fontSize: 11, fontWeight: '700', color: '#9ca3af', marginBottom: 4 }}>🏥 受診推奨期限</Text>
              <Text style={{ fontSize: 20, fontWeight: '900', color: rec.mustGo ? '#b91c1c' : '#374151', marginBottom: 8 }}>
                {rec.deadlineDays}
              </Text>
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <View style={{ backgroundColor: '#f3f4f6', borderRadius: 10, paddingHorizontal: 8, paddingVertical: 2 }}>
                  <Text style={{ fontSize: 11, color: '#6b7280' }}>推奨科</Text>
                </View>
                <Text style={{ fontSize: 14, fontWeight: '700', color: '#374151' }}>{rec.department}</Text>
              </View>
              <Text style={{ fontWeight: '700', color: '#1f2937', fontSize: 13, marginBottom: 4 }}>📌 {rec.action}</Text>
              <Text style={{ fontSize: 13, color: '#4b5563', lineHeight: 20 }}>{rec.detail}</Text>
              {rec.mustGo && (
                <View style={{ marginTop: 10, backgroundColor: '#fee2e2', borderRadius: 8, borderWidth: 1, borderColor: '#fecaca', padding: 8 }}>
                  <Text style={{ fontSize: 12, fontWeight: '700', color: '#b91c1c' }}>⚠️ 医療機関への受診が必要です</Text>
                </View>
              )}
            </View>

            {/* 72時間ウィンドウアラート */}
            {checkedSymptoms['rash'] && daysSince !== null && daysSince < 3 && (
              <View style={{ backgroundColor: '#dc2626', borderRadius: 14, padding: 16 }}>
                <Text style={{ color: '#fff', fontWeight: '900', fontSize: 15 }}>
                  ⏰ 抗ウイルス薬の72時間ウィンドウ内！
                </Text>
                <Text style={{ color: '#fff', fontSize: 13, marginTop: 4, lineHeight: 20, opacity: 0.95 }}>
                  発疹出現から{daysSince}日です。今すぐ受診することで、帯状疱疹後神経痛（PHN）への進行を大幅に抑制できます。
                </Text>
              </View>
            )}

            {/* 皮疹写真 + AI診断サマリー */}
            {rashPhoto && (
              <View style={s.card}>
                <Text style={s.sectionLabel}>📷 皮疹写真 / AI解析</Text>
                <Image
                  source={{ uri: rashPhoto }}
                  style={[s.photoPreview, { height: 200, marginTop: 8 }]}
                  resizeMode="contain"
                />
                {rashStartDate && daysSince !== null && (
                  <Text style={{ fontSize: 11, color: '#9ca3af', marginTop: 4 }}>
                    発疹確認日：{rashStartDate}（{daysSince}日前）
                  </Text>
                )}

                {/* AI結果サマリー */}
                {aiResult && (
                  <View style={[
                    { marginTop: 10, borderRadius: 10, padding: 12, borderWidth: 1, gap: 6 },
                    aiResult.shingles_detected
                      ? { backgroundColor: '#fef2f2', borderColor: '#fca5a5' }
                      : { backgroundColor: '#f0fdf4', borderColor: '#86efac' },
                  ]}>
                    <Text style={{ fontWeight: '700', fontSize: 12, color: '#374151' }}>
                      🤖 AI診断（EfficientNetB0）
                    </Text>
                    <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                      <View style={[
                        { borderRadius: 6, paddingHorizontal: 8, paddingVertical: 2 },
                        aiResult.shingles_detected ? { backgroundColor: '#dc2626' } : { backgroundColor: '#16a34a' },
                      ]}>
                        <Text style={{ color: '#fff', fontSize: 11, fontWeight: '700' }}>
                          {aiResult.confidence.toFixed(1)}%
                        </Text>
                      </View>
                      <Text style={{ fontSize: 13, fontWeight: '700', color: aiResult.shingles_detected ? '#b91c1c' : '#15803d', flex: 1 }}>
                        {CLASS_LABELS_JA[aiResult.predicted_class] || aiResult.predicted_class}
                      </Text>
                    </View>
                    {aiScore > 0 && (
                      <Text style={{ fontSize: 11, color: '#dc2626', fontWeight: '600' }}>
                        AI検出によりスコア +{aiScore}点 加算済み
                      </Text>
                    )}
                  </View>
                )}
                {aiError && (
                  <Text style={{ fontSize: 11, color: '#9ca3af', marginTop: 6 }}>
                    ※ AI解析未実施（接続エラー）—スコアは症状・年齢のみ
                  </Text>
                )}
              </View>
            )}

            {/* 痛みエリアサマリー */}
            {selectedRegions.length > 0 && (
              <View style={s.card}>
                <Text style={s.sectionLabel}>🗺️ 痛みエリアサマリー</Text>
                <View style={{ flexDirection: 'row', gap: 12, marginTop: 8 }}>
                  <Svg width={80} height={150} viewBox="0 0 200 375">
                    <Ellipse cx="100" cy="18" rx="18" ry="20" fill="#F5DEB3" stroke="#C8A882" strokeWidth="1.5" />
                    <Rect x="90" y="35" width="20" height="16" rx="4" fill="#F5DEB3" stroke="#C8A882" strokeWidth="1" />
                    <Path d="M58,50 Q100,46 142,50 L148,230 Q130,238 100,240 Q70,238 52,230Z" fill="#F5DEB3" stroke="#C8A882" strokeWidth="1.5" />
                    <Path d="M30,58 Q50,52 58,55 L62,200 Q48,205 30,198Z" fill="#F5DEB3" stroke="#C8A882" strokeWidth="1.5" />
                    <Path d="M142,55 Q150,52 170,58 L170,198 Q152,205 138,200Z" fill="#F5DEB3" stroke="#C8A882" strokeWidth="1.5" />
                    <Path d="M52,230 Q75,240 100,240 L98,370 Q65,368 46,350Z" fill="#F5DEB3" stroke="#C8A882" strokeWidth="1.5" />
                    <Path d="M100,240 Q125,240 148,230 L154,350 Q135,368 102,370Z" fill="#F5DEB3" stroke="#C8A882" strokeWidth="1.5" />
                    {dermatomeRegions
                      .filter(r => selectedRegions.includes(r.id))
                      .map(r => (
                        <Circle key={r.id} cx={r.cx} cy={r.cy} r={r.r} fill={r.color} opacity={0.9} />
                      ))}
                  </Svg>
                  <View style={{ flex: 1 }}>
                    {painSide && painSide !== 'unknown' && (
                      <Text style={{ fontSize: 12, marginBottom: 6 }}>
                        側：
                        <Text style={{ fontWeight: '700', color: (painSide === 'left' || painSide === 'right') ? '#dc2626' : '#374151' }}>
                          {painSide === 'left' ? '左のみ' : painSide === 'right' ? '右のみ' : '両側'}
                        </Text>
                      </Text>
                    )}
                    <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 4 }}>
                      {selectedRegions.map(id => {
                        const region = dermatomeRegions.find(r => r.id === id);
                        return region ? (
                          <View key={id} style={{ backgroundColor: region.color, borderRadius: 4, paddingHorizontal: 6, paddingVertical: 1 }}>
                            <Text style={{ color: '#fff', fontSize: 10 }}>{id}</Text>
                          </View>
                        ) : null;
                      })}
                    </View>
                    {selectedRegions.some(id => highRiskRegionIds.includes(id)) && (
                      <Text style={{ fontSize: 11, color: '#dc2626', fontWeight: '700', marginTop: 6 }}>⚠️ 好発部位あり</Text>
                    )}
                  </View>
                </View>
              </View>
            )}

            {/* チェック内容サマリー */}
            <View style={s.card}>
              <Text style={s.sectionLabel}>📝 チェック内容</Text>
              {painDurVal !== '' && (
                <Text style={{ fontSize: 12, color: '#4b5563', marginTop: 6 }}>
                  痛みの期間：<Text style={{ fontWeight: '700' }}>{painDurVal}{painDurUnit}前から</Text>
                </Text>
              )}
              <View style={{ marginTop: 8, gap: 6 }}>
                {riskFactors.filter(r => checkedRisks[r.id]).map(r => (
                  <View key={r.id} style={{ flexDirection: 'row', alignItems: 'flex-start', gap: 8 }}>
                    <View style={{ width: 8, height: 8, borderRadius: 4, backgroundColor: '#60a5fa', marginTop: 3 }} />
                    <Text style={{ fontSize: 12, color: '#4b5563', flex: 1, lineHeight: 18 }}>{r.label}</Text>
                  </View>
                ))}
                {symptoms.filter(sym => checkedSymptoms[sym.id]).map(sym => (
                  <View key={sym.id} style={{ flexDirection: 'row', alignItems: 'flex-start', gap: 8 }}>
                    <View style={{
                      width: 8, height: 8, borderRadius: 4, marginTop: 3,
                      backgroundColor: sym.tag === '主要症状' ? '#f87171' : '#fb923c',
                    }} />
                    <Text style={{ fontSize: 12, color: '#4b5563', flex: 1, lineHeight: 18 }}>{sym.label}</Text>
                  </View>
                ))}
                {otherSymptoms !== '' && (
                  <View style={{ flexDirection: 'row', alignItems: 'flex-start', gap: 8, marginTop: 2 }}>
                    <View style={{ width: 8, height: 8, borderRadius: 4, backgroundColor: '#9ca3af', marginTop: 3 }} />
                    <Text style={{ fontSize: 12, color: '#4b5563', flex: 1, lineHeight: 18 }}>その他：{otherSymptoms}</Text>
                  </View>
                )}
                {!Object.values(checkedRisks).some(Boolean) && !Object.values(checkedSymptoms).some(Boolean) && !otherSymptoms && (
                  <Text style={{ fontSize: 12, color: '#9ca3af' }}>選択なし</Text>
                )}
              </View>
            </View>

            {/* 免責事項 */}
            <View style={s.disclaimerBox}>
              <Text style={{ fontSize: 11, color: '#a16207', lineHeight: 17 }}>
                ⚠️ このチェッカーは医療診断を代替するものではありません。判定結果は参考情報として使用し、必要に応じて医師・看護師に相談してください。
              </Text>
            </View>

            <TouchableOpacity onPress={handleReset} style={s.resetBtn}>
              <Text style={s.resetBtnText}>🔄 新しいチェックを始める</Text>
            </TouchableOpacity>

            <View style={{ height: 40 }} />
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

// ─────────────────────────────────────────────────────────
// スタイル
// ─────────────────────────────────────────────────────────
const s = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },

  /* ── ヘッダー ── */
  header: {
    backgroundColor: '#2563eb',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  headerSub: {
    color: '#bfdbfe',
    fontSize: 11,
    marginBottom: 2,
  },
  headerTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '700',
  },

  /* ── ステップバー ── */
  stepBarContainer: {
    backgroundColor: '#ffffff',
    paddingVertical: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  stepBarContent: {
    paddingHorizontal: 8,
    gap: 4,
    alignItems: 'center',
  },
  stepItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  stepCircle: {
    width: 24, height: 24,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepCircleOn:  { backgroundColor: '#2563eb' },
  stepCircleOff: { backgroundColor: '#e5e7eb' },
  stepNum: { fontSize: 11, fontWeight: '700' },
  stepLabel: { fontSize: 10 },
  stepLabelOn:  { color: '#2563eb', fontWeight: '700' },
  stepLabelOff: { color: '#9ca3af' },
  stepConnector: { height: 1, width: 10 },
  stepConnectorOn:  { backgroundColor: '#60a5fa' },
  stepConnectorOff: { backgroundColor: '#e5e7eb' },

  /* ── スクロール ── */
  scroll: { flex: 1 },
  scrollContent: {
    padding: 16,
    gap: 16,
  },

  /* ── カード ── */
  card: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
    elevation: 2,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#374151',
    marginBottom: 16,
  },

  /* ── フォーム ── */
  inputLabel: {
    fontSize: 13,
    color: '#4b5563',
    fontWeight: '500',
    marginBottom: 4,
  },
  textInput: {
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 14,
    color: '#111827',
    backgroundColor: '#ffffff',
  },
  blueBox: {
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
    gap: 8,
  },
  sectionLabel: {
    fontSize: 13,
    fontWeight: '700',
    color: '#374151',
    marginBottom: 6,
  },

  /* ── チェックボックス ── */
  checkRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    padding: 12,
    borderRadius: 10,
    borderWidth: 1,
  },
  checkBox: {
    width: 20, height: 20,
    borderRadius: 4,
    borderWidth: 2,
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  checkLabel: {
    fontSize: 13,
    color: '#374151',
    flex: 1,
    lineHeight: 19,
  },

  /* ── タグバッジ ── */
  tagBadge: {
    alignSelf: 'flex-start',
    borderRadius: 10,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  tagTxt: { fontSize: 11, fontWeight: '700' },

  /* ── ボタン ── */
  primaryBtn: {
    backgroundColor: '#2563eb',
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryBtnText: {
    color: '#ffffff',
    fontWeight: '700',
    fontSize: 15,
  },
  outlineBtn: {
    flex: 1,
    borderWidth: 1.5,
    borderColor: '#d1d5db',
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  outlineBtnText: {
    color: '#6b7280',
    fontWeight: '700',
    fontSize: 14,
  },
  rowBtns: {
    flexDirection: 'row',
    gap: 12,
  },
  unitBtn: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1.5,
  },
  unitBtnOn:  { backgroundColor: '#2563eb', borderColor: '#2563eb' },
  unitBtnOff: { backgroundColor: '#ffffff', borderColor: '#d1d5db' },
  unitBtnTxt: { fontSize: 13, fontWeight: '600' },
  sideBtn: {
    flex: 1,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 60,
  },
  sideBtnOn:  { backgroundColor: '#2563eb', borderColor: '#2563eb' },
  sideBtnOff: { backgroundColor: '#ffffff', borderColor: '#d1d5db' },
  sideBtnTxt: { fontSize: 11, fontWeight: '700' },

  /* ── 写真 ── */
  photoPreview: {
    width: '100%',
    height: 240,
    borderRadius: 12,
    backgroundColor: '#000000',
  },
  dangerBtn: {
    borderWidth: 1,
    borderColor: '#fca5a5',
    borderRadius: 10,
    paddingVertical: 10,
    alignItems: 'center',
  },

  /* ── ボックス ── */
  warningBox: {
    backgroundColor: '#fefce8',
    borderWidth: 1,
    borderColor: '#fde68a',
    borderRadius: 10,
    padding: 10,
  },
  disclaimerBox: {
    backgroundColor: '#fefce8',
    borderWidth: 1,
    borderColor: '#fde68a',
    borderRadius: 12,
    padding: 12,
  },
  resetBtn: {
    backgroundColor: '#374151',
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: 'center',
  },
  resetBtnText: {
    color: '#ffffff',
    fontWeight: '700',
    fontSize: 15,
  },
});
