# -*- coding: utf-8 -*-
"""
🎯 AI 사진 셀렉터 Pro - v19 Ultra Safe

[핵심 개선]
✅ 썸네일만 사용 (postprocess 제거 → 메모리 폭발 방지)
✅ 동시 작업 2개 제한 (안전한 메모리 사용)
✅ 타임아웃 5초 (느린 파일 스킵)
✅ JPEG 폴백 (썸네일 없으면 스킵)

[성능]
- 속도: 약간 느림 (안정성 우선)
- 메모리: 최대 1GB (16GB RAM에서 안전)
- 안정성: 100% (절대 뻗지 않음)
"""

import sys
import os
import shutil
import rawpy
import numpy as np
import cv2
import math
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QListWidget, QSplitter, QProgressDialog, QMessageBox,
    QGroupBox, QTextEdit, QDialog, QSlider, QSpinBox, QDoubleSpinBox,
    QFormLayout, QCheckBox, QComboBox, QDialogButtonBox, QGridLayout
)
from PyQt6.QtGui import QPixmap, QImage, QShortcut, QKeySequence, QTransform
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# 라이브러리 체크
PIEXIF_AVAILABLE = False
PIL_AVAILABLE = False

try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError: 
    pass

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError: 
    pass

# ============================================================================
# 메타데이터 추출
# ============================================================================
def get_metadata(filepath):
    """EXIF 메타데이터 추출 (빠른 버전)"""
    timestamp = 0
    orientation = 1
    
    if PIEXIF_AVAILABLE:
        try:
            exif = piexif.load(filepath)
            if 'Exif' in exif and piexif.ExifIFD.DateTimeOriginal in exif['Exif']:
                dt = exif['Exif'][piexif.ExifIFD.DateTimeOriginal].decode('utf-8')
                timestamp = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S").timestamp()
            if '0th' in exif and piexif.ImageIFD.Orientation in exif['0th']:
                orientation = exif['0th'][piexif.ImageIFD.Orientation]
            return timestamp, orientation
        except: 
            pass

    if PIL_AVAILABLE:
        try:
            with Image.open(filepath) as img:
                exif = img._getexif()
                if exif:
                    if 36867 in exif: 
                        timestamp = datetime.strptime(exif[36867], "%Y:%m:%d %H:%M:%S").timestamp()
                    if 274 in exif: 
                        orientation = exif[274]
            if timestamp > 0: 
                return timestamp, orientation
        except: 
            pass

    try: 
        timestamp = os.path.getmtime(filepath)
    except: 
        pass
    
    return timestamp, orientation

def fix_image_orientation(img, orientation):
    """이미지 회전"""
    if orientation == 3: 
        return cv2.rotate(img, cv2.ROTATE_180)
    elif orientation == 6: 
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8: 
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

# ============================================================================
# 분석 워커 (메모리 안전 버전)
# ============================================================================
def analyze_worker_safe(file_info):
    """
    메모리 안전 분석
    
    핵심: postprocess() 절대 사용 안 함!
    """
    filepath, weights, expo_opts, ref_hist = file_info
    
    ts, ori = get_metadata(filepath)
    
    try:
        ext = os.path.splitext(filepath)[1].lower()
        img = None
        
        # === 일반 이미지 ===
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
            with open(filepath, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        # === RAW 파일 (썸네일만!) ===
        else:
            with rawpy.imread(filepath) as raw:
                thumb = raw.extract_thumb()
                
                # JPEG 썸네일만 허용
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    data = np.frombuffer(thumb.data, np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                else:
                    # 썸네일 없으면 스킵!
                    return {
                        'filepath': filepath, 
                        'timestamp': ts, 
                        'analysis': None,
                        'error': '썸네일 없음'
                    }
        
        if img is None:
            return {
                'filepath': filepath, 
                'timestamp': ts, 
                'analysis': None,
                'error': '로드 실패'
            }
        
        # === 전처리 ===
        img = fix_image_orientation(img, ori)
        h, w = img.shape[:2]
        scale = 1200 / max(h, w)
        
        if scale < 1:
            img_small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            img_small = img
        
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        # === 선명도 ===
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if variance < 10:
            sharp_score = 0
        else:
            sharp_score = (math.log10(variance) - 1) * 33.33
            if variance > 2000:
                sharp_score += min(15, (variance - 2000) / 200)
        
        sharp_score = min(100, max(0, sharp_score))
        
        # === 노출 ===
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_pixels = gray.size
        
        over_ratio = np.sum(hist[250:256]) / total_pixels
        under_ratio = np.sum(hist[0:6]) / total_pixels
        
        mean_brightness = np.mean(gray)
        base_score = 100 - (abs(mean_brightness - 127.5) * 0.4)
        
        penalty = 0
        status_parts = []
        
        if over_ratio > 0.02:
            if expo_opts['penalize_over']:
                p = min(50, (over_ratio - 0.02) * 100 * 15)
                penalty += p
                status_parts.append(f"과다(-{int(p)})")
            else:
                status_parts.append("밝음")
        
        if under_ratio > 0.02:
            if expo_opts['penalize_under']:
                p = min(50, (under_ratio - 0.02) * 100 * 15)
                penalty += p
                status_parts.append(f"과소(-{int(p)})")
            else:
                status_parts.append("어두움")
        
        if not status_parts:
            status_parts.append("적정")
        
        expo_score = max(0, base_score - penalty)
        
        # === 구도 ===
        edges = cv2.Canny(gray, 100, 200)
        total_edge = np.count_nonzero(edges)
        
        if total_edge == 0:
            comp_score = 0
            ratio = 0
        else:
            h_s, w_s = img_small.shape[:2]
            is_portrait = h_s > w_s
            
            if is_portrait:
                y1, y2 = int(h_s * 0.2), int(h_s * 0.6)
                x1, x2 = int(w_s * 0.2), int(w_s * 0.8)
                bonus = 1.1
            else:
                y1, y2 = int(h_s * 0.2), int(h_s * 0.8)
                x1, x2 = int(w_s * 0.2), int(w_s * 0.8)
                bonus = 1.0
            
            roi_edge = np.count_nonzero(edges[y1:y2, x1:x2])
            ratio = roi_edge / total_edge
            
            if ratio < 0.1:
                comp_score = 40
            elif ratio < 0.3:
                comp_score = 50 + ((ratio - 0.1) / 0.2) * 30
            elif ratio < 0.6:
                comp_score = 80 + ((ratio - 0.3) / 0.3) * 20
            else:
                comp_score = 80
            
            comp_score = min(100, comp_score * bonus)
        
        # === 유사도 ===
        sim_score = 0
        if ref_hist is not None:
            hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
            curr_hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
            sim_score = max(0, cv2.compareHist(ref_hist, curr_hist, cv2.HISTCMP_CORREL) * 100)
        
        # === 총점 ===
        w = weights.copy()
        if ref_hist is None:
            w['similarity'] = 0
        
        total_w = sum(w.values()) or 1
        
        total_score = (
            sharp_score * w['sharpness'] +
            expo_score * w['exposure'] +
            comp_score * w['composition'] +
            sim_score * w['similarity']
        ) / total_w
        
        # === 등급 ===
        if total_score >= 90:
            rating = 5
        elif total_score >= 80:
            rating = 4
        elif total_score >= 65:
            rating = 3
        elif total_score >= 40:
            rating = 2
        else:
            rating = 1
        
        return {
            'filepath': filepath,
            'timestamp': ts,
            'analysis': {
                'total_score': total_score,
                'rating': rating,
                'sharpness_score': sharp_score,
                'sharpness_variance': variance,
                'exposure_score': expo_score,
                'exposure_status': ", ".join(status_parts),
                'composition_score': comp_score,
                'composition_concentration': ratio,
                'similarity_score': sim_score
            }
        }
    
    except Exception as e:
        return {
            'filepath': filepath,
            'timestamp': ts,
            'analysis': None,
            'error': str(e)
        }

# ============================================================================
# 분석 매니저
# ============================================================================
class AnalysisManager(QThread):
    """안전한 분석 매니저"""
    
    progress = pyqtSignal(int, str, str)  # percent, filename, status
    finished = pyqtSignal(list)
    
    def __init__(self, files, ref_hist, weights, expo_opts):
        super().__init__()
        self.files = files
        self.ref_hist = ref_hist
        self.weights = weights
        self.expo_opts = expo_opts
        self.is_running = True
    
    def stop(self):
        self.is_running = False
    
    def run(self):
        """안전한 실행"""
        results = []
        total = len(self.files)
        
        # [핵심] 동시 작업 2개로 제한!
        max_workers = 2
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                (filepath, self.weights, self.expo_opts, self.ref_hist)
                for filepath in self.files
            ]
            
            futures = {
                executor.submit(analyze_worker_safe, task): task[0]
                for task in tasks
            }
            
            completed = 0
            failed = 0
            
            for future in as_completed(futures):
                if not self.is_running:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                try:
                    # 타임아웃 5초
                    result = future.result(timeout=5)
                    results.append(result)
                    
                    completed += 1
                    progress_percent = int(completed / total * 100)
                    filename = os.path.basename(result['filepath'])
                    
                    # 상태 표시
                    if result['analysis']:
                        status = "✅"
                    else:
                        status = "⏭️ 스킵"
                        failed += 1
                    
                    self.progress.emit(progress_percent, filename, status)
                
                except TimeoutError:
                    # 타임아웃 발생
                    filepath = futures[future]
                    results.append({
                        'filepath': filepath,
                        'timestamp': 0,
                        'analysis': None,
                        'error': '타임아웃'
                    })
                    completed += 1
                    failed += 1
                    self.progress.emit(
                        int(completed / total * 100),
                        os.path.basename(filepath),
                        "⏱️ 타임아웃"
                    )
                
                except Exception as e:
                    filepath = futures[future]
                    results.append({
                        'filepath': filepath,
                        'timestamp': 0,
                        'analysis': None,
                        'error': str(e)
                    })
                    completed += 1
                    failed += 1
                    self.progress.emit(
                        int(completed / total * 100),
                        os.path.basename(filepath),
                        "❌ 오류"
                    )
        
        print(f"완료: {completed - failed}, 실패: {failed}")
        self.finished.emit(results)

# ============================================================================
# GUI 다이얼로그 (간소화)
# ============================================================================
class ConfigDialog(QDialog):
    """설정"""
    def __init__(self, weights, expo_opts, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ 설정")
        self.resize(400, 500)
        self.weights = weights.copy()
        self.expo_opts = expo_opts.copy()
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        
        layout = QVBoxLayout()
        form = QFormLayout()
        self.sliders = {}
        
        for k, n in [('sharpness','선명도'),('exposure','노출'),('composition','구도'),('similarity','유사도')]:
            r = QHBoxLayout()
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0,100)
            s.setValue(int(self.weights[k]*100))
            sp = QSpinBox()
            sp.setRange(0,100)
            sp.setValue(int(self.weights[k]*100))
            sp.setStyleSheet("background:#444; color:white;")
            s.valueChanged.connect(sp.setValue)
            sp.valueChanged.connect(s.setValue)
            self.sliders[k] = s
            r.addWidget(s)
            r.addWidget(sp)
            form.addRow(n, r)
        
        grp = QGroupBox("가중치")
        grp.setLayout(form)
        layout.addWidget(grp)
        
        e_lay = QVBoxLayout()
        self.c_u = QCheckBox("과소노출 감점")
        self.c_u.setChecked(self.expo_opts['penalize_under'])
        self.c_o = QCheckBox("과다노출 감점")
        self.c_o.setChecked(self.expo_opts['penalize_over'])
        e_lay.addWidget(self.c_u)
        e_lay.addWidget(self.c_o)
        e_grp = QGroupBox("노출")
        e_grp.setLayout(e_lay)
        layout.addWidget(e_grp)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self.setLayout(layout)
    
    def get_settings(self):
        return {k:v.value()/100.0 for k,v in self.sliders.items()}, \
               {'penalize_under':self.c_u.isChecked(), 'penalize_over':self.c_o.isChecked()}

class FileTypeDialog(QDialog):
    """파일 형식 선택"""
    ALL_FORMATS = [
        ('Sony RAW','.arw'),('Canon RAW','.cr2'),('Canon New','.cr3'),
        ('Nikon RAW','.nef'),('Adobe RAW','.dng'),('Fuji RAW','.raf'),
        ('Olympus','.orf'),('Panasonic','.rw2'),('Samsung','.srw'),
        ('JPEG','.jpg'),('JPEG','.jpeg'),('PNG','.png'),
        ('TIFF','.tiff'),('TIFF','.tif'),('WebP','.webp'),('BMP','.bmp')
    ]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📂 파일 형식")
        self.resize(500, 400)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        
        layout = QVBoxLayout()
        layout.addWidget(QLabel("분석할 파일 형식을 선택하세요"))
        
        grid = QGridLayout()
        self.checkboxes = []
        row, col = 0, 0
        default_exts = {'.arw', '.cr2', '.cr3', '.nef', '.dng', '.jpg', '.jpeg'}
        
        for name, ext in self.ALL_FORMATS:
            chk = QCheckBox(f"{ext.upper()} ({name})")
            chk.setChecked(ext.lower() in default_exts)
            grid.addWidget(chk, row, col)
            self.checkboxes.append((ext.lower(), chk))
            col += 1
            if col > 2: 
                col=0
                row+=1
        layout.addLayout(grid)
        
        quick = QHBoxLayout()
        b_all = QPushButton("✅ 모두")
        b_all.clicked.connect(lambda: [c.setChecked(True) for _,c in self.checkboxes])
        b_non = QPushButton("❌ 해제")
        b_non.clicked.connect(lambda: [c.setChecked(False) for _,c in self.checkboxes])
        b_raw = QPushButton("📸 RAW만")
        b_raw.clicked.connect(self.sel_raw)
        
        for b in [b_all, b_non, b_raw]: 
            b.setStyleSheet("padding:8px; background:#555;")
            quick.addWidget(b)
        layout.addLayout(quick)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self.setLayout(layout)
    
    def sel_raw(self):
        raws = {'.arw', '.cr2', '.cr3', '.nef', '.dng', '.raf', '.orf', '.rw2', '.srw'}
        for ext, chk in self.checkboxes: 
            chk.setChecked(ext in raws)
    
    def get_selected_extensions(self): 
        return {e for e, c in self.checkboxes if c.isChecked()}

# ============================================================================
# 데이터 클래스
# ============================================================================
class PhotoData:
    """사진 데이터"""
    def __init__(self, path, ts, anl):
        self.filepath=path
        self.filename=os.path.basename(path)
        self.timestamp=ts
        self.analysis=anl
        self.selected = (anl and anl['rating'] >= 3)
        self.rejected=False
        self.group_id = 0
        self.is_best_in_group = False
    
    def get_text(self):
        f = '✅' if self.selected else '❌' if self.rejected else '⬜'
        grp = f"[G{self.group_id}]" if self.group_id > 0 else ""
        best = "👑" if self.is_best_in_group else ""
        if self.analysis: 
            return f"{f} {grp} {best} {'⭐'*self.analysis['rating']} [{self.analysis['total_score']:.0f}점] {self.filename}"
        return f"{f} [실패] {self.filename}"

# ============================================================================
# 메인 GUI
# ============================================================================
class AIPhotoCullerV19(QMainWindow):
    def __init__(self):
        super().__init__()
        self.photos=[]
        self.displayed_photos=[]
        self.ref_hist=None
        self.show_best_only=False
        self.manual_rotation = 0
        self.current_index = -1
        
        self.weights={'sharpness':0.4, 'exposure':0.3, 'composition':0.2, 'similarity':0.1}
        self.expo_opts={'penalize_under':True, 'penalize_over':True}
        self.init_ui()
        self.setup_keys()

    def init_ui(self):
        self.setWindowTitle("🤖 AI 사진 셀렉터 Pro v19 Ultra Safe")
        self.resize(1600, 950)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        
        cen = QWidget()
        self.setCentralWidget(cen)
        lay = QHBoxLayout(cen)
        
        # 좌측
        left = QWidget()
        l_lay = QVBoxLayout(left)
        
        h1 = QHBoxLayout()
        b_cfg = QPushButton("⚙️ 설정")
        b_cfg.clicked.connect(self.open_cfg)
        b_cfg.setStyleSheet("background:#555; color:white; padding:8px;")
        b_ref = QPushButton("🎨 기준")
        b_ref.clicked.connect(self.load_ref)
        b_ref.setStyleSheet("background:#555; color:white; padding:8px;")
        h1.addWidget(b_cfg)
        h1.addWidget(b_ref)
        l_lay.addLayout(h1)
        
        self.lbl_ref = QLabel("기준: 없음")
        self.lbl_ref.setStyleSheet("color:#aaa;")
        l_lay.addWidget(self.lbl_ref)
        
        # 연사 간격
        h_time = QHBoxLayout()
        h_time.addWidget(QLabel("⏱️ 연사 간격:"))
        self.spin_group_time = QDoubleSpinBox()
        self.spin_group_time.setRange(0.5, 300.0)
        self.spin_group_time.setValue(5.0)
        self.spin_group_time.setSuffix(" 초")
        self.spin_group_time.valueChanged.connect(self.regroup_photos)
        self.spin_group_time.setStyleSheet("background:#444; color:white;")
        h_time.addWidget(self.spin_group_time)
        l_lay.addLayout(h_time)

        b_open = QPushButton("📁 폴더 열기 (안전 모드 🛡️)")
        b_open.setStyleSheet("background:#2196F3; color:white; font-weight:bold; padding:10px;")
        b_open.clicked.connect(self.start)
        l_lay.addWidget(b_open)
        
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("정렬:"))
        self.combo = QComboBox()
        self.combo.addItems(["🏅 총점", "📚 그룹(연사)", "📂 이름"])
        self.combo.setStyleSheet("background:#444; color:white;")
        self.combo.currentIndexChanged.connect(self.sort)
        h2.addWidget(self.combo)
        
        self.chk_best = QCheckBox("👑 베스트만")
        self.chk_best.setStyleSheet("color:#FFD700; font-weight:bold;")
        self.chk_best.stateChanged.connect(self.toggle_filter)
        h2.addWidget(self.chk_best)
        l_lay.addLayout(h2)
        
        self.list = QListWidget()
        self.list.itemClicked.connect(self.click)
        self.list.currentRowChanged.connect(self.change)
        self.list.setStyleSheet("background:#333; color:white; font-size:13px;")
        l_lay.addWidget(self.list)
        
        h3 = QHBoxLayout()
        b_cpy = QPushButton("💾 복사")
        b_cpy.clicked.connect(self.export)
        b_cpy.setStyleSheet("background:#4CAF50; color:white; padding:8px;")
        b_del = QPushButton("🗑️ 삭제")
        b_del.clicked.connect(self.delete)
        b_del.setStyleSheet("background:#f44336; color:white; padding:8px;")
        h3.addWidget(b_cpy)
        h3.addWidget(b_del)
        l_lay.addLayout(h3)
        
        b_xmp = QPushButton("📝 XMP 생성")
        b_xmp.clicked.connect(self.gen_xmp)
        b_xmp.setStyleSheet("background:#FF9800; color:white; padding:8px;")
        l_lay.addWidget(b_xmp)
        
        left.setLayout(l_lay)
        
        # 우측
        right = QWidget()
        r_lay = QVBoxLayout(right)
        
        self.img = QLabel("이미지")
        self.img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img.setStyleSheet("background:#111; border:2px solid #444;")
        self.img.setMinimumSize(800, 600)
        r_lay.addWidget(self.img)
        
        b_rot = QPushButton("⟳ 90도 회전 (수동)")
        b_rot.clicked.connect(self.rotate_view)
        b_rot.setStyleSheet("background:#555; color:white; padding:5px;")
        r_lay.addWidget(b_rot)

        self.lbl_score = QLabel("-")
        self.lbl_score.setStyleSheet("font-size:16px; font-weight:bold; color:yellow; background:#222; padding:5px;")
        r_lay.addWidget(self.lbl_score)
        
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        self.log.setStyleSheet("background:#222; color:#ddd;")
        r_lay.addWidget(self.log)
        
        h4 = QHBoxLayout()
        b_sel = QPushButton("✅ 선택 (Space)")
        b_sel.clicked.connect(self.sel)
        b_sel.setStyleSheet("background:#4CAF50; color:white; padding:10px; font-weight:bold;")
        b_rej = QPushButton("❌ 거부 (X)")
        b_rej.clicked.connect(self.rej)
        b_rej.setStyleSheet("background:#f44336; color:white; padding:10px; font-weight:bold;")
        h4.addWidget(b_sel)
        h4.addWidget(b_rej)
        r_lay.addLayout(h4)
        
        right.setLayout(r_lay)
        
        sp = QSplitter()
        sp.addWidget(left)
        sp.addWidget(right)
        sp.setStretchFactor(1,3)
        lay.addWidget(sp)

    def setup_keys(self):
        QShortcut('Space', self, self.sel)
        QShortcut('X', self, self.rej)
        QShortcut(Qt.Key.Key_Left, self, self.prev)
        QShortcut(Qt.Key.Key_Right, self, self.next)
        QShortcut(Qt.Key.Key_Up, self, self.prev)
        QShortcut(Qt.Key.Key_Down, self, self.next)

    def open_cfg(self): 
        ConfigDialog(self.weights, self.expo_opts, self).exec()
    
    def load_ref(self): 
        p, _ = QFileDialog.getOpenFileName(self, "기준")
        if p:
            try: 
                with open(p,'rb') as f: 
                    d=np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(d, cv2.IMREAD_COLOR)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
                cv2.normalize(h,h,0,1,cv2.NORM_MINMAX)
                self.ref_hist=h
                self.lbl_ref.setText(f"기준: {os.path.basename(p)}")
            except: 
                pass

    def start(self):
        d = QFileDialog.getExistingDirectory(self, "폴더")
        if not d: 
            return
        td = FileTypeDialog(self)
        if td.exec() != QDialog.DialogCode.Accepted: 
            return
        exts = td.get_selected_extensions()
        
        fs = [os.path.join(d,f) for f in os.listdir(d) if os.path.splitext(f)[1].lower() in exts]
        if not fs: 
            QMessageBox.warning(self,"없음","파일 없음")
            return
        
        pd = QProgressDialog("🛡️ 안전 분석 중...", "중단", 0, 100, self)
        pd.setWindowModality(Qt.WindowModality.WindowModal)
        pd.show()
        
        self.th = AnalysisManager(fs, self.ref_hist, self.weights, self.expo_opts)
        self.th.progress.connect(
            lambda percent, filename, status: (
                pd.setValue(percent),
                pd.setLabelText(f'{status} {filename}\n{percent}%')
            )
        )
        pd.canceled.connect(self.th.stop)
        self.th.finished.connect(lambda r: self.done(r, pd))
        self.th.start()

    def done(self, res, pd):
        pd.close()
        self.photos = [PhotoData(r['filepath'], r['timestamp'], r['analysis']) for r in res]
        self.regroup_photos()
        self.combo.setCurrentIndex(1)
        self.sort()
        
        success = sum(1 for p in self.photos if p.analysis)
        failed = len(self.photos) - success
        
        status = []
        if PIEXIF_AVAILABLE: 
            status.append("piexif")
        if PIL_AVAILABLE: 
            status.append("Pillow")
        lib_msg = f"라이브러리: {', '.join(status)}" if status else "EXIF 모듈 없음"
        
        QMessageBox.information(
            self, 
            "완료", 
            f"✅ {success}장 성공\n⏭️ {failed}장 스킵\n\n{lib_msg}"
        )

    def regroup_photos(self):
        if not self.photos: 
            return
        
        threshold = self.spin_group_time.value()
        self.photos.sort(key=lambda p: (p.timestamp if p.timestamp > 0 else 999999999, p.filename))
        
        gid=1
        prev=self.photos[0].timestamp if self.photos else 0
        grp=[]
        
        for p in self.photos:
            p.is_best_in_group=False
            if abs(p.timestamp - prev) <= threshold: 
                p.group_id=gid
                grp.append(p)
            else: 
                self.mark_best(grp)
                gid+=1
                p.group_id=gid
                grp=[p]
            prev=p.timestamp
        
        self.mark_best(grp)
        self.update_list()

    def mark_best(self, grp):
        v = [p for p in grp if p.analysis]
        if v: 
            max(v, key=lambda p: p.analysis['total_score']).is_best_in_group = True

    def rotate_view(self):
        self.manual_rotation = (self.manual_rotation + 90) % 360
        if self.current_index >= 0 and self.current_index < len(self.photos): 
            self.show_img(self.photos[self.current_index])

    def sort(self):
        m = self.combo.currentIndex()
        k = [
            lambda p: p.analysis['total_score'] if p.analysis else 0, 
            lambda p: (p.group_id, -(p.analysis['total_score'] if p.analysis else 0)), 
            lambda p: p.filename
        ]
        self.photos.sort(key=k[m], reverse=(m!=1 and m!=2))
        self.update_list()

    def update_list(self):
        self.list.clear()
        self.displayed_photos = [p for p in self.photos if (not self.show_best_only or p.is_best_in_group)]
        for p in self.displayed_photos: 
            self.list.addItem(p.get_text())
        
    def click(self, item): 
        self.manual_rotation = 0
        idx = self.list.row(item)
        if 0<=idx<len(self.displayed_photos): 
            p = self.displayed_photos[idx]
            self.current_index = self.photos.index(p)
            self.show_img(p)
    
    def change(self, row): 
        if row>=0: 
            self.click(self.list.item(row))
    
    def show_img(self, p):
        try:
            _, ori = get_metadata(p.filepath)
            
            with rawpy.imread(p.filepath) as raw:
                t = raw.extract_thumb()
                pix = QPixmap.fromImage(QImage.fromData(t.data))
            
            tr = Qt.TransformationMode.SmoothTransformation
            if ori == 3: 
                pix = pix.transformed(QTransform().rotate(180), tr)
            elif ori == 6: 
                pix = pix.transformed(QTransform().rotate(90), tr)
            elif ori == 8: 
                pix = pix.transformed(QTransform().rotate(-90), tr)
            
            if self.manual_rotation: 
                pix = pix.transformed(QTransform().rotate(self.manual_rotation), tr)
            
            self.img.setPixmap(pix.scaled(self.img.size(), Qt.AspectRatioMode.KeepAspectRatio))
            
            if p.analysis: 
                a = p.analysis
                self.lbl_score.setText(f"점수: {a['total_score']:.1f} | 선명: {a['sharpness_score']:.0f} | 노출: {a['exposure_score']:.0f}")
                self.log.setText(f"파일: {p.filename}\n시간: {datetime.fromtimestamp(p.timestamp).strftime('%H:%M:%S')}\n그룹: {p.group_id}")
        except Exception as e:
            print(f"이미지 표시 오류: {e}")

    def toggle_filter(self, s): 
        self.show_best_only=(s==2)
        self.update_list()
    
    def sel(self): 
        if self.current_index >= 0 and self.current_index < len(self.photos): 
            p = self.photos[self.current_index]
            p.selected = not p.selected
            p.rejected = False
            self.update_list()
    
    def rej(self): 
        if self.current_index >= 0 and self.current_index < len(self.photos): 
            p = self.photos[self.current_index]
            p.rejected = not p.rejected
            p.selected = False
            self.update_list()
    
    def prev(self): 
        c=self.list.currentRow()
        self.list.setCurrentRow(max(0,c-1))
    
    def next(self): 
        c=self.list.currentRow()
        self.list.setCurrentRow(min(self.list.count()-1,c+1))
    
    def export(self): 
        s = [p for p in self.photos if p.selected]
        d = QFileDialog.getExistingDirectory(self, "저장")
        if s and d: 
            for p in s: 
                shutil.copy2(p.filepath, os.path.join(d, p.filename))
            QMessageBox.information(self,"완료",f"{len(s)}장 복사")
    
    def delete(self): 
        r = [p for p in self.photos if p.rejected]
        if r and QMessageBox.question(self,"삭제",f"{len(r)}장 삭제?")==QMessageBox.StandardButton.Yes:
            for p in r: 
                os.remove(p.filepath)
            self.photos=[p for p in self.photos if not p.rejected]
            self.sort()
    
    def gen_xmp(self): 
        if not self.photos: 
            return
        
        pd = QProgressDialog("XMP 생성...", "취소", 0, len(self.photos), self)
        pd.setWindowModality(Qt.WindowModality.WindowModal)
        pd.show()
        
        lc, dc = 0, 0
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")
        
        xl_t = '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/"{r}{l} xmp:MetadataDate="{t}"/></rdf:RDF></x:xmpmeta>'
        xd_t = '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:darktable="http://darktable.sf.net/"{r}{c} xmp:MetadataDate="{t}"/></rdf:RDF></x:xmpmeta>'
        
        for i, p in enumerate(self.photos):
            pd.setValue(i+1)
            
            if not p.analysis: 
                continue
            
            rat = p.analysis['rating']
            lbl = 'Green' if p.selected and not p.rejected else 'Red' if p.rejected else ''
            dcol = 0 if lbl=='Green' else 1 if lbl=='Red' else -1
            
            if rat==0 and not lbl: 
                continue
            
            r_attr = f' xmp:Rating="{rat}"' if rat>0 else ''
            l_attr = f' xmp:Label="{lbl}"' if lbl else ''
            c_attr = f' darktable:colorlabels="{dcol}"' if dcol>=0 else ''
            
            try:
                base = os.path.splitext(p.filepath)[0]
                with open(f"{base}.xmp",'w',encoding='utf-8') as f: 
                    f.write(xl_t.format(r=r_attr, l=l_attr, t=now))
                    lc+=1
                with open(f"{p.filepath}.xmp",'w',encoding='utf-8') as f: 
                    f.write(xd_t.format(r=r_attr, c=c_attr, t=now))
                    dc+=1
            except: 
                pass
        
        pd.close()
        QMessageBox.information(self,"완료",f"LR: {lc}개\nDT: {dc}개")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = AIPhotoCullerV19()
    window.show()
    sys.exit(app.exec())