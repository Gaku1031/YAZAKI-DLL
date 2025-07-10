# -*- mode: python ; coding: utf-8 -*-

import os
import sys

# 基本設定
APP_NAME = "BloodPressureEstimation"
SCRIPT_PATH = "bp_estimation_balanced_20mb.py"

# バランス調整済み除外モジュール（20MB目標）
EXCLUDED_MODULES = [
    # GUI関連（完全除外）
    'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'wx', 'kivy', 'toga',
    
    # 画像処理（不要部分）
    'PIL.ImageTk', 'PIL.ImageQt', 'PIL.ImageDraw2', 'PIL.ImageEnhance',
    'matplotlib', 'seaborn', 'plotly', 'bokeh',
    
    # MediaPipe不要コンポーネント（軽量化）
    'mediapipe.tasks.python.audio',
    'mediapipe.tasks.python.text', 
    'mediapipe.model_maker',
    'mediapipe.python.solutions.pose',
    'mediapipe.python.solutions.hands',
    'mediapipe.python.solutions.holistic',
    'mediapipe.python.solutions.objectron',
    'mediapipe.python.solutions.selfie_segmentation',
    'mediapipe.python.solutions.drawing_utils',
    'mediapipe.python.solutions.drawing_styles',
    
    # TensorFlow軽量化（重い部分のみ除外）
    'tensorflow.lite', 'tensorflow.examples', 'tensorflow.python.tools',
    'tensorflow.python.debug', 'tensorflow.python.profiler',
    'tensorflow.python.distribute', 'tensorflow.python.tpu',
    
    # sklearn軽量化（精度に影響しない部分のみ）
    'sklearn.datasets', 'sklearn.feature_extraction.text',
    'sklearn.neural_network', 'sklearn.gaussian_process',
    'sklearn.cluster', 'sklearn.decomposition',
    'sklearn.feature_selection', 'sklearn.covariance',
    
    # scipy軽量化（signal処理は保持）
    'scipy.ndimage', 'scipy.interpolate', 'scipy.integrate',
    'scipy.optimize', 'scipy.sparse', 'scipy.spatial',
    'scipy.special', 'scipy.linalg', 'scipy.odr',
    
    # 開発・テスト関連
    'numpy.tests', 'scipy.tests', 'sklearn.tests',
    'pandas.tests', 'pandas.plotting', 'pandas.io.formats.style',
    'IPython', 'jupyter', 'notebook', 'jupyterlab',
    'pytest', 'unittest', 'doctest',
    
    # 並行処理（DLLでは不要）
    'multiprocessing', 'concurrent.futures', 'asyncio',
    'threading', 'queue',
    
    # その他重いモジュール
    'email', 'xml', 'html', 'urllib3', 'requests',
    'cryptography', 'ssl', 'socket', 'http'
]

# バランス調整済み隠れたインポート
HIDDEN_IMPORTS = [
    # OpenCV
    'cv2.cv2',
    
    # MediaPipe FaceMesh専用
    'mediapipe.python._framework_bindings',
    'mediapipe.python.solutions.face_mesh',
    'mediapipe.python.solutions.face_mesh_connections',
    
    # NumPy コア
    'numpy.core._methods',
    'numpy.lib.format',
    'numpy.random._pickle',
    
    # joblib（軽量モデル用）
    'joblib.numpy_pickle',
    
    # sklearn（必要最小限）
    'sklearn.tree._tree',
    'sklearn.ensemble._forest',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors._typedefs',
    'sklearn.utils._heap',
    'sklearn.utils._sorting',
    'sklearn.utils._vector_sentinel',
    
    # scipy signal（信号処理用）
    'scipy.signal._max_len_seq_inner',
    'scipy.signal._upfirdn_apply',
    'scipy.signal._sosfilt',
    'scipy.special._ufuncs_cxx',
]

# データファイル
DATAS = [
    ('models', 'models'),
]

# バイナリファイル
BINARIES = []

a = Analysis(
    [SCRIPT_PATH],
    pathex=[],
    binaries=BINARIES,
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDED_MODULES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# バランス調整済みファイル除外
def balanced_file_exclusion(binaries):
    excluded = []
    for name, path, kind in binaries:
        # MediaPipe不要コンポーネント除外
        if any(unused in name.lower() for unused in [
            'pose_landmark', 'hand_landmark', 'holistic', 'objectron', 
            'selfie', 'audio', 'text', 'drawing'
        ]):
            print(f"MediaPipe不要コンポーネント除外: {name}")
            continue
        
        # 大きなTensorFlowコンポーネント除外
        if any(tf_comp in name.lower() for tf_comp in [
            'tensorflow-lite', 'tf_lite', 'tflite', 'tensorboard',
            'tf_debug', 'tf_profiler', 'tf_distribute'
        ]):
            print(f"TensorFlow重いコンポーネント除外: {name}")
            continue
        
        # システムライブラリ除外
        if any(lib in name.lower() for lib in [
            'api-ms-', 'ext-ms-', 'kernel32', 'user32', 'advapi32',
            'ws2_32', 'shell32', 'ole32', 'oleaut32'
        ]):
            continue
        
        # 中程度のファイル除外（7MB以上）
        try:
            if os.path.exists(path) and os.path.getsize(path) > 7 * 1024 * 1024:
                file_size_mb = os.path.getsize(path) / (1024*1024)
                print(f"大きなファイル除外: {name} ({file_size_mb:.1f}MB)")
                continue
        except:
            pass
        
        excluded.append((name, path, kind))
    
    return excluded

a.binaries = balanced_file_exclusion(a.binaries)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,  # 適度なUPX圧縮
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
