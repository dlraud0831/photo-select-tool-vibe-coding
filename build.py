import PyInstaller.__main__
import shutil
import os

# v16 최종 이름
APP_NAME = "AI_Photo_Selector_Pro_v16_Final" 

if os.path.exists('build'): shutil.rmtree('build')
if os.path.exists('dist'): shutil.rmtree('dist')
if os.path.exists(f'{APP_NAME}.spec'): os.remove(f'{APP_NAME}.spec')

print(f"🚀 {APP_NAME} 빌드 시작!")

options = [
    'ai_photo_culler.py',
    f'--name={APP_NAME}',
    '--onedir',
    '--noconsole',  # 이제 에러 잡았으니 검은 창 꺼도 됩니다!
    '--clean',
    '--collect-all=rawpy',
    '--hidden-import=piexif',
    '--hidden-import=PIL', # Pillow 강제 포함
]

if os.path.exists('select_icon.ico'):
    options.append('--icon=select_icon.ico')

PyInstaller.__main__.run(options)
print("✅ 빌드 끝!")