# -*- coding: utf-8 -*-
import os
from PIL import Image, ImageDraw, ImageFont

def create_hangul_icon_gungsuh():
    print("🎨 '셀렉' 궁서체 아이콘 생성 중...")
    
    # 1. 캔버스 설정 (256x256)
    size = 256
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 2. 배경 그리기 (둥근 사각형 - 흰색)
    # 색상: (255, 255, 255) = White
    bg_color = (255, 255, 255, 255)
    margin = 20
    draw.rounded_rectangle(
        [(margin, margin), (size - margin, size - margin)],
        radius=60,  # 둥근 모서리 정도
        fill=bg_color,
        outline=(0, 0, 0, 255), # 테두리 추가 (검은색, 선택사항)
        width=5                 # 테두리 두께
    )

    # 3. 폰트 설정 (윈도우 기본 궁서체 사용)
    # 보통 C:/Windows/Fonts/gungsuh.ttc 경로에 있습니다.
    font_path = "C:/Windows/Fonts/gungsuh.ttc" 
    
    if not os.path.exists(font_path):
        # ttc가 없으면 ttf로 시도하거나 바탕체(batang)로 대체
        font_path = "C:/Windows/Fonts/batang.ttc"
    
    try:
        # 글자 크기 (아이콘 크기에 맞춰 100으로 설정)
        font = ImageFont.truetype(font_path, 100)
    except IOError:
        print("⚠️ 궁서체 폰트 파일을 찾을 수 없습니다. 기본 폰트로 시도합니다.")
        font = ImageFont.load_default()

    # 4. 글자 쓰기 ("셀렉")
    text = "셀렉"
    
    # 글자 크기 계산해서 정중앙에 배치하기
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 중앙 좌표 계산
    x = (size - text_width) / 2
    y = (size - text_height) / 2 - 10 

    # 글자 색상: 검은색 ("black")
    draw.text((x, y), text, font=font, fill="black")

    # 5. 파일로 저장
    output_filename = 'select_icon.ico'
    img.save(
        output_filename,
        format='ICO',
        sizes=[(256, 256), (128, 128), (64, 64), (32, 32)]
    )
    print(f"✅ 궁서체 아이콘 생성 완료: {output_filename}")

if __name__ == '__main__':
    create_hangul_icon_gungsuh()