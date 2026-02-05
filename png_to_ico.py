# -*- coding: utf-8 -*-
import os
from PIL import Image

def convert_to_ico(input_filename, output_filename):
    print(f"🔄 {input_filename} 변환 시작...")
    
    try:
        # 1. 이미지 열기
        img = Image.open(input_filename)
        
        # 2. 아이콘용 크기 리스트 (큰 것부터 작은 것까지)
        # 윈도우는 상황에 따라 다른 크기의 아이콘을 씁니다.
        icon_sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
        
        # 3. ICO 파일로 저장
        img.save(
            output_filename, 
            format='ICO', 
            sizes=icon_sizes
        )
        print(f"✅ 성공! 아이콘이 생성되었습니다: {output_filename}")
        
    except FileNotFoundError:
        print(f"❌ 오류: '{input_filename}' 파일을 찾을 수 없습니다.")
        print("이미지 파일 이름을 확인하거나 폴더에 파일이 있는지 확인하세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == '__main__':
    # 여기에 변환하고 싶은 파일 이름을 적으세요
    # 예: logo.png, my_picture.jpg 등
    input_file = "logo.png"   # <--- 여기 파일명 수정
    output_file = "my_icon.ico"
    
    convert_to_ico(input_file, output_file)