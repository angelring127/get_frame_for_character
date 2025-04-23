from PIL import Image, ImageDraw

def create_template_with_frames():
    # 5000x5000 크기의 흰색 이미지 생성
    width = 5000
    height = 5000
    white_image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(white_image)
    
    # 프레임 크기와 간격 설정
    frame_size = 130
    gap = 20  # 프레임 간 간격
    
    # 최대 500개의 프레임을 배치하기 위한 행과 열 계산
    total_frames = 484
    # 정사각형에 가깝게 배치하기 위해 제곱근 사용
    cols = int(total_frames ** 0.5)
    rows = (total_frames + cols - 1) // cols  # 올림 나눗셈
    
    # 실제 사용할 프레임 개수
    actual_frames = min(rows * cols, total_frames)
    
    # 시작 위치 계산 (중앙 정렬)
    start_x = (width - (cols * (frame_size + gap) - gap)) // 2
    start_y = (height - (rows * (frame_size + gap) - gap)) // 2
    
    # 프레임 그리기
    frame_count = 0
    for row in range(rows):
        for col in range(cols):
            if frame_count >= actual_frames:
                break
                
            # 프레임의 좌상단 좌표
            x1 = start_x + col * (frame_size + gap)
            y1 = start_y + row * (frame_size + gap)
            # 프레임의 우하단 좌표
            x2 = x1 + frame_size
            y2 = y1 + frame_size
            
            # 검은색 테두리 그리기 (두께 2픽셀)
            draw.rectangle([x1, y1, x2, y2], outline='black', width=2)
            frame_count += 1
    
    # 4번째 줄 하단에 검은 라인 추가 (더 두껍고 길게)
    line_y = start_y + 3 * (frame_size + gap) + frame_size + gap//2
    draw.line([(start_x - 400, line_y), (start_x + cols * (frame_size + gap) + 400, line_y)], fill='black', width=3)
    
    # 이미지 저장
    white_image.save('template_with_frames.png')
    print(f"{width}x{height} 크기의 이미지에 {rows}x{cols}개의 프레임이 추가되었습니다.")
    print(f"총 {frame_count}개의 프레임이 그려졌습니다.")
    print(f"프레임 간 간격: {gap}픽셀")

if __name__ == "__main__":
    create_template_with_frames()
