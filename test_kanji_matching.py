import cv2
import numpy as np
from proc02_v3_ORB import calculate_kanji_similarity
import os

def test_kanji_matching():
    # 이미지 폴더 경로
    image_dir = 'images'
    
    # 각 문제별로 테스트 실행
    for question_num in range(1, 4):
        question_file = f'question{question_num:02d}.png'
        question_path = os.path.join(image_dir, question_file)
        
        # 문제 이미지 로드
        base_img = cv2.imread(question_path)
        if base_img is None:
            print(f"Error: Cannot load question image {question_path}")
            continue
            
        print(f"\n=== 문제 {question_num} 테스트 ===")
        print(f"문제 이미지: {question_file}")
        print("-" * 50)
        
        # 정답 이미지 테스트
        print("\n[정답 이미지 테스트]")
        for i in range(1, 6):
            answer_file = f'answer{question_num:02d}_{i:02d}.jpg'
            answer_path = os.path.join(image_dir, answer_file)
            
            test_img = cv2.imread(answer_path)
            if test_img is None:
                print(f"Error: Cannot load answer image {answer_path}")
                continue
                
            results = calculate_kanji_similarity(base_img, test_img)
            
            print(f"\n정답 이미지 {i}: {answer_file}")
            print(f"최종 유사도 점수: {results['final_score']:.2f}%")
            print(f"MSE 유사도: {results['mse_score']:.2f}%")
            print(f"획수 유사도: {results['stroke_score']:.2f}%")
            print(f"템플릿 매칭: {results['template_score']:.2f}%")
            print(f"구조적 유사도: {results['structural_score']:.2f}%")
            
            evaluate_results(results, True)
            print("-" * 50)
        
        # 오답 이미지 테스트
        print("\n[오답 이미지 테스트]")
        wrong_file = f'wrongAnswer{question_num:02d}.png'
        wrong_path = os.path.join(image_dir, wrong_file)
        
        test_img = cv2.imread(wrong_path)
        if test_img is None:
            print(f"Error: Cannot load wrong answer image {wrong_path}")
            continue
            
        results = calculate_kanji_similarity(base_img, test_img)
        
        print(f"\n오답 이미지: {wrong_file}")
        print(f"최종 유사도 점수: {results['final_score']:.2f}%")
        print(f"MSE 유사도: {results['mse_score']:.2f}%")
        print(f"획수 유사도: {results['stroke_score']:.2f}%")
        print(f"템플릿 매칭: {results['template_score']:.2f}%")
        print(f"구조적 유사도: {results['structural_score']:.2f}%")
        
        evaluate_results(results, False)
        print("-" * 50)

def evaluate_results(results, expected_high_similarity=True):
    """결과 평가 함수"""
    if expected_high_similarity:
        if results['final_score'] < 90:
            print("Warning: 같은 글자인데 유사도가 너무 낮습니다!")
        if results['stroke_score'] < 90:
            print("Warning: 같은 글자인데 획수 유사도가 너무 낮습니다!")
        if results['template_score'] < 70:
            print("Warning: 같은 글자인데 템플릿 매칭 점수가 너무 낮습니다!")
    else:
        if results['final_score'] > 30:
            print("Warning: 다른 글자인데 유사도가 너무 높습니다!")
        if results['stroke_score'] > 30:
            print("Warning: 다른 글자인데 획수 유사도가 너무 높습니다!")

if __name__ == "__main__":
    test_kanji_matching() 