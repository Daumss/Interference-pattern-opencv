import os
import sys
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# 1. Termux:X11 및 디스플레이 자동 시작
def start_x11_environment():
    os.environ["DISPLAY"] = ":1"
    os.system("termux-x11 :1 &")
    time.sleep(3)
    os.system("fluxbox &")
    time.sleep(2)
    os.system("am start --user 0 -n com.termux.x11/com.termux.x11.MainActivity")
    time.sleep(1)


# 2. Termux:X11 디스플레이 자동 종료
def stop_x11_environment():
    print("\nTermux:X11 환경을 정리합니다...")
    
    os.system("am broadcast -a com.termux.x11.ACTION_STOP -p com.termux.x11")
    time.sleep(1)
    os.system("pkill fluxbox")
    os.system("pkill termux-x11")

# 3. DNG 로드 (16bit Linear)
def load_dng_16bit(dng_path):
    print(f"\n[1/4] DNG 파일 로드 중: {dng_path}")
    with rawpy.imread(dng_path) as raw:
        rgb = raw.postprocess(
            gamma=(1, 1),        # 감마 보정 없음 (선형 데이터)
            no_auto_bright=True, # 자동 밝기 조정 끄기
            output_bps=16,       # 16비트 출력 (0~65535)
            use_camera_wb=True
        )

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray16 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    max_val = np.max(gray16)
    print(f"      ▶ 16-bit 최대 밝기: {max_val} / 65535")
    if max_val >= 65000:
        print("센서 포화(Clipping) 발생! 셔터 속도를 올려 노출을 줄이세요.")

    return gray16


# 4. X11 창에서 터치/클릭으로 좌표 선택
def select_points_on_x11(gray16):
    print("\n[2/4] 화면에서 순서대로 4번 클릭하세요:")
    print("      1) 자의 기준 눈금 1 (예: 1cm 지점)")
    print("      2) 자의 기준 눈금 2 (예: 10cm 지점)")
    print("      3) 간섭 무늬 스캔 상단 Y 경계")
    print("      4) 간섭 무늬 스캔 하단 Y 경계")
    print("      * 클릭 완료 후 키보드의 아무 키나 누르세요.")

    #brightness control
    min_val = np.min(gray16)
    max_val = np.max(gray16)
    if max_val > min_val:
        vis_img = ((gray16 - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        vis_img = (gray16 / 256).astype(np.uint8)
    vis_img = cv2.equalizeHist(vis_img)

    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
    pts = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            color = (0, 0, 255) if len(pts) <= 2 else (0, 255, 0)
            cv2.circle(vis_img, (x, y), 4, color, -1)
            cv2.putText(vis_img, f"P{len(pts)} ({x},{y})", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.imshow("Termux:X11 Slit Selector", vis_img)

    cv2.imshow("Termux:X11 Slit Selector", vis_img)
    cv2.setMouseCallback("Termux:X11 Slit Selector", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(pts) < 4:
        raise ValueError("좌표 4개가 모두 지정되지 않았습니다.")

    mark1_x = pts[0][0]
    mark2_x = pts[1][0]
    scan_y_range = [min(pts[2][1], pts[3][1]), max(pts[2][1], pts[3][1])]

    return mark1_x, mark2_x, scan_y_range


# 5. 가우시안 서브픽셀 피팅
def gaussian(x, amp, mean, sigma, offset):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + offset

def fit_subpixel(profile, int_peaks, win=8):
    sub_peaks = []
    x_arr = np.arange(len(profile))

    for p in int_peaks:
        x_min = max(0, p - win)
        x_max = min(len(profile), p + win + 1)
        x_sub = x_arr[x_min:x_max]
        y_sub = profile[x_min:x_max]

        p0 = [np.max(y_sub) - np.min(y_sub), p, 2.0, np.min(y_sub)]
        try:
            popt, _ = curve_fit(gaussian, x_sub, y_sub, p0=p0, maxfev=2000)
            sub_peaks.append(popt[1])
        except RuntimeError:
            sub_peaks.append(float(p))

    return np.array(sub_peaks)


# 6. 메인 파이프라인 (시작 ~ 처리 ~ 종료 감싸기)
def run_analysis(dng_path, real_dist_mm=40.0):
    try:
        # [앞부분] Termux:X11 구동
        start_x11_environment()

        # 1) DNG 로드
        gray16 = load_dng_16bit(dng_path)

        # 2) 터치로 좌표 입력받기
        mark1_x, mark2_x, y_range = select_points_on_x11(gray16)

        # 3) 스케일 계산
        px_dist = abs(mark2_x - mark1_x)
        mm_per_px = real_dist_mm / px_dist
        print(f"\n[3/4] 스케일: 1 px = {mm_per_px:.5f} mm ({px_dist} px 차이)")

        # 4) 1D 스캔 및 피크 추출
        roi = gray16[y_range[0]:y_range[1], :]
        profile_1d = np.mean(roi, axis=0)

        peaks_int, _ = find_peaks(profile_1d, distance=12, prominence=np.max(profile_1d) * 0.08)
        sub_peaks = fit_subpixel(profile_1d, peaks_int)

        # 5) 계산 결과 출력
        diffs_px = np.diff(sub_peaks)
        diffs_mm = diffs_px * mm_per_px

        print("\n[4/4] --- 최종 간섭 무늬 분석 결과 ---")
        for i, (px, mm) in enumerate(zip(diffs_px, diffs_mm)):
            print(f"  · 무늬 {i+1}-{i+2} 간격: {px:.3f} px  ->  {mm:.4f} mm")

        print(f"\n  ★ 평균 무늬 간격(Δx): {np.mean(diffs_mm):.4f} mm (표준편차: ±{np.std(diffs_mm):.4f} mm)")

        # 6) 결과 시각화 창 띄우기
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        img8 = (gray16 / 256).astype(np.uint8)
        ax1.imshow(img8, cmap='gray')
        ax1.axhline(y_range[0], color='r', linestyle='--')
        ax1.axhline(y_range[1], color='r', linestyle='--')
        ax1.set_title("Scanned ROI Range")

        x_mm = np.arange(len(profile_1d)) * mm_per_px
        ax2.plot(x_mm, profile_1d, color='b', label='Intensity')
        ax2.plot(sub_peaks * mm_per_px, profile_1d[np.round(sub_peaks).astype(int)], 'ro', label='Sub-pixel Peaks')
        ax2.set_xlabel("Distance (mm)")
        ax2.set_ylabel("16-bit Intensity")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        print("\n* 결과 그래프 창을 닫으면 프로그램이 자동으로 종료됩니다.")
        plt.show()

    finally:
        # [뒷부분] 작업이 끝나거나 에러가 발생해도 X11 환경을 무조건 정상 종료
        stop_x11_environment()


if __name__ == "__main__":
    # 실행 시 파일 경로 인자 넘기기
    if len(sys.argv) > 1:
        target_dng = sys.argv[1]
    else:
        raise "need dng file path on argv1"
    
    # 기준 눈금 간 실제 거리 (예: 1cm ~ 10cm 눈금을 클릭했다면 90mm)
    REAL_DISTANCE_BETWEEN_MARKS_MM = 90.0

    run_analysis(target_dng, REAL_DISTANCE_BETWEEN_MARKS_MM)
