import time
import sys
import numpy as np
from pathlib import Path
import json
import os
import cv2
import tkinter as tk
from tkinter import scrolledtext
import threading

# 导入 pykinect_azure
import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH

# 导入你的模块
from tools import Tools
from module_load import module_load, diagnosis_module_load


class VisualizationWindow:
    """可视化窗口类，用于显示文字结果"""
    def __init__(self, title="Processing Results"):
        self.root = None
        self.text_widget = None
        self.title = title
        self.is_open = False
        
    def open(self):
        """在新线程中打开窗口"""
        if not self.is_open:
            thread = threading.Thread(target=self._create_window, daemon=True)
            thread.start()
            time.sleep(0.5)  # 等待窗口创建
    
    def _create_window(self):
        """创建 Tkinter 窗口"""
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry("600x400")
        
        # 创建滚动文本框
        self.text_widget = scrolledtext.ScrolledText(
            self.root, 
            wrap=tk.WORD,
            width=70,
            height=20,
            font=("Consolas", 10)
        )
        self.text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # 关闭按钮
        btn_close = tk.Button(
            self.root, 
            text="Close", 
            command=self._close_window,
            height=2
        )
        btn_close.pack(pady=5)
        
        self.is_open = True
        self.root.mainloop()
    
    def _close_window(self):
        """关闭窗口"""
        if self.root:
            self.is_open = False
            self.root.quit()
            self.root.destroy()
    
    def update_text(self, text):
        """更新文本内容"""
        if self.root and self.text_widget:
            try:
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(tk.END, text)
                self.text_widget.see(tk.END)
            except:
                pass
    
    def close(self):
        """外部调用关闭"""
        if self.is_open and self.root:
            self.root.after(0, self._close_window)


class KinectPipeline:
    def __init__(self):
        self.tools = Tools()
        self.screenshot_dir = Path("./screenshot")
        self.screenshot_dir.mkdir(exist_ok=True)
        
        self.medical_img_dir = Path("./medical_img")
        self.medical_img_dir.mkdir(exist_ok=True)
        
        self.reports_dir = Path("./generated_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # 加载所有模块
        print("\n=== Loading Modules ===")
        self.modules_dict = module_load()
        print("✓ All modules loaded")
        
        # Kinect 设备
        self.device = None
        
        # 用于帧差检测
        self.prev_frame = None
        
        # 可视化窗口
        self.viz_window = None
        
        # 显示窗口位置
        self.window_positions = {
            'main': (50, 50),
            'monitor': (800, 50),
            'rough': (50, 500),
            'restored': (400, 500),
            'depth': (800, 500)
        }
    
    def open_kinect(self):
        """打开 Kinect Azure 设备"""
        print("\n=== Opening Kinect Azure ===")
        try:
            # 初始化 Kinect 库
            pykinect.initialize_libraries()
            
            # 配置设备
            device_config = pykinect.default_configuration
            device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
            device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
            device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
            
            # 启动设备
            self.device = pykinect.start_device(config=device_config)
            
            print("✓ Kinect Azure opened successfully")
            print(f"  Color Resolution: 1920x1080")
            print(f"  Depth Mode: WFOV 2x2 Binned")
            print(f"  FPS: 30")
            
            # 预热相机（丢弃前几帧）
            print("  Warming up...")
            for i in range(10):
                capture = self.device.update()
                ret, color_image = capture.get_color_image()
                if ret:
                    print(f"    Frame {i+1}/10 captured")
                time.sleep(0.1)
            
            print("✓ Kinect camera ready!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to open Kinect Azure: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def capture_frame(self):
        """捕获一帧彩色图像"""
        if self.device is None:
            return None
        
        try:
            capture = self.device.update()
            ret, color_image = capture.get_color_image()
            
            if ret:
                if color_image.shape[2] == 4:  # BGRA
                    frame = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
                else:
                    frame = color_image
                return frame
            
            return None
            
        except Exception as e:
            if not hasattr(self, '_last_error_time') or time.time() - self._last_error_time > 5:
                print(f"⚠️  Error capturing frame: {e}")
                self._last_error_time = time.time()
            return None
    
    def capture_depth_frame(self):
        """捕获深度图像"""
        if self.device is None:
            return None
        
        try:
            capture = self.device.update()
            ret, depth_image = capture.get_depth_image()
            
            if ret:
                return depth_image
            
            return None
            
        except Exception as e:
            if not hasattr(self, '_last_depth_error_time') or time.time() - self._last_depth_error_time > 5:
                print(f"⚠️  Error capturing depth: {e}")
                self._last_depth_error_time = time.time()
            return None
    
    def show_image_window(self, window_name, image, position=None, max_size=(640, 480)):
        """显示图像窗口，并调整大小和位置"""
        if image is None:
            return
        
        # 调整图像大小以适应屏幕
        h, w = image.shape[:2]
        scale = min(max_size[0]/w, max_size[1]/h)
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            display_img = cv2.resize(image, (new_w, new_h))
        else:
            display_img = image.copy()
        
        # 添加标题信息
        info_text = f"{window_name} [{display_img.shape[1]}x{display_img.shape[0]}]"
        cv2.putText(display_img, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display_img)
        
        # 设置窗口位置
        if position:
            cv2.moveWindow(window_name, position[0], position[1])
    
    def create_text_visualization(self, stage_results):
        """创建文本可视化"""
        text = "=" * 60 + "\n"
        text += "MEDICAL IMAGE ANALYSIS PIPELINE\n"
        text += "=" * 60 + "\n\n"
        
        for i, stage in enumerate(stage_results, 1):
            text += f"[Stage {i}] {stage['name']}\n"
            text += f"Time: {stage['time']:.0f} ms\n"
            text += f"Status: {stage['status']}\n"
            if stage.get('details'):
                text += f"Details: {stage['details']}\n"
            text += "-" * 60 + "\n\n"
        
        return text
    
    def process_frame(self, frame, visualize=True):
        """处理单帧图像的完整流水线（带可视化）"""
        print("\n" + "="*60)
        print("PROCESSING FRAME WITH VISUALIZATION")
        print("="*60)
        
        stage_results = []
        time_start = time.time()
        
        try:
            # Stage 1: 定位显示器
            print("\n[Stage 1] Localizing Monitor...")
            stage_time = time.time()
            
            monitor_photo = self.tools.localize_monitor(
                self.modules_dict['monitor_detect_module'], 
                frame, 
                62
            )
            
            stage_1_time = (time.time() - stage_time) * 1000
            stage_results.append({
                'name': 'Monitor Localization',
                'time': stage_1_time,
                'status': '✓ Success',
                'details': f'Found TV region: {monitor_photo.shape[1]}x{monitor_photo.shape[0]}'
            })
            
            if visualize:
                self.show_image_window('1. Monitor Region', monitor_photo, 
                                      self.window_positions['monitor'], (500, 400))
            
            # Stage 2: 帧差检测
            print("\n[Stage 2] Frame Difference Detection...")
            stage_time = time.time()
            
            if self.prev_frame is not None:
                isSame, percent = self.tools.diff_two_frames(self.prev_frame, monitor_photo)
                stage_2_time = (time.time() - stage_time) * 1000
                if isSame:
                    stage_results.append({
                        'name': 'Frame Difference',
                        'time': stage_2_time,
                        'status': '⊘ SKIP (Same Frame)',
                        'details': f'Similarity: {percent:.2f}%'
                    })
                    print(f"  ** SKIP ** Similarity: {percent:.2f}%")
                    return None, stage_results
                else:
                    stage_results.append({
                        'name': 'Frame Difference',
                        'time': stage_2_time,
                        'status': '✓ New Frame',
                        'details': f'Difference: {100-percent:.2f}%'
                    })
                    print(f"  ** NEW FRAME ** Difference: {100-percent:.2f}%")
            else:
                stage_2_time = (time.time() - stage_time) * 1000
                stage_results.append({
                    'name': 'Frame Difference',
                    'time': stage_2_time,
                    'status': '✓ First Frame',
                    'details': 'No previous frame to compare'
                })
            
            self.prev_frame = monitor_photo
            
            # Stage 3: 定位医学图像
            print("\n[Stage 3] Localizing Medical Image...")
            stage_time = time.time()
            
            rough_medical_img, flag = self.tools.localize_image(monitor_photo)
            
            stage_3_time = (time.time() - stage_time) * 1000
            
            if not flag:
                stage_results.append({
                    'name': 'Medical Image Localization',
                    'time': stage_3_time,
                    'status': '✗ Failed',
                    'details': 'No medical image detected'
                })
                print("  ❌ No medical image detected")
                return None, stage_results
            
            stage_results.append({
                'name': 'Medical Image Localization',
                'time': stage_3_time,
                'status': '✓ Success',
                'details': f'Extracted: {rough_medical_img.shape[1]}x{rough_medical_img.shape[0]}'
            })
            
            if visualize:
                self.show_image_window('2. Rough Medical Image', rough_medical_img,
                                      self.window_positions['rough'], (350, 350))
            
            # Stage 4: 图像修复
            print("\n[Stage 4] Restoring Image Quality...")
            stage_time = time.time()
            
            medical_img = self.tools.restorer(
                rough_medical_img, 
                self.modules_dict['image_restoer_module']
            )
            
            stage_4_time = (time.time() - stage_time) * 1000
            stage_results.append({
                'name': 'Image Restoration',
                'time': stage_4_time,
                'status': '✓ Success',
                'details': f'Enhanced: {medical_img.shape[1]}x{medical_img.shape[0]}'
            })
            
            if visualize:
                self.show_image_window('3. Restored Medical Image', medical_img,
                                      self.window_positions['restored'], (350, 350))
            
            # Stage 5: 部位识别
            print("\n[Stage 5] Identifying Body Part...")
            stage_time = time.time()
            
            PartType = self.tools.identification(
                medical_img,
                self.modules_dict['discrimination_module'][0],
                self.modules_dict['discrimination_module'][1],
                self.modules_dict['discrimination_module'][2]
            )
            
            stage_5_time = (time.time() - stage_time) * 1000
            stage_results.append({
                'name': 'Body Part Identification',
                'time': stage_5_time,
                'status': '✓ Success',
                'details': f'Identified as: {PartType}'
            })
            print(f"  Identified: {PartType}")
            
            # Stage 6: 加载诊断模型
            print(f"\n[Stage 6] Loading Diagnosis Models for {PartType}...")
            stage_time = time.time()
            
            self.modules_dict = diagnosis_module_load(self.modules_dict, PartType)
            
            stage_6_time = (time.time() - stage_time) * 1000
            stage_results.append({
                'name': 'Model Loading',
                'time': stage_6_time,
                'status': '✓ Success',
                'details': f'Loaded models for {PartType}'
            })
            
            # Stage 7: 疾病诊断
            print("\n[Stage 7] Diagnosing...")
            stage_time = time.time()
            
            try:
                classifier_results = self.tools.diagnosis(
                    medical_img,
                    self.modules_dict['diagnosis_module'][PartType],
                    PartType
                )
                
                stage_7_time = (time.time() - stage_time) * 1000
                stage_results.append({
                    'name': 'Disease Diagnosis',
                    'time': stage_7_time,
                    'status': '✓ Success',
                    'details': f'Completed {len(classifier_results)} character report'
                })
                print(f"  Diagnosis completed: {len(classifier_results)} chars")
                
            except Exception as e:
                stage_7_time = (time.time() - stage_time) * 1000
                stage_results.append({
                    'name': 'Disease Diagnosis',
                    'time': stage_7_time,
                    'status': '✗ Failed',
                    'details': str(e)
                })
                classifier_results = f"Diagnosis error: {str(e)}"
                print(f"  ❌ Diagnosis failed: {e}")
            
            # Stage 8: 生成报告
            print("\n[Stage 8] Generating Report...")
            stage_time = time.time()
            
            try:
                generated_report = self.tools.report_generation(medical_img, classifier_results)
                
                stage_8_time = (time.time() - stage_time) * 1000
                stage_results.append({
                    'name': 'Report Generation',
                    'time': stage_8_time,
                    'status': '✓ Success',
                    'details': f'Generated {len(generated_report)} character report'
                })
                print(f"  Report generated: {len(generated_report)} chars")
                
            except Exception as e:
                stage_8_time = (time.time() - stage_time) * 1000
                stage_results.append({
                    'name': 'Report Generation',
                    'time': stage_8_time,
                    'status': '✗ Failed',
                    'details': str(e)
                })
                generated_report = f"Report error\n\nDiagnosis:\n{classifier_results}"
                print(f"  ❌ Report generation failed: {e}")
            
            # Stage 9: 多媒体展示
            print("\n[Stage 9] Multimedia Display...")
            stage_time = time.time()
            
            try:
                self.tools.multimedia_dispaly(generated_report)
                stage_9_time = (time.time() - stage_time) * 1000
                stage_results.append({
                    'name': 'Multimedia Display',
                    'time': stage_9_time,
                    'status': '✓ Success',
                    'details': 'Audio played'
                })
            except Exception as e:
                stage_9_time = (time.time() - stage_time) * 1000
                stage_results.append({
                    'name': 'Multimedia Display',
                    'time': stage_9_time,
                    'status': '⚠ Warning',
                    'details': str(e)
                })
                print(f"  ⚠️  Multimedia display failed: {e}")
            
            # 总结
            total_time = (time.time() - time_start) * 1000
            stage_results.append({
                'name': 'TOTAL PROCESSING',
                'time': total_time,
                'status': '✓ Complete',
                'details': f'{len(stage_results)} stages completed'
            })
            
            print(f"\n✓ Total processing time: {total_time:.0f} ms")
            print("="*60)
            
            # 显示文本结果
            if visualize and self.viz_window:
                summary_text = self.create_text_visualization(stage_results)
                summary_text += "\n" + "="*60 + "\n"
                summary_text += "DIAGNOSIS RESULTS:\n"
                summary_text += "="*60 + "\n"
                summary_text += classifier_results + "\n\n"
                summary_text += "="*60 + "\n"
                summary_text += "GENERATED REPORT:\n"
                summary_text += "="*60 + "\n"
                summary_text += generated_report
                
                self.viz_window.update_text(summary_text)
            
            return {
                'original_frame': frame,
                'monitor_photo': monitor_photo,
                'rough_medical_img': rough_medical_img,
                'medical_img': medical_img,
                'part_type': PartType,
                'diagnosis': classifier_results,
                'report': generated_report,
                'timestamp': time.strftime("%Y%m%d%H%M%S"),
                'stage_results': stage_results
            }, stage_results
            
        except Exception as e:
            print(f"\n❌ Fatal error in process_frame: {e}")
            import traceback
            traceback.print_exc()
            
            stage_results.append({
                'name': 'FATAL ERROR',
                'time': (time.time() - time_start) * 1000,
                'status': '✗ Failed',
                'details': str(e)
            })
            
            return None, stage_results
    
    def save_results(self, results):
        """保存结果"""
        if results is None:
            return
        
        timestamp = results['timestamp']
        part_type = results['part_type']
        
        print(f"\n=== Saving Results ===")
        
        # 1. 保存医学图像
        medical_img_path = self.medical_img_dir / f"{timestamp}_{part_type}.png"
        cv2.imwrite(str(medical_img_path), results['medical_img'])
        print(f"  ✓ Medical image: {medical_img_path}")
        
        # 2. 保存原始捕获
        original_path = self.screenshot_dir / f"capture_{timestamp}.png"
        cv2.imwrite(str(original_path), results['original_frame'])
        print(f"  ✓ Original frame: {original_path}")
        
        # 3. 保存粗糙提取
        rough_path = self.screenshot_dir / f"rough_{timestamp}.png"
        cv2.imwrite(str(rough_path), results['rough_medical_img'])
        print(f"  ✓ Rough extraction: {rough_path}")
        
        # 4. 保存报告到 JSON
        reports_path = self.reports_dir / f"{part_type}.json"
        report_entry = {
            time.strftime("%Y-%m-%d %H:%M:%S"): results['report']
        }
        
        if reports_path.exists():
            with open(reports_path, 'r', encoding='utf-8') as f:
                try:
                    reports = json.load(f)
                except json.JSONDecodeError:
                    reports = []
        else:
            reports = []
        
        reports.append(report_entry)
        
        with open(reports_path, 'w', encoding='utf-8') as f:
            json.dump(reports, f, ensure_ascii=False, indent=4)
        
        print(f"  ✓ Report: {reports_path}")
        print("="*40)
        
        return timestamp
    
    def run(self):
        """运行主循环"""
        if not self.open_kinect():
            return
        
        print("\n=== Kinect Pipeline Started (Visualization Mode) ===")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot only")
        print("  'p' - Process and analyze current frame")
        print("  'a' - Auto mode (process every frame)")
        print("  'd' - Toggle depth map")
        print("  'v' - Toggle visualization")
        print("  'c' - Clear all windows")
        
        # 创建文本可视化窗口
        self.viz_window = VisualizationWindow("Processing Pipeline Results")
        self.viz_window.open()
        
        frame_count = 0
        last_time = time.time()
        fps_display = 0
        processing = False
        auto_mode = False
        show_depth = False
        visualize = True
        error_count = 0
        max_errors = 10
        
        try:
            while True:
                frame = self.capture_frame()
                
                if frame is None:
                    error_count += 1
                    if error_count >= max_errors:
                        print(f"\n❌ Too many consecutive errors ({max_errors}), exiting...")
                        break
                    time.sleep(0.1)
                    continue
                
                error_count = 0
                frame_count += 1
                
                # 计算 FPS
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps_display = frame_count
                    frame_count = 0
                    last_time = current_time
                
                # 显示主画面
                display_frame = frame.copy()
                info_y = 30
                
                # FPS
                cv2.putText(display_frame, f"FPS: {fps_display}", (10, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                info_y += 40
                
                # 相机信息
                cv2.putText(display_frame, "Kinect Azure", 
                            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                info_y += 30
                
                # 模式指示
                if auto_mode:
                    cv2.putText(display_frame, "AUTO MODE", (10, info_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                elif processing:
                    cv2.putText(display_frame, "Processing...", (10, info_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "Ready - Press 'p'", (10, info_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                info_y += 30
                
                # 可视化状态
                viz_status = "ON" if visualize else "OFF"
                cv2.putText(display_frame, f"Visualization: {viz_status}", (10, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # 显示主窗口
                self.show_image_window('0. Kinect Live Feed', display_frame,
                                      self.window_positions['main'], (700, 500))
                
                # 深度图
                if show_depth:
                    depth_frame = self.capture_depth_frame()
                    if depth_frame is not None:
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_frame, alpha=0.03), 
                            cv2.COLORMAP_JET
                        )
                        self.show_image_window('Depth Map', depth_colormap,
                                              self.window_positions['depth'], (400, 400))
                
                # 自动模式处理
                if auto_mode and not processing:
                    processing = True
                    try:
                        results, stages = self.process_frame(frame, visualize=visualize)
                        if results is not None:
                            self.save_results(results)
                            print("✓ Auto processing complete!")
                    except Exception as e:
                        print(f"❌ Auto processing failed: {e}")
                        import traceback
                        traceback.print_exc()
                    processing = False
                
                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                
                elif key == ord('s'):
                    filename = self.screenshot_dir / f"screenshot_{int(time.time())}.png"
                    cv2.imwrite(str(filename), frame)
                    print(f"✓ Screenshot saved: {filename}")
                
                elif key == ord('p') and not processing:
                    processing = True
                    print("\nProcessing frame with visualization...")
                    
                    try:
                        results, stages = self.process_frame(frame, visualize=visualize)
                        if results is not None:
                            self.save_results(results)
                            print("✓ Processing complete!")
                    except Exception as e:
                        print(f"❌ Processing failed: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    processing = False
                
                elif key == ord('a'):
                    auto_mode = not auto_mode
                    mode_str = "ON" if auto_mode else "OFF"
                    print(f"\n{'='*40}")
                    print(f"Auto mode: {mode_str}")
                    print(f"{'='*40}")
                
                elif key == ord('d'):
                    show_depth = not show_depth
                    depth_str = "ON" if show_depth else "OFF"
                    print(f"Depth display: {depth_str}")
                    if not show_depth:
                        cv2.destroyWindow('Depth Map')
                
                elif key == ord('v'):
                    visualize = not visualize
                    viz_str = "ON" if visualize else "OFF"
                    print(f"Visualization: {viz_str}")
                    if not visualize:
                        # 关闭可视化窗口
                        for name in ['1. Monitor Region', '2. Rough Medical Image', 
                                    '3. Restored Medical Image']:
                            try:
                                cv2.destroyWindow(name)
                            except:
                                pass
                
                elif key == ord('c'):
                    print("Clearing all windows...")
                    cv2.destroyAllWindows()
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            print("\nCleaning up...")
            if self.viz_window:
                self.viz_window.close()
            if self.device is not None:
                self.device.close()
            cv2.destroyAllWindows()
            print("Done!")

# 主程序入口
if __name__ == "__main__":
    pipeline = KinectPipeline()
    pipeline.run()