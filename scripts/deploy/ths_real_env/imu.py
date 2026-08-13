import contextlib
import math
import multiprocessing as mp
import os
import struct
import time
from typing import ClassVar

import serial


class IMUReader:
    """
    IMU 数据读取类。
    在独立子进程中通过串口解析 IMU 帧数据，主进程通过实例属性实时访问。
    通过 imu_type 选择 IMU 型号：
      - "yuansheng": 元生 IMU（被动上报，帧头 0x59 0x53，53 字节，460800 波特率）
      - "wit":       维特 IMU（主动轮询，指令 0xFF 0xAA 0x03 0x0C 0x00，回复帧头 0x55 0x52，33 字节，230400 波特率）
    """

    # 各型号 IMU 的默认串口参数
    IMU_DEFAULTS: ClassVar[dict[str, dict[str, int | bytes]]] = {
        "yuansheng": {
            "baud_rate": 460800,
            "frame_head": b"\x59\x53",
            "frame_min_len": 53,
        },
        "wit": {
            "baud_rate": 230400,
            "frame_head": b"\x55\x52",
            "frame_min_len": 33,
        },
    }

    def __init__(
        self,
        cpu_id: int | None = None,
        serial_port: str = "/dev/ttyUSB0",
        baud_rate: int | None = None,
        imu_type: str = "yuansheng",
        frame_head: bytes | None = None,
        frame_min_len: int | None = None,
        buffer_reset_time: float = 0.1,
    ):
        if imu_type not in self.IMU_DEFAULTS:
            raise ValueError(f"不支持的 imu_type: {imu_type}，可选: {list(self.IMU_DEFAULTS)}")
        self.cpu_id = cpu_id
        self.imu_type = imu_type

        # 共享内存：进程间通过锁同步
        self._imu_data = mp.Array("d", 13)
        self._running = mp.Value("b", False)
        self._process: mp.Process | None = None

        # 串口参数透传（未显式指定时使用对应型号的默认值）
        defaults = self.IMU_DEFAULTS[imu_type]
        self._serial_port = serial_port
        self._baud_rate = baud_rate if baud_rate is not None else defaults["baud_rate"]
        self._frame_head = frame_head if frame_head is not None else defaults["frame_head"]
        self._frame_min_len = frame_min_len if frame_min_len is not None else defaults["frame_min_len"]
        self._buffer_reset_time = buffer_reset_time

        # 初始化即自动启动
        self.start()
        time.sleep(0.1)  # 给点时间初始化

    # ------------------------------------------------------------------
    # 进程控制
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._running.value = True
        self._process = mp.Process(
            target=self._read_loop,
            args=(
                self._imu_data,
                self._running,
                self.cpu_id,
                self.imu_type,
                self._serial_port,
                self._baud_rate,
                self._frame_head,
                self._frame_min_len,
                self._buffer_reset_time,
            ),
            daemon=True,
        )
        self._process.start()

    def close(self) -> None:
        self._running.value = False
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
            self._process = None

    # ------------------------------------------------------------------
    # 外部访问接口
    # ------------------------------------------------------------------
    @property
    def data(self) -> list[float]:
        """返回完整 IMU 数组（长度 13），索引含义与原始代码保持一致"""
        with self._imu_data.get_lock():
            return list(self._imu_data)

    @property
    def gyro(self) -> tuple[float, float, float]:
        """角速度 (Wx, Wy, Wz)，单位：rad/s"""
        with self._imu_data.get_lock():
            return (self._imu_data[3], self._imu_data[4], self._imu_data[5])

    @property
    def euler(self) -> tuple[float, float, float]:
        """欧拉角 (Roll, Pitch, Yaw)，单位：rad"""
        with self._imu_data.get_lock():
            return (self._imu_data[6], self._imu_data[7], self._imu_data[8])

    @property
    def quaternion(self) -> tuple[float, float, float, float]:
        """四元数 (Qw, Qx, Qy, Qz)"""
        with self._imu_data.get_lock():
            return (
                self._imu_data[9],
                self._imu_data[10],
                self._imu_data[11],
                self._imu_data[12],
            )

    # ------------------------------------------------------------------
    # 子进程循环
    # ------------------------------------------------------------------
    @staticmethod
    def _write_imu_data(imu_data, gyro, euler, quat):
        """将解析结果写入共享内存，gyro/euler 单位：度（函数内转 rad）"""
        with imu_data.get_lock():
            imu_data[3] = math.radians(gyro[0])
            imu_data[4] = math.radians(gyro[1])
            imu_data[5] = math.radians(gyro[2])
            imu_data[6] = math.radians(euler[0])
            imu_data[7] = math.radians(euler[1])
            imu_data[8] = math.radians(euler[2])
            imu_data[9] = quat[0]
            imu_data[10] = quat[1]
            imu_data[11] = quat[2]
            imu_data[12] = quat[3]

    @staticmethod
    def _parse_yuansheng_frame(frame):
        """解析元生 IMU 帧（帧头 0x59 0x53，53 字节），返回 (gyro, euler, quat)，单位：度"""
        Wx = struct.unpack("<i", frame[7:11])[0] * 0.000001
        Wy = struct.unpack("<i", frame[11:15])[0] * 0.000001
        Wz = struct.unpack("<i", frame[15:19])[0] * 0.000001
        Rx = struct.unpack("<i", frame[21:25])[0] * 0.000001
        Ry = struct.unpack("<i", frame[25:29])[0] * 0.000001
        Rz = struct.unpack("<i", frame[29:33])[0] * 0.000001
        QW0 = struct.unpack("<i", frame[35:39])[0] * 0.000001
        QX1 = struct.unpack("<i", frame[39:43])[0] * 0.000001
        QY2 = struct.unpack("<i", frame[43:47])[0] * 0.000001
        QZ3 = struct.unpack("<i", frame[47:51])[0] * 0.000001
        return (Wx, Wy, Wz), (Rx, Ry, Rz), (QW0, QX1, QY2, QZ3)

    @staticmethod
    def _unwrap_signed(raw, full_scale):
        """将无符号原始值按量程映射为带符号值"""
        if raw >= full_scale:
            raw -= 2 * full_scale
        return raw

    @classmethod
    def _parse_wit_frame(cls, frame):
        """解析维特 IMU 帧（帧头 0x55 0x52，33 字节），返回 (gyro, euler, quat)，角速度/角度单位：度"""
        Wx = cls._unwrap_signed((frame[3] << 8 | frame[2]) / 32768 * 2000, 2000)
        Wy = cls._unwrap_signed((frame[5] << 8 | frame[4]) / 32768 * 2000, 2000)
        Wz = cls._unwrap_signed((frame[7] << 8 | frame[6]) / 32768 * 2000, 2000)
        Rx = cls._unwrap_signed((frame[14] << 8 | frame[13]) / 32768 * 180, 180)
        Ry = cls._unwrap_signed((frame[16] << 8 | frame[15]) / 32768 * 180, 180)
        Rz = cls._unwrap_signed((frame[18] << 8 | frame[17]) / 32768 * 180, 180)
        QW0 = cls._unwrap_signed((frame[25] << 8 | frame[24]) / 32768, 1)
        QX1 = cls._unwrap_signed((frame[27] << 8 | frame[26]) / 32768, 1)
        QY2 = cls._unwrap_signed((frame[29] << 8 | frame[28]) / 32768, 1)
        QZ3 = cls._unwrap_signed((frame[31] << 8 | frame[30]) / 32768, 1)
        return (Wx, Wy, Wz), (Rx, Ry, Rz), (QW0, QX1, QY2, QZ3)

    @classmethod
    def _read_loop(
        cls,
        imu_data,
        running,
        cpu_id,
        imu_type,
        serial_port,
        baud_rate,
        frame_head,
        frame_min_len,
        buffer_reset_time,
    ):
        # CPU 亲和性绑定（仅 Linux）
        if cpu_id is not None:
            with contextlib.suppress(AttributeError):
                os.sched_setaffinity(0, {cpu_id})
        try:
            ser = serial.Serial(serial_port, baud_rate, timeout=0.001)
        except Exception as e:
            print(f"[IMU] 串口打开失败 {serial_port}: {e}")
            return

        if imu_type == "wit":
            cls._wit_imu_loop(ser, running, imu_data, frame_head, frame_min_len)
        else:
            cls._yuansheng_imu_loop(ser, running, imu_data, frame_head, frame_min_len, buffer_reset_time)

        ser.close()
        print("[IMU] 读取进程已退出")

    @classmethod
    def _yuansheng_imu_loop(cls, ser, running, imu_data, frame_head, frame_min_len, buffer_reset_time):
        buffer = b""
        while running.value:
            start_time = time.time()
            while running.value:
                if time.time() - start_time > buffer_reset_time:
                    buffer = b""
                    start_time = time.time()
                    # print("[IMU] 重置缓冲区", time.time())
                    break

                count = ser.inWaiting()
                if count > 0:
                    buffer += ser.read(count)

                if len(buffer) >= 2:
                    head_pos = buffer.find(frame_head)
                    if head_pos != -1:
                        buffer = buffer[head_pos:]
                        if len(buffer) >= frame_min_len:
                            frame = buffer[:frame_min_len]
                            buffer = buffer[frame_min_len:]
                            try:
                                gyro, euler, quat = cls._parse_yuansheng_frame(frame)
                                cls._write_imu_data(imu_data, gyro, euler, quat)
                            except Exception as e:
                                print(f"[IMU] 解析失败: {e}，已跳过该帧")
                                buffer = buffer[2:]
                                continue
                    else:
                        buffer = buffer[-1:]

                time.sleep(0.0001)
            time.sleep(0.0001)

    @classmethod
    def _wit_imu_loop(cls, ser, running, imu_data, frame_head, frame_min_len):
        """维特 IMU 读取循环：发送轮询指令后等待 33 字节回复并解析"""
        read_imu_cmd = bytes([0xFF, 0xAA, 0x03, 0x0C, 0x00])
        input_error_s = 0  # 通讯故障累加器
        analysis_error_s = 0  # 解算故障累加器
        while running.value:
            ser.flushInput()  # 发送前清空接收缓冲区
            ser.write(read_imu_cmd)
            wait_count = 0  # 接收周期数累加器
            count = 0
            while running.value:
                count = ser.inWaiting()
                if count >= frame_min_len:
                    wait_count = 0
                    break
                time.sleep(0.0005)
                wait_count += 1
                if wait_count > 100:  # 接收等待超出 100 个 0.5 毫秒
                    input_error_s += 1
                    break

            if not running.value:
                break

            if input_error_s >= 5:
                input_error_s = 5
                print("[IMU] 连续超时接收 error，判定 IMU 故障")

            com_input = ser.read(count)
            if count == frame_min_len and len(com_input) >= frame_min_len and com_input[0:2] == frame_head:
                try:
                    gyro, euler, quat = cls._parse_wit_frame(com_input)
                except Exception as e:
                    print(f"[IMU] 解析失败: {e}，已跳过该帧")
                    continue
                # 故障检测：欧拉角全零视为数据无效
                if euler[0] == 0 and euler[1] == 0 and euler[2] == 0:
                    print("[IMU] 欧拉角数据为空，判定 IMU 故障")
                else:
                    input_error_s = 0
                    analysis_error_s = 0
                    cls._write_imu_data(imu_data, gyro, euler, quat)
            else:
                analysis_error_s += 1
                if analysis_error_s >= 5:
                    analysis_error_s = 5
                    print(f"[IMU] 数据解析故障 error, count = {count}")


if __name__ == "__main__":
    imu = IMUReader(
        cpu_id=7,
        serial_port="/dev/ttyUSB0",
        imu_type="wit",
    )
    for i in range(10):
        print(imu.quaternion)
        time.sleep(1)
    imu.close()
