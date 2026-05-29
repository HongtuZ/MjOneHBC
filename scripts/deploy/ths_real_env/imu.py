import contextlib
import math
import multiprocessing as mp
import os
import struct
import time

import serial


class IMUReader:
    """
    IMU 数据读取类。
    在独立子进程中通过串口解析 IMU 帧数据，主进程通过实例属性实时访问。
    """

    def __init__(
        self,
        cpu_id: int | None = None,
        serial_port: str = "/dev/ttyUSB0",
        baud_rate: int = 460800,
        frame_head: bytes = b"\x59\x53",
        frame_min_len: int = 53,
        buffer_reset_time: float = 0.1,
    ):
        self.cpu_id = cpu_id

        # 共享内存：进程间通过锁同步
        self._imu_data = mp.Array("d", 13)
        self._running = mp.Value("b", False)
        self._process: mp.Process | None = None

        # 串口参数透传
        self._serial_port = serial_port
        self._baud_rate = baud_rate
        self._frame_head = frame_head
        self._frame_min_len = frame_min_len
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
    def _read_loop(
        imu_data,
        running,
        cpu_id,
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

                                with imu_data.get_lock():
                                    imu_data[3] = math.radians(Wx)
                                    imu_data[4] = math.radians(Wy)
                                    imu_data[5] = math.radians(Wz)
                                    imu_data[6] = math.radians(Rx)
                                    imu_data[7] = math.radians(Ry)
                                    imu_data[8] = math.radians(Rz)
                                    imu_data[9] = QW0
                                    imu_data[10] = QX1
                                    imu_data[11] = QY2
                                    imu_data[12] = QZ3
                            except Exception as e:
                                print(f"[IMU] 解析失败: {e}，已跳过该帧")
                                buffer = buffer[2:]
                                continue
                    else:
                        buffer = buffer[-1:]

                time.sleep(0.0001)
            time.sleep(0.0001)

        ser.close()
        print("[IMU] 读取进程已退出")


if __name__ == "__main__":
    imu = IMUReader(cpu_id=7, serial_port="/dev/ttyUSB0", baud_rate=460800)
    for i in range(10):
        print(imu.quaternion)
        time.sleep(1)
    imu.close()
