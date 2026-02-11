import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()

    def reset(self):
        self.last_error = np.array([0.0, 0.0, 0.0])
        self.integral = np.array([0.0, 0.0, 0.0])

    def update(self, current_pos, target_pos, dt):
        if dt <= 0:
            return np.array([0.0, 0.0, 0.0])

        error = target_pos - current_pos

        p_term = self.kp * error

        self.integral += error * dt
        i_term = self.ki * self.integral

        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative

        output_force = p_term + i_term + d_term

        self.last_error = error

        return output_force

class HedefYoneticisi:
    def __init__(self, start_target, wait_time=2.0):
        self.current_target = np.array(start_target, dtype=float)
        self.wait_time = wait_time

        self.arrival_time = None
        self.is_waiting = False
        self.target_name = "Ilk Hedef"

    def get_target(self):
        return self.current_target

    def check_status(self, current_pos_simge, pid_controller):

        distance = np.linalg.norm(self.current_target - current_pos_simge)

        if distance < 0.2 and not self.is_waiting:
            print(f"Hedefe ulaşıldı! {self.wait_time} sn bekleniyor...")
            self.arrival_time = time.time()
            self.is_waiting = True

        if self.is_waiting:
            gecen_sure = time.time() - self.arrival_time

            if gecen_sure >= self.wait_time:
                self.generate_random_target(pid_controller)
                self.is_waiting = False

    def generate_random_target(self, pid_controller):
        new_x = np.random.uniform(0, 10)
        new_y = np.random.uniform(0, 10)
        new_z = np.random.uniform(0, 10)

        self.current_target = np.array([new_x, new_y, new_z])
        self.target_name = f"Rastgele ({new_x:.1f}, {new_y:.1f}, {new_z:.1f})"

        print(f"Yeni Hedef Atandı: {self.target_name}")

        pid_controller.reset()

class KareSimge:
    def __init__(self, start_x, start_y, start_z):
        self.pos = np.array([start_x, start_y, start_z], dtype=float)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)

class NewtonEulerSolver:
    def __init__(self, mass=1.0):
        self.mass = mass

    def solve(self, force_vector, simge_state, dt):
        acceleration = force_vector / self.mass

        simge_state.velocity += acceleration * dt

        simge_state.velocity *= 0.95

        simge_state.pos += simge_state.velocity * dt

        return simge_state.pos

DELTA_T = 0.05
pid = PIDController(kp=5.5, ki=0.0, kd=5.0)

kare = KareSimge(0, 0, 0)
fizik = NewtonEulerSolver(mass=1.0)
hedef_yoneticisi = HedefYoneticisi(start_target=[8, 8, 8], wait_time=2.0)

plt.ion()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

trail_x, trail_y, trail_z = [], [], []

print("Simülasyon Başladı. Kapatmak için penceredeki X işaretine basın.")

try:
    while True:

        if not plt.fignum_exists(fig.number):
            print("Pencere kapatıldı. Simülasyon sonlandırılıyor.")
            break

        current_target = hedef_yoneticisi.get_target()
        force = pid.update(kare.pos, current_target, DELTA_T)
        yeni_konum = fizik.solve(force, kare, DELTA_T)
        hedef_yoneticisi.check_status(yeni_konum, pid)

        trail_x.append(yeni_konum[0])
        trail_y.append(yeni_konum[1])
        trail_z.append(yeni_konum[2])

        if len(trail_x) > 150:
            trail_x.pop(0); trail_y.pop(0); trail_z.pop(0)

        ax.cla()

        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 12)
        ax.set_zlim(-2, 12)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        durum_str = "BEKLIYOR" if hedef_yoneticisi.is_waiting else "HEDEFE GIDIYOR"
        ax.set_title(f"PID + Newton-Euler | Durum: {durum_str}\n{hedef_yoneticisi.target_name}")

        ax.scatter([current_target[0]], [current_target[1]], [current_target[2]],
                   color='black', marker='x', s=100, label='Hedef')

        ax.scatter([yeni_konum[0]], [yeni_konum[1]], [yeni_konum[2]],
                   color='red', marker='s', s=120, label='Robot')

        ax.plot(trail_x, trail_y, trail_z, color='gray', linestyle='--', alpha=0.5)

        ax.legend(loc='upper left')

        plt.draw()
        plt.pause(DELTA_T) # Zaman adımı kadar bekle

except KeyboardInterrupt:
    print("Kullanıcı tarafından durduruldu.")

plt.close('all')