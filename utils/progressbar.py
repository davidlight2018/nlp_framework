import time


class ProgressBar:
    def __init__(self, n_total, width=30, desc="Training"):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info=None):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = "%d:%02d:%02d" % (eta // 3600, (eta % 3600) // 60, eta % 60)
            elif eta > 60:
                eta_format = "%d:%02d" % (eta // 60, eta % 60)
            else:
                eta_format = "%ds" % eta
            time_info = f" - ETA: {eta_format}"
        else:
            if time_per_unit >= 1:
                time_info = f" {time_per_unit:.1f}s/step"
            elif time_per_unit >= 1e-3:
                time_info = f" {time_per_unit * 1e3:.1f}ms/step"
            else:
                time_info = f" {time_per_unit * 1e6:.1f}us/step"

            time_used = now - self.start_time
            if time_used > 3600:
                time_used_format = "%d:%02d:%02d" % (time_used // 3600, (time_used % 3600) // 60, time_used % 60)
            elif time_used > 60:
                time_used_format = "%d:%02d" % (time_used // 60, time_used % 60)
            else:
                time_used_format = "%ds" % time_used

            time_info += f" - time: {time_used_format}"

        show_bar += time_info
        if info is not None and isinstance(info, dict):
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')
