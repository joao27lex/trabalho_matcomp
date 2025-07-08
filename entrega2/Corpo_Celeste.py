class Corpo_Celeste:
    def __init__(
        self,
        massa,
        raio,
        color,
        name,
        pos_x,
        pos_y,
        vel_x=0.0,
        vel_y=0.0,
        abg=True,
        awg=True,
    ):
        self.massa = massa
        self.raio = raio
        self.color = color
        self.name = name
        self.pos_x = float(pos_x)
        self.pos_y = float(pos_y)
        self.vel_x = float(vel_x)
        self.vel_y = float(vel_y)
        # abg - affected by gravity, se esse corpo celeste vai ter sua aceleração afetada pela gravidade. Padrão False
        self.abg = abg
        # awg - affect with gravity, se esse corpo celeste vai afetar a aceleração dos outros. Padrão True
        self.awg = awg
        self.trace = []

    # essa funcao é pro equacoes_movimento
    # x, y, vx, vy = estado
    def return_estado(self):
        info = [self.pos_x, self.pos_y, self.vel_x, self.vel_y]
        return info

    def return_pos(self):
        info = [self.pos_x, self.pos_y]
        return info

    def return_vel(self):
        info = [self.vel_x, self.vel_y]
        return info

    def return_name(self):
        name = self.name
        return name
