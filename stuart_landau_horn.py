"""HORN-like module using Stuart-Landau oscillators instead of DHO.

This file provides `HORN_SL`, a class with a compatible interface to the
original `HORN` class in `horn/model.py` but where each node is a
Stuart–Landau oscillator.

Constructor signature matches the original: (num_input, num_nodes, num_output,
h, alpha, omega, gamma, lam=1.0). Extra `lam` (Hopf parameter) is optional and
defaults to 1.0. The original `alpha` is used as the input gain (forcing
amplitude) to preserve calling compatibility.

Dynamics (continuous form) per node z = x + i y:
    dz/dt = (lam - |z|^2) * z + i * omega_rad * z + coupling + input

We discretize using a semi-implicit step similar in spirit to the original
HORN implementation: update the "velocity"-like variable (y) first, then
update x using the new y.
"""
import math
import torch


class HORN_SL(torch.nn.Module):
    def __init__(self, num_input, num_nodes, num_output, h, alpha, omega, gamma, lam: float = 1.0):
        """Stuart-Landau HORN-like network.

        Args:
            num_input, num_nodes, num_output: as in original HORN
            h: integration timestep
            alpha: input gain (kept same role as in original HORN)
            omega: natural frequency in Hz (will be converted to angular frequency)
            gamma: small damping factor (used to stabilise update)
            lam: Stuart–Landau Hopf parameter (positive -> self-oscillation)
        """
        super().__init__()

        self.num_input = num_input
        self.num_nodes = num_nodes
        self.num_output = num_output

        # core scalars
        self.h = h
        self.input_gain = alpha
        # convert frequency in Hz to angular frequency (rad / time_unit)
        self.omega = omega
        self.omega_rad = 2.0 * math.pi * float(self.omega)
        self.gamma = gamma
        # Hopf parameter
        self.lam = float(lam)

        # recurrent gain normalization (like original)
        self.gain_rec = 1.0 / math.sqrt(self.num_nodes)

        # linear layers (same names as original for compatibility)
        self.i2h = torch.nn.Linear(num_input, num_nodes)
        self.h2h = torch.nn.Linear(num_nodes, num_nodes)
        self.h2o = torch.nn.Linear(num_nodes, num_output)

    def _dzdt(self, x, y, input_t):
        """Compute time-derivative (dx/dt, dy/dt) for Stuart–Landau nodes simultaneously.

        We interpret z = x + i y and compute dz/dt = (lam - |z|^2) z + i*omega*z
        plus recurrent coupling and input forcing. Derivatives for x and y are
        returned as two real tensors.
        """
        # input forcing (real-valued); keep tanh nonlinearity from original HORN
        forcing = self.input_gain * torch.tanh(self.i2h(input_t))

        # radius^2 per node
        r2 = x * x + y * y

        # Stuart-Landau intrinsic real and imag components
        a = (self.lam - r2)
        intrinsic_x = a * x - self.omega_rad * y
        intrinsic_y = a * y + self.omega_rad * x

        # recurrent coupling applied to real and imag parts (scaled)
        coupling_x = self.gain_rec * self.h2h(x)
        coupling_y = self.gain_rec * self.h2h(y)

        # small linear damping on both components for numerical stability
        damp_x = - self.gamma * x
        damp_y = - self.gamma * y

        dxdt = intrinsic_x + coupling_x + forcing + damp_x
        dydt = intrinsic_y + coupling_y + damp_y

        return dxdt, dydt

    def _rk4_step(self, x, y, input_t, h=None):
        """Single RK4 integration step updating x,y simultaneously."""
        if h is None:
            h = self.h
        k1x, k1y = self._dzdt(x, y, input_t)
        k2x, k2y = self._dzdt(x + 0.5*h*k1x, y + 0.5*h*k1y, input_t)
        k3x, k3y = self._dzdt(x + 0.5*h*k2x, y + 0.5*h*k2y, input_t)
        k4x, k4y = self._dzdt(x + h*k3x, y + h*k3y, input_t)

        x_new = x + (h/6.0) * (k1x + 2.0*k2x + 2.0*k3x + k4x)
        y_new = y + (h/6.0) * (k1y + 2.0*k2y + 2.0*k3y + k4y)
        return x_new, y_new

    def forward(self, batch, random_init=None, record=False):
        """Run the model across time steps. API mirrors original HORN.forward.

        Args:
            batch: tensor (time_steps, batch_size, num_input)
            random_init: optional std for Gaussian init
            record: if True, returns recorded trajectories in the ret dict
        Returns:
            dict with key 'output' and optionally recorded 'rec_x_t', 'rec_y_t'
        """
        batch_size = batch.size(1)
        num_timesteps = batch.size(0)

        ret = {}
        if record:
            rec_x_t = torch.zeros(batch_size, num_timesteps, self.num_nodes)
            rec_y_t = torch.zeros(batch_size, num_timesteps, self.num_nodes)
            ret['rec_x_t'] = rec_x_t
            ret['rec_y_t'] = rec_y_t

        # initial conditions
        if random_init is not None:
            x_0 = torch.randn(batch_size, self.num_nodes) * random_init
            y_0 = torch.randn(batch_size, self.num_nodes) * random_init
        else:
            x_0 = torch.zeros(batch_size, self.num_nodes)
            y_0 = torch.zeros(batch_size, self.num_nodes)

        x_t = x_0
        y_t = y_0

        for t in range(num_timesteps):
            # present input only as a pulse at t==0; thereafter use zero input
            if t == 0:
                inp = batch[t]
            else:
                inp = torch.zeros_like(batch[t])
            # integrate x,y simultaneously using RK4 per timestep
            x_t, y_t = self._rk4_step(x_t, y_t, inp, h=self.h)
            if record:
                ret['rec_x_t'][:, t, :] = x_t
                ret['rec_y_t'][:, t, :] = y_t

        output = self.h2o(x_t)
        ret['output'] = output
        return ret
