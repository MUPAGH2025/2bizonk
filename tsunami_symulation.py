import os
os.environ["NUMBA_THREADING_LAYER"] = "omp"

import sys

import numpy as np
from matplotlib import pyplot
from matplotlib import animation
from open_atmos_jupyter_utils import show_anim
from PyMPDATA import ScalarField, Solver, Stepper, VectorField, Options, boundary_conditions

from matplotlib import pyplot
from ipywidgets import HBox
from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.pylabtools import select_figure_formats
from open_atmos_jupyter_utils.temporary_file import TemporaryFile


def show_plot(filename=None, fig=pyplot, inline_format='svg'):
    """ the missing click-to-save-as-pdf-or-svg button for matplotlib/Jupyter 
    (use instead of *.show()) """
    link = save_and_make_link(fig, filename)
    select_figure_formats(InteractiveShell.instance(), {inline_format})
    pyplot.show()
    #display(link)


def save_and_make_link(fig, filename=None):
    """ saves a figure as pdf and returns a Jupyter display()-able click-to-download widget """
    temporary_files = [
        TemporaryFile(suffix=suffix, filename=(
            filename if filename is None else
            ((filename if not filename.endswith('.pdf') else filename[:-4]) + suffix)
        ))
        for suffix in ('.pdf', '.svg')
    ]
    for temporary_file in temporary_files:
        fig.savefig(temporary_file.absolute_path, bbox_inches='tight')
    return HBox([temporary_file.make_link_widget() for temporary_file in temporary_files])


class ShallowWaterEquationsIntegrator:
    def __init__(self, *, h_initial: np.ndarray, bathymetry: np.ndarray, uh_initial: np.ndarray = None,
                 vh_initial: np.ndarray = None, options: Options = None):
        """ initializes the solvers for a given initial condition of `h` assuming zero momenta at t=0 """
        options = options or Options(nonoscillatory=True, infinite_gauge=True)
        X, Y, grid = 0, 1, h_initial.shape
        if uh_initial is None:
            uh_initial = np.zeros(grid)
        if vh_initial is None:
            vh_initial = np.zeros(grid)
        stepper = Stepper(options=options, grid=grid)
        kwargs = {
            'boundary_conditions': [boundary_conditions.Constant(value=0)] * len(grid),
            'halo': options.n_halo,
        }
        advectees = {
            "h": ScalarField(h_initial, **kwargs),
            "uh": ScalarField(uh_initial, **kwargs),
            "vh": ScalarField(vh_initial, **kwargs),
        }
        self.advector = VectorField((
                np.zeros((grid[X] + 1, grid[Y])),
                np.zeros((grid[X], grid[Y] + 1))
            ), **kwargs
        )
        self.solvers = { k: Solver(stepper, v, self.advector) for k, v in advectees.items() }
        self.bathymetry = bathymetry

    def __getitem__(self, key):
        """ returns `key` advectee field of the current solver state """
        return self.solvers[key].advectee.get()
    
    def _apply_half_rhs(self, *, key, axis, g_times_dt_over_dxy):
        """ applies half of the source term in the given direction """
        self[key][:] -= .5 * g_times_dt_over_dxy * self['h'] * np.gradient(self['h']-self.bathymetry, axis=axis)

    def _update_courant_numbers(self, *, axis, key, mask, dt_over_dxy):
        """ computes the Courant number component from fluid column height and momenta fields """
        velocity = np.where(mask, np.nan, 0)
        momentum = self[key]
        np.divide(momentum, self['h'], where=mask, out=velocity)

        all = slice(None, None) 
        all_but_last = slice(None, -1)
        all_but_first_and_last = slice(1, -1)

        velocity_at_cell_boundaries = velocity[( 
            (all_but_last, all),
            (all, all_but_last),
        )[axis]] + np.diff(velocity, axis=axis) / 2 
        courant_number = self.advector.get_component(axis)[(
            (all_but_first_and_last, all),
            (all, all_but_first_and_last)
        )[axis]]
        courant_number[:] = velocity_at_cell_boundaries * dt_over_dxy[axis]
        assert np.amax(np.abs(courant_number)) <= 1

    def __call__(self, *, nt: int, g: float, dt_over_dxy: tuple, outfreq: int, eps: float=1e-8):
        """ integrates `nt` timesteps and returns a dictionary of solver states recorded every `outfreq` step[s] """
        output = {k: [] for k in self.solvers.keys()}
        for it in range(nt + 1): 
            if it != 0:
                mask = self['h'] > eps
                for axis, key in enumerate(("uh", "vh")):
                    self._update_courant_numbers(axis=axis, key=key, mask=mask, dt_over_dxy=dt_over_dxy)
                self.solvers["h"].advance(n_steps=1)
                for axis, key in enumerate(("uh", "vh")):
                    self._apply_half_rhs(key=key, axis=axis, g_times_dt_over_dxy=g * dt_over_dxy[axis])
                    self.solvers[key].advance(n_steps=1)
                    self._apply_half_rhs(key=key, axis=axis, g_times_dt_over_dxy=g * dt_over_dxy[axis])
            if it % outfreq == 0:
                for key in self.solvers.keys():
                    output[key].append(self[key].copy())
        return output
    

def calc_dt_over_dxdy_and_g(*,grid, L_km, v_tsunami_km_per_h, g_m_per_s2=10.0, C=0.4):
    v_tsunami = v_tsunami_km_per_h / 60  # to km/min
    nx, ny = grid
    dx_km = L_km / (nx - 1)
    dy_km = L_km / (ny - 1)
    dt = dy_km / v_tsunami * C
    dt_over_dxy = (dt / dx_km, dt / dy_km)
    g = g_m_per_s2 * 60 * 60 / 1000.0  # to km/min^2
    print(f"dx = {dx_km:.2f} km, dy = {dy_km:.2f} km, dt = {dt:.5f}, dt_over_dxy = {dt_over_dxy[0]:.5f}, {dt_over_dxy[1]:.5f}")
    return dt_over_dxy, dt, dx_km, dy_km, g


def generate_bathymetry(*,
    grid,
    x_width=1000.0,
    y_width=1000.0,
    depth_deep=6.0,
    depth_shelf=0.3,
    y_shelf_start=50,
    y_shelf_end=150,
    y_slope_end=300,
    draw_profile=True,
):

    nx, ny = grid
    y = np.linspace(0.0, x_width, ny)
    x = np.linspace(0.0, y_width, nx)
    X, Y = np.meshgrid(x, y, indexing="ij")

    def smoothstep(y, y0, y1):
        t = np.clip((y - y0) / (y1 - y0), 0, 1)
        return t * t * (3 - 2 * t)

    depth_1d = np.zeros_like(y)

    mask_beach = (y <= y_shelf_start)
    slope = smoothstep(y[mask_beach], 0, y_shelf_start)
    depth_1d[mask_beach] = depth_shelf * slope

    mask_shelf = (y > y_shelf_start) & (y <= y_shelf_end)
    depth_1d[mask_shelf] = depth_shelf

    mask_slope = (y > y_shelf_end) & (y <= y_slope_end)
    slope = smoothstep(y[mask_slope], y_shelf_end, y_slope_end)
    depth_1d[mask_slope] = depth_shelf + (depth_deep - depth_shelf) * slope

    mask_deep = (y > y_slope_end)
    depth_1d[mask_deep] = depth_deep

    bathymetry = np.tile(depth_1d, (nx, 1))

    if draw_profile:
        pyplot.figure(figsize=(10, 6))
        pyplot.plot(y, depth_1d, label="Bathymetry profile")
        pyplot.fill_between(y, 0, depth_1d, color='lightblue')
        pyplot.gca().invert_yaxis()
        pyplot.xlabel("y [km]")
        pyplot.ylabel("Depth [km]")
        pyplot.title("Bathymetry Profile")
        pyplot.grid()
        show_plot(filename="bathymetry_profile.png")
    return bathymetry, X, Y, depth_1d

def make_h_initial(*,
    bathymetry, X, Y,
    uplift_amp_m = 7.0,
    subsidence_amp_m = 2.5,
    x_frac = 0.5,
    x_width_km = 200.0,
    y_frac = 0.5,
    y_width_km = 50.0,
    draw_profile=True,
):

    Lx = X.max() - X.min()
    Ly = Y.max() - Y.min()

    uplift_km     = uplift_amp_m     / 1000.0
    subsidence_km = subsidence_amp_m / 1000.0
    x0 = x_frac * Lx
    y0 = y_frac * Ly
    sigma_x = x_width_km / 2.0
    sigma_y = y_width_km / 2.0
    y1 = y0 - sigma_y 
    y2 = y0 + sigma_y

    gauss_y_uplift = np.exp(-((Y - y1) ** 2) / (2.0 * sigma_y ** 2))
    gauss_y_subs   = np.exp(-((Y - y2) ** 2) / (2.0 * sigma_y ** 2))
    gauss_x = np.exp(-((X - x0) ** 2) / (2.0 * sigma_x ** 2))

    shapeY = uplift_km * gauss_y_uplift - subsidence_km * gauss_y_subs
    shapeX = gauss_x
    eta0 = shapeX * shapeY

    h_initial = bathymetry + eta0

    uh_initial = np.zeros_like(h_initial)
    vh_initial = np.zeros_like(h_initial)

    if draw_profile:
        fig, (ax_eta, ax_bathy) = pyplot.subplots(1, 2, figsize=(12, 5))

        im = ax_eta.imshow(
            eta0 * 1000.0,
            origin="lower",
            extent=[Y.min(), Y.max(), X.min(), X.max()],
            aspect="auto"
        )
        cbar = fig.colorbar(im, ax=ax_eta, pad=0.02)
        cbar.set_label("wave height [m]", fontsize=15)
        ax_eta.set_xlabel("y [km]", fontsize=15)
        ax_eta.set_ylabel("x [km]", fontsize=15)
        ax_eta.set_title("Initial disturbance of the water surface", fontsize=15)
        ax_eta.grid(False)

        y_line = Y[0, :]
        depth_profile = bathymetry[0, :]

        ax_bathy.plot(y_line, depth_profile, label="Bathymetry profile")
        ax_bathy.fill_between(y_line, 0.0, depth_profile, alpha=0.3)
        ax_bathy.invert_yaxis()
        ax_bathy.set_xlabel("y (km)", fontsize=15)
        ax_bathy.set_ylabel("Depth (km)", fontsize=15)
        ax_bathy.set_title("Bathymetry", fontsize=15)
        ax_bathy.grid(True)
        ax_bathy.legend()

        pyplot.tight_layout()
        show_plot(filename="initial_eta.png");
    return h_initial, uh_initial, vh_initial


def animate_cross_section_y(
    output,
    bathymetry,
    y_coords,
    dt,
    outfreq,
    shelf_end,
    slope_end,
    x_index=None,
    create_animation=True,
):
    h_snaps = np.array(output["h"])

    nx, ny = bathymetry.shape
    if x_index is None:
        x_index = nx // 2

    wave_all = (h_snaps[:, x_index, :] - bathymetry[x_index, :]) * 1e3 

    mask = y_coords < 0.7 * y_coords.max()
    y_coords = y_coords[mask]
    wave_all = wave_all[:, mask]
    bathy_section_km = bathymetry[x_index, mask]

    n_snaps, ny = wave_all.shape

    w_min = np.min(wave_all)
    w_max = np.max(wave_all)
    padding = 0.05 * (w_max - w_min if w_max != 0.0 else 1.0)
    ylim_wave = (w_min - padding, w_max + padding)
    bmax, bmin = np.min(bathy_section_km), np.max(bathy_section_km)

    padding_b = 0.05 * (bmax - bmin if bmax != bmin else 1.0)
    ylim_bathy = (bmin - padding_b, bmax + padding_b)

    h_max = np.zeros(n_snaps)
    time_max = np.zeros(n_snaps)
    bathymetry_max = np.zeros(n_snaps)

    def plot(frame, *, ylim=ylim_wave):
        psi = wave_all[frame, :]
        t_min = frame * outfreq * dt
        h_max[frame] = np.max(psi)
        time_max[frame] = t_min
        bathymetry_max[frame] = bathy_section_km[np.argmax(psi)]

        fig, ax = pyplot.subplots(figsize=(10, 4))

        ax.plot(y_coords, psi, color="#125d92", linewidth=1.5, label="tsunami wave")

        ax.axvline(x=shelf_end, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(x=slope_end, color="black", linestyle="--", linewidth=0.8)

        ax.set(
            xlim=(y_coords[0], y_coords[-1]),
            ylim=ylim,
            xlabel="$y$ [km]",
            ylabel="$wave height$ [m]",
            title=f"Tsunami propagation,  t = {t_min:.2f} min",
        )

        ax.tick_params(labelsize=16)
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.title.set_size(15)

        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_xlim(0, y_coords.max())
        ax2 = ax.twinx()
        ax2.plot(
            y_coords,
            bathy_section_km,
            linestyle="--",
            linewidth=1.2,
            color="#631212",
            label="bathymetry",
        )
        ax2.invert_yaxis()
        ax2.set_ylabel("depth (km)", fontsize=20)
        ax2.set_ylim(ylim_bathy)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        pyplot.tight_layout()
        return fig

    if create_animation:
        show_anim(plot, range(n_snaps))
    else:
        for frame in range(n_snaps):
            psi = wave_all[frame, :]
            t_min = frame * outfreq * dt
            h_max[frame] = np.max(psi)
            time_max[frame] = t_min
            bathymetry_max[frame] = bathy_section_km[np.argmax(psi)]

        fig, ax = pyplot.subplots(figsize=(12, 6))
        ax2 = ax.twinx()

        def update(frame):
            ax.clear()
            ax2.clear()

            psi = wave_all[frame, :]
            t_min = frame * outfreq * dt

            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()

            ax.plot(y_coords, psi, color="#125d92", linewidth=1.5, label="tsunami wave")
            ax.axvline(x=shelf_end, color="black", linestyle="--", linewidth=0.8)
            ax.axvline(x=slope_end, color="black", linestyle="--", linewidth=0.8)

            ax.set(
                xlim=(y_coords[0], y_coords[-1]),
                ylim=ylim_wave,
                xlabel="$y$ [km]",
                ylabel="$wave height$ [m]",
                title=f"Tsunami propagation,  t = {t_min:.2f} min",
            )
            ax.grid(True, linewidth=0.3, alpha=0.4)
            ax.set_xlim(0, y_coords.max())

            ax.tick_params(labelsize=14)
            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)
            ax.title.set_size(14)

            ax2.plot(
                y_coords,
                bathy_section_km,
                linestyle="--",
                linewidth=1.2,
                color="#631212",
                label="bathymetry",
            )
            ax2.invert_yaxis()
            ax2.set_ylabel("depth [km]", fontsize=14)
            ax2.set_ylim(ylim_bathy)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            if lines1 or lines2:
                ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            return ax, ax2

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=n_snaps,
            interval=50,
            blit=False,
        )
        anim.save("tsunami_cross_section.gif", writer="pillow", fps=10)
        pyplot.close(fig)

    fig_max, ax_max = pyplot.subplots(figsize=(10, 4))
    ax_max.plot(time_max, h_max, label="Max wave height", color="#125d92")
    ax_max.set(
        xlabel="time [min]",
        ylabel="max wave height [m]",
        title="maximum wave height over time",
    )

    ax_max.tick_params(labelsize=16)
    ax_max.xaxis.label.set_size(15)
    ax_max.yaxis.label.set_size(15)
    ax_max.title.set_size(15)

    ax_max.grid(True, linewidth=0.3, alpha=0.4)

    ax_max2 = ax_max.twinx()
    ax_max2.plot(
        time_max,
        bathymetry_max,
        linestyle="--",
        linewidth=1.2,
        color="#631212",
        label="bathymetry at max height",
    )
    ax_max2.set_ylabel("depth [km]", fontsize=15)
    ax_max2.invert_yaxis()
    ax_max2.set_yscale("log")

    lines1, labels1 = ax_max.get_legend_handles_labels()
    lines2, labels2 = ax_max2.get_legend_handles_labels()
    if lines1 or lines2:
        ax_max.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    pyplot.tight_layout()
    show_plot(filename="max_wave_height_over_time.png")
    






def plot_space_time_diagram(
    output, bathymetry, y_coords, dt, outfreq, 
    shelf_end, slope_end, cmap="viridis", x_index=None,
    pos_segments=None, 
):

    h_snaps = np.array(output["h"])
    nx, ny = bathymetry.shape
    if x_index is None:
        x_index = nx // 2

    wave = (h_snaps[:, x_index, :] - bathymetry[x_index, :]) * 1e3
    n_snaps = wave.shape[0]
    time = np.arange(n_snaps) * outfreq * dt

    fig, ax = pyplot.subplots(figsize=(10, 6))
    im = ax.imshow(
        wave,
        aspect="auto",
        origin="lower",
        extent=[y_coords[0], y_coords[-1], time[0], time[-1]],
        cmap=cmap
    )

    ax.set_xlabel("y [km]", fontsize=15)
    ax.set_ylabel("time [min]", fontsize=15)
    ax.set_xlim(0, 600)
    ax.set_title("Space-time diagram of tsunami wave height", fontsize=15)
    ax.grid(False)

    ax.axvline(x=shelf_end, color='black', linestyle='--', linewidth=2)
    ax.axvline(x=slope_end, color='black', linestyle='--', linewidth=2)

    cbar = pyplot.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("wave height [m]", fontsize=20)

    if pos_segments is not None:
        for idx, (y_min_seg, y_max_seg, t_min_seg, t_max_seg) in enumerate(pos_segments):

            mask_y = (y_coords >= y_min_seg) & (y_coords <= y_max_seg)
            seg_indices = np.where(mask_y)[0]

            sub_wave_y = wave[:, seg_indices]
            sub_y      = y_coords[seg_indices]

            mask_t = (time >= t_min_seg) & (time <= t_max_seg)
            t_indices = np.where(mask_t)[0]

            sub_wave = sub_wave_y[t_indices, :]
            t_seg    = time[t_indices]

            crest_local_idx = np.argmax(sub_wave, axis=1)
            crest_y = sub_y[crest_local_idx]

            a, b = np.polyfit(t_seg, crest_y, 1)
            v_km_per_min = a
            v_kmh = v_km_per_min * 60.0

            ax.plot(a * t_seg + b, t_seg, '--', linewidth=2, color='red')

            ax.text(
                0.52, 0.97 - idx * 0.1,
                (
                    f"seg {idx+1}: y $\in$ [{y_min_seg:.0f},{y_max_seg:.0f}] km, "
                    f"t $\in$ [{t_min_seg:.0f},{t_max_seg:.0f}] min\n"
                    f"v = {-v_kmh:.1f} km/h"
                ),
                transform=ax.transAxes,
                fontsize=12,
                va='top',
                color='white',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2')   
            )

    pyplot.tight_layout()
    show_plot(filename=f"space_time_diagram.png");
    