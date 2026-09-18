"""A shallow tray with raised handles, in place of robosuite's deep pot.

Why this object exists
----------------------
The marbles are there so that tilting costs something. robosuite's
``PotWithHandlesObject`` is a 12 cm wide box with 13 cm walls, and a ball never
escapes a wall taller than itself, so its contents stay put until the pot is
past 90 deg (measured: ``scripts/t6_spill_curve.py``). A tilt of 45 deg -- an
obviously botched two-arm lift -- spills nothing at all.

    Escape angle over a lip
    -----------------------
    A ball does not float over the lip, it pivots over the lip's top edge.
    Ball radius r, lip height L above the floor. The ball's centre sits at
    height r; its contact with the lip edge is at horizontal offset

        x_e = √(2rL − L²)

    Gravity's along-slope component (∝ sin θ) tips it over about that edge with
    lever arm (r − L); the normal component (∝ cos θ) holds it back with lever
    arm x_e. It rolls out when

        tan θ_escape = √(2rL − L²) / (r − L)

    The denominator is the design constraint: **L ≥ r never spills**, at any
    angle below 90 deg, for any number of balls. Solving for θ = 45 deg gives
    2u² − 4u + 1 = 0 with u = L/r, so

        L = 0.293 · r

The tray keeps the pot's *handles* at a graspable height while dropping its
walls to that lip. Those two are coupled in the original object --
``handle_z = body_half_size[2] − handle_radius`` puts the handles at the rim --
so a simply-shallower pot has its handles lying on the table where the fingers
cannot reach them. Here the handles ride on vertical posts instead.

The class is named so that its object is still called ``pot``: every site, body
and joint name downstream (``pot_center``, ``pot_handle0``, ``pot_joint0``) is
unchanged, and so is every grasp check that reads ``handle0_geoms``. The shape
changes; nothing else does.
"""

from __future__ import annotations

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.models.objects import PotWithHandlesObject
from robosuite.utils.mjcf_utils import add_to_dict, array_to_string


class TrayWithHandlesObject(PotWithHandlesObject):
    """``PotWithHandlesObject`` geometry, re-laid-out as a tray.

    Args:
        lip_height: wall height above the inner floor, in metres. Set it from
            the marble radius: ``0.293 * r`` gives a 45 degree spill.
        handle_height: height of the handle bars above the tray's centre plane.
            Independent of the lip -- that independence is the whole point.
    """

    def __init__(self, name, lip_height=0.0053, handle_height=0.055, **kwargs):
        self.lip_height = lip_height
        self.handle_height = handle_height
        kwargs.setdefault("thickness", 0.01)
        # Outer half-height follows from the lip: floor top sits at
        # -half + thickness and the rim at +half, so lip = 2*half - thickness.
        half = (lip_height + kwargs["thickness"]) / 2.0
        kwargs.setdefault("body_half_size", (0.07, 0.07, half))
        super().__init__(name=name, **kwargs)

    @property
    def inner_half_width(self) -> float:
        return float(self.body_half_size[0] - self.thickness)

    # --- the object is not symmetric about its centre --------------------
    #
    # CompositeObject derives both offsets from ``total_size``, which assumes
    # the shape fills its bounding box in both directions. This one does not:
    # the handles reach 6.5 cm up and the base stops 1.2 cm down. Left to the
    # default, ``bottom_offset`` reports -6.5 cm, the placement sampler lifts
    # the tray by that much to "rest" it on the table, and it hovers 5 cm in
    # the air with its marbles.

    @property
    def bottom_offset(self):
        return np.array([0.0, 0.0, -float(self.body_half_size[2])])

    @property
    def top_offset(self):
        return np.array([0.0, 0.0, float(self.handle_height + self.handle_radius)])

    def _get_geom_attrs(self):
        bhs = np.asarray(self.body_half_size, dtype=float)
        t = self.thickness
        # Bounding box must cover the handles, which now stand well above the
        # tray -- CompositeObject uses total_size for its own placement maths
        # and a box that clips the handles puts the tray through the table.
        top = self.handle_height + self.handle_radius
        total = np.array([
            bhs[0] + self.handle_width / 2 + self.handle_radius,
            bhs[1] + self.handle_length * 2,
            max(bhs[2], top),
        ])
        obj_args: dict = {}
        base_args = {
            "total_size": total,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
        }
        self._handle0_geoms = []
        self._handle1_geoms = []
        self._important_sites = {}

        # --- the tray itself: base plate plus four low walls ----------------
        self.pot_base = ["base"]
        add_to_dict(
            dic=obj_args, geom_types="box",
            geom_locations=(0, 0, -bhs[2] + t / 2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array([bhs[0], bhs[1], t / 2]),
            geom_names="base",
            geom_rgbas=None if self.use_texture else self.rgba_body,
            geom_materials="pot_mat" if self.use_texture else None,
            geom_frictions=None, density=self.density,
        )
        x_off = np.array([0, -(bhs[0] - t / 2), 0, bhs[0] - t / 2])
        y_off = np.array([-(bhs[1] - t / 2), 0, bhs[1] - t / 2, 0])
        w_vals = np.array([bhs[0], bhs[1], bhs[0], bhs[1]])
        r_vals = np.array([np.pi / 2, 0, -np.pi / 2, np.pi])
        for i, (x, y, w, r) in enumerate(zip(x_off, y_off, w_vals, r_vals)):
            add_to_dict(
                dic=obj_args, geom_types="box",
                geom_locations=(x, y, 0),
                geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, r])), to="wxyz"),
                geom_sizes=np.array([t / 2, w, bhs[2]]),
                geom_names=f"body{i}",
                geom_rgbas=None if self.use_texture else self.rgba_body,
                geom_materials="pot_mat" if self.use_texture else None,
                geom_frictions=None, density=self.density,
            )

        # --- handles: up from the tray edge, out, then a cross bar ----------
        #
        # Three segments per handle, and all three are load-bearing visually:
        # uprights AT THE TRAY EDGE, struts running outboard at handle height,
        # then the bar the gripper closes on. An earlier version put the
        # uprights under the bar itself, 9 cm outboard of the tray, joined to
        # nothing -- one rigid body, so it simulated fine, and looked like a
        # handle hovering in the air beside a tray.
        hr = self.handle_radius
        hw = self.handle_width
        upright_half = (self.handle_height - bhs[2]) / 2.0 + hr
        upright_z = bhs[2] + upright_half - hr
        for i, (g_list, side, rgba) in enumerate(zip(
            [self._handle0_geoms, self._handle1_geoms],
            [1.0, -1.0],
            [self.rgba_handle_0, self.rgba_handle_1],
        )):
            edge_y = side * (bhs[1] - hr)
            bar_y = side * (bhs[1] + self.handle_length)
            for bar_side, suffix in ((-1.0, "-"), (1.0, "+")):
                x = bar_side * hw / 2
                # Upright, standing on the tray's own rim.
                name = f"handle{i}_up{suffix}"
                g_list.append(name)
                add_to_dict(
                    dic=obj_args, geom_types="box",
                    geom_locations=(x, edge_y, upright_z),
                    geom_quats=(1, 0, 0, 0),
                    geom_sizes=np.array([hr, hr, upright_half]),
                    geom_names=name,
                    geom_rgbas=None if self.use_texture else rgba,
                    geom_materials=f"handle{i}_mat" if self.use_texture else None,
                    geom_frictions=(self.handle_friction, 0.005, 0.0001),
                    density=self.density,
                )
                # Strut, reaching outboard from the upright to the bar.
                name = f"handle{i}_arm{suffix}"
                g_list.append(name)
                add_to_dict(
                    dic=obj_args, geom_types="box",
                    geom_locations=(x, side * (bhs[1] + self.handle_length / 2), self.handle_height),
                    geom_quats=(1, 0, 0, 0),
                    geom_sizes=np.array([hr, self.handle_length / 2 + hr, hr]),
                    geom_names=name,
                    geom_rgbas=None if self.use_texture else rgba,
                    geom_materials=f"handle{i}_mat" if self.use_texture else None,
                    geom_frictions=(self.handle_friction, 0.005, 0.0001),
                    density=self.density,
                )
            # The bar the gripper actually closes on.
            name = f"handle{i}_c"
            g_list.append(name)
            add_to_dict(
                dic=obj_args, geom_types="box",
                geom_locations=(0, bar_y, self.handle_height),
                geom_quats=(1, 0, 0, 0),
                geom_sizes=np.array([hw / 2 + hr, hr, hr]),
                geom_names=name,
                geom_rgbas=None if self.use_texture else rgba,
                geom_materials=f"handle{i}_mat" if self.use_texture else None,
                geom_frictions=(self.handle_friction, 0.005, 0.0001),
                density=self.density,
            )

        # Sites, registered the way the parent registers them: the task looks
        # them up through ``important_sites``, not by raw name, so a site that
        # exists in the XML but is missing from that dict fails at reset with a
        # bare KeyError.
        site_attrs = []
        for i, side in enumerate((1.0, -1.0)):
            site = self.get_site_attrib_template()
            name = f"handle{i}"
            site.update({
                "name": name,
                "pos": array_to_string(np.array([
                    0.0,
                    side * (bhs[1] + self.handle_length - 0.005),
                    self.handle_height,
                ])),
                "size": "0.005",
                "rgba": array_to_string(
                    self.rgba_handle_0 if i == 0 else self.rgba_handle_1
                ),
            })
            site_attrs.append(site)
            self._important_sites[name] = self.naming_prefix + name

        centre = self.get_site_attrib_template()
        centre.update({"name": "center", "size": "0.005"})
        site_attrs.append(centre)
        self._important_sites["center"] = self.naming_prefix + "center"

        obj_args.update(base_args)
        obj_args["sites"] = site_attrs
        return obj_args
