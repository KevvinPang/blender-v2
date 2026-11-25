"""Deterministic skirt collision and tail springs for the Ellen control takes.

Uses the model's original deform bones. No mesh, weight or rest-pose edits.
The legacy frame handler stays paused: every simulated frame is baked with the
body, so random access and rendering never depend on an unbaked physics cache.
"""
import math
import re
import json
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree


class SecondaryMotion:
    def __init__(self, session):
        self.session = session
        self.rig = session.rig
        self.layout = json.loads(self.rig.get('ellen_physics_layout', '{}'))
        self.skirts, self.tail = [], []
        columns = {}
        for bone in self.rig.pose.bones:
            match = re.fullmatch(r'Q_(\d+)_(\d+)', bone.name)
            if match:
                row, column = map(int, match.groups())
                columns.setdefault(column, []).append((row, bone))
        for column, rows in sorted(columns.items()):
            chain = [b for _, b in sorted(rows)]
            # The final upward-pointing terminal is a marker, not cloth.
            if len(chain) > 2 and chain[-1].bone.tail_local.z > chain[-1].bone.head_local.z:
                chain = chain[:-1]
            if len(chain) >= 3:
                self.skirts.append(chain)
        if 'skirts' in self.layout:
            self.skirts = [self.bound_chain(names) for names in self.layout['skirts']]
        if not getattr(session.p, 'skirt_enabled', True):
            self.skirts = []
        self.tail = [b for b in self.rig.pose.bones if re.fullmatch(r'W_\d+_1', b.name)]
        self.tail.sort(key=lambda b: int(b.name.split('_')[1]))
        if self.tail and self.tail[-1].bone.tail_local.z > self.tail[-1].bone.head_local.z:
            self.tail = self.tail[:-1]
        if 'tail' in self.layout:
            self.tail = self.bound_chain(self.layout['tail'])
        if not getattr(session.p, 'tail_enabled', True):
            self.tail = []
        self.bones = [b for chain in self.skirts for b in chain]+self.tail
        self.states = {}
        self.names = {b.name for b in self.bones}
        self.contacts = 0
        self.tail_contacts = 0
        self.tail_colliders = []
        self.collider_meshes = {}
        self.last_capsules = []
        self.old_mutes = {b.name: b.wiggle_mute for b in self.bones if getattr(b, 'wiggle_enable', False)}
        # Baked springs must not be simulated a second time by Wiggle playback.
        for b in self.bones:
            if b.name in self.old_mutes:
                b.wiggle_mute = True

    def bound_chain(self, names):
        """Explicit skin bindings keep original names, rest bones and weights intact."""
        chain = [self.rig.pose.bones[n] for n in names]
        if any(b.parent != a for a,b in zip(chain,chain[1:])):
            raise ValueError('人物物理链不连续：'+str(names))
        return chain

    def close(self, committed):
        if not committed:
            for name, muted in self.old_mutes.items():
                self.rig.pose.bones[name].wiggle_mute = muted

    def capsules(self):
        s = self.session
        capsules = []
        for side in ('L', 'R'):
            # Pose matrices include the evaluated IK and MMD D-bone transforms.
            leg, knee, ankle = (s.bones[k+side] for k in ('leg', 'knee', 'ankle'))
            hip = s.current_rig_world @ leg.matrix.translation
            joint = s.current_rig_world @ knee.matrix.translation
            foot = s.current_rig_world @ ankle.matrix.translation
            radii = self.layout.get('capsule_radii', {})
            capsules.append((hip, joint, radii.get('leg'+side, .084)*s.scale))
            capsules.append((joint, foot, radii.get('knee'+side, .055)*s.scale))
        return capsules

    def protect_skirt(self, point, radial, center, envelope):
        """Keep the skirt on the outside, including a thigh crossing its surface.

        A nearest-surface push can send a panel inward when a leg has already
        crossed it. Ray/sphere envelopes keep each panel on its original side.
        Overlapping spheres approximate the moving thigh and shin capsules.
        """
        distance = (point.x-center.x)*radial.x+(point.y-center.y)*radial.y
        required = distance
        for along, radius_squared, height in envelope:
            discriminant = radius_squared-(point.z-height)**2
            if discriminant > 0:
                required = max(required, along+math.sqrt(discriminant))
        if required > distance:
            self.contacts += 1
            point += radial*(required-distance)
        return point

    def update_tail_colliders(self):
        """Reuse the tail's authored collider objects; evaluate their moving parents."""
        objects = {}
        for b in self.tail:
            collection = getattr(b, 'wiggle_collider_collection', None)
            collider = getattr(b, 'wiggle_collider', None)
            kind = getattr(b, 'wiggle_collider_type', 'Collection')
            candidates = (collection.all_objects if kind == 'Collection' and collection else
                          [collider] if kind == 'Object' and collider else [])
            for ob in candidates:
                if ob.type == 'MESH' and ob.name in self.session.scene.objects:
                    objects[ob.name] = ob
        self.tail_colliders = []
        for ob in objects.values():
            # The supplied colliders are rigid meshes parented to body bones.
            # Modified meshes are rebuilt so deformed collision surfaces also work.
            key = ob.data.as_pointer()
            if key not in self.collider_meshes or ob.modifiers or ob.data.shape_keys:
                if ob.modifiers or ob.data.shape_keys:
                    import bpy
                    evaluated = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
                    mesh = evaluated.to_mesh()
                else:
                    evaluated, mesh = None, ob.data
                try:
                    tree = BVHTree.FromPolygons([v.co for v in mesh.vertices],
                                               [tuple(p.vertices) for p in mesh.polygons])
                    self.collider_meshes[key] = tree
                finally:
                    if evaluated: evaluated.to_mesh_clear()
            world = ob.matrix_world.copy()
            if abs(world.determinant()) < 1e-10: continue
            inverse = world.inverted()
            self.tail_colliders.append((ob, self.collider_meshes[key], world, inverse,
                                        inverse.to_3x3().transposed()))

    def project_tail(self, point, previous, length, bone, rest, index):
        """Resolve contacts on the fixed-length sphere around the preceding joint."""
        normal = Vector()
        radius = max(0., getattr(bone, 'wiggle_radius', .012))*self.session.scale
        collection = getattr(bone, 'wiggle_collider_collection', None)
        collider = getattr(bone, 'wiggle_collider', None)
        kind = getattr(bone, 'wiggle_collider_type', 'Collection')
        allowed = ({ob.name for ob in collection.all_objects} if kind == 'Collection' and collection else
                   {collider.name} if kind == 'Object' and collider else set())
        for _ in range(3):
            touched = False
            for ob, tree, world, inverse, normals in self.tail_colliders:
                if ob.name not in allowed: continue
                hit = tree.find_nearest(inverse @ point)
                if hit[0] is None: continue
                surface, n = world @ hit[0], (normals @ hit[1]).normalized()
                clearance = radius
                if index <= 2:
                    # The tail is attached inside the hip collider. Preserve only
                    # that authored root overlap instead of stretching its root out.
                    rest_hit = tree.find_nearest(inverse @ rest)
                    if rest_hit[0] is not None:
                        rn = (normals @ rest_hit[1]).normalized()
                        clearance = min(clearance, (rest-world @ rest_hit[0]).dot(rn))
                if (point-surface).dot(n) >= clearance-1e-6: continue
                required = (surface+n*clearance-previous).dot(n)/max(length, 1e-9)
                component = max(-1., min(1., required))
                direction = (point-previous).normalized()
                tangent = direction-n*direction.dot(n)
                if tangent.length_squared < 1e-12:
                    tangent = n.orthogonal()
                direction = tangent.normalized()*math.sqrt(max(0., 1-component*component))+n*component
                point = previous+direction*length
                normal = n
                self.tail_contacts += 1
                touched = True
            if not touched: break
        return point, normal

    def simulate(self, chain, dt, kind, neutral, spheres, opening):
        s, p = self.session, self.session.p
        transforms = [neutral[b.name] for b in chain]
        targets = [m.translation.copy() for m in transforms]
        targets.append(transforms[-1] @ Vector((0, chain[-1].length, 0)))
        if kind == 'TAIL':
            # A slight rest tilt also applies in still shots, without waiting
            # for gravity to overcome the tail's strong shape springs.
            rotation = s.current_rig_world.to_quaternion()
            tilt = rotation @ Quaternion((1, 0, 0), -getattr(p, 'tail_droop', math.radians(7))) @ rotation.inverted()
            targets = [targets[0]+tilt @ (v-targets[0]) for v in targets]
        anchor = targets[0]
        key = chain[0].name
        if key not in self.states:
            initial = ([opening[b.name].translation.copy() for b in chain]+
                       [opening[chain[-1].name] @ Vector((0, chain[-1].length, 0))])
            if s.mode == 'NEW':
                initial = [v.copy() for v in targets]
            self.states[key] = (initial, [Vector() for _ in targets])
        points, velocities = self.states[key]
        lengths = [b.length*s.scale for b in chain]
        amount = p.skirt_softness if kind == 'SKIRT' else p.tail_inertia
        rear = min(1., max(0., chain[-1].bone.tail_local.y)/.12) if kind == 'SKIRT' else 0.
        stiffness = (125. if kind == 'SKIRT' else 75.)/max(.3, amount)
        stiffness *= 1+rear*3
        damping = 14. if kind == 'SKIRT' else 9.
        substeps = max(1, math.ceil(dt*60)) if dt else 1
        h = dt/substeps if dt else 0.
        center = s.current_rig_world @ Vector((0, 0, 0))
        # A model-space outward normal avoids collapse across the skirt tube.
        outward = chain[-1].bone.tail_local.copy()
        outward.z = 0
        radial = s.current_rig_world.to_quaternion() @ outward.normalized()
        radial.z = 0
        radial.normalize()
        # A sphere's projection onto this radial plane is constant for the
        # whole chain/frame. Only the cloth point's height changes in the solve.
        envelope = []
        for c, radius in spheres:
            x, y = c.x-center.x, c.y-center.y
            along = x*radial.x+y*radial.y
            squared = radius*radius-(x*x+y*y-along*along)
            if squared > 0:
                envelope.append((along, squared, c.z))
        if not dt and s.mode != 'NEW':
            return points, transforms
        for _ in range(substeps):
            before = [v.copy() for v in points]
            contact_normals = [Vector() for _ in points]
            if rear:
                # Most root translation follows immediately at the back of
                # the skirt; retain a small amount for visible cloth inertia.
                follow = (anchor-points[0])*(rear*.85)
                for i in range(1, len(points)):
                    points[i] += follow
            for i in range(1, len(points)):
                if h:
                    acc = (targets[i]-points[i])*stiffness
                    acc.z -= (.35 if kind == 'SKIRT' else .8)*s.scale*amount
                    velocities[i] = (velocities[i]+acc*h)*math.exp(-damping*h)
                    points[i] += velocities[i]*h
            points[0] = anchor.copy()
            for iteration in range(5):
                # Root is pinned; all remaining segments keep their rest length.
                for i, length in enumerate(lengths, 1):
                    delta = points[i]-points[i-1]
                    if delta.length < 1e-8:
                        delta = targets[i]-targets[i-1]
                    correction = delta*(1-length/max(1e-8, delta.length))
                    if i == 1:
                        points[i] -= correction
                    else:
                        points[i-1] += correction*.45
                        points[i] -= correction*.55
                if kind == 'SKIRT':
                    for i in range(2, len(points)):
                        self.protect_skirt(points[i], radial, center, envelope)
                        if rear:
                            # Keep the rear hem below the buttocks. Blend at
                            # the side seams so the skirt has no hard fold.
                            rise = getattr(p, 'rear_skirt_lift', .018)+(1-rear)*.08
                            points[i].z = min(points[i].z, targets[i].z+rise)
                else:
                    # The heavy tail has a limited bend per segment and cannot
                    # collapse into a corkscrew during a fast 180-degree turn.
                    for i in range(1, len(points)):
                        rest = (targets[i]-targets[i-1]).normalized()
                        direction = (points[i]-points[i-1]).normalized()
                        q = rest.rotation_difference(direction)
                        limit = .16+min(i, 10)*.055
                        if q.angle > limit:
                            direction = Quaternion().slerp(q, limit/q.angle) @ rest
                        previous = ((points[i-1]-points[i-2]).normalized() if i > 1 else rest)
                        bend = previous.rotation_difference(direction)
                        bend_limit = .12+.004*i
                        if bend.angle > bend_limit:
                            direction = Quaternion().slerp(bend, bend_limit/bend.angle) @ previous
                        # A forward pass must always restore the segment's
                        # length; otherwise a fast root turn stretches the tail.
                        points[i] = points[i-1]+direction*lengths[i-1]
                        points[i], hit = self.project_tail(points[i], points[i-1], lengths[i-1],
                                                          chain[i-1], targets[i], i)
                        if hit.length_squared: contact_normals[i] = hit
                points[0] = anchor.copy()
            if h:
                for i in range(1, len(points)):
                    velocities[i] = (points[i]-before[i])/h
                    if kind == 'TAIL' and contact_normals[i].length_squared:
                        n = contact_normals[i]
                        speed = velocities[i].dot(n)
                        friction = min(1., max(0., getattr(chain[i-1], 'wiggle_friction', .5)))
                        bounce = max(0., getattr(chain[i-1], 'wiggle_bounce', 0.))
                        velocities[i] = (velocities[i]-n*speed)*(1-friction)+n*max(0., -speed*bounce)
                    if velocities[i].length > 3*s.scale:
                        velocities[i] = velocities[i].normalized()*3*s.scale
        return points, transforms

    def step(self, dt):
        if not self.bones:
            return
        s = self.session
        self.contacts = 0
        self.tail_contacts = 0
        if self.tail: self.update_tail_colliders()
        # Read only the evaluated body, before changing any secondary bones.
        capsules = self.capsules()
        self.last_capsules = capsules
        margin = (s.p.skirt_clearance+.008)*s.scale
        spheres = [(a.lerp(b, i/5), radius*(1-.24*i/5)+margin)
                   for a, b, radius in capsules for i in range(6)]
        cache = {}
        def fk(b):
            if b.name not in cache:
                kwargs = {'parent_matrix': fk(b.parent),
                          'parent_matrix_local': b.parent.bone.matrix_local} if b.parent else {}
                basis = Matrix.Identity(4) if b.name in self.names else b.matrix_basis
                cache[b.name] = b.bone.convert_local_to_pose(basis, b.bone.matrix_local, **kwargs)
            return cache[b.name]
        neutral = {b.name: s.current_rig_world @ fk(b) for b in self.bones}
        # Retain the exact current pose at the seam of APPEND/CURRENT takes.
        opening = ({b.name: s.current_rig_world @ b.matrix for b in self.bones}
                   if not self.states else neutral)
        inverse = s.current_rig_world.inverted()
        scale = s.current_rig_world.to_scale()
        for chain, kind in [(c, 'SKIRT') for c in self.skirts]+([(self.tail, 'TAIL')] if self.tail else []):
            points, transforms = self.simulate(chain, dt, kind, neutral, spheres, opening)
            if not dt and s.mode != 'NEW':
                continue
            parent = fk(chain[0].parent)
            for i, b in enumerate(chain):
                rest = transforms[i].to_quaternion()
                direction = (points[i+1]-points[i]).normalized()
                q = (rest @ Vector((0, 1, 0))).rotation_difference(direction) @ rest
                world = Matrix.LocRotScale(points[i], q, scale)
                posed = inverse @ world
                kwargs = {'parent_matrix': parent, 'parent_matrix_local': b.parent.bone.matrix_local}
                basis = b.bone.convert_local_to_pose(posed, b.bone.matrix_local, invert=True, **kwargs)
                b.rotation_mode = s.saved_pose[b.name][1]
                b.matrix_basis = basis
                parent = posed
