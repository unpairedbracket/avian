//! Applies forces and velocities to bodies in order to move them according to the equations of motion
//! using numerical integration.
//!
//! See [`IntegratorPlugin`].

#[doc(alias = "symplectic_euler")]
pub mod semi_implicit_euler;

use crate::prelude::*;
use bevy::{
    ecs::{intern::Interned, query::QueryData, schedule::ScheduleLabel},
    prelude::*,
};
use derive_more::derive::From;

/// Integrates Newton's 2nd law of motion, applying forces and moving entities according to their velocities.
///
/// This acts as a prediction for the next positions and orientations of the bodies. The [solver](dynamics::solver)
/// corrects these predicted positions to take constraints like contacts and joints into account.
///
/// Currently, only the [semi-implicit (symplectic) Euler](semi_implicit_euler) integration scheme
/// is supported. It is the standard for game physics, being simple, efficient, and sufficiently accurate.
///
/// The plugin adds systems in the [`IntegrationSet::Velocity`] and [`IntegrationSet::Position`] system sets.
pub struct IntegratorPlugin {
    schedule: Interned<dyn ScheduleLabel>,
}

impl IntegratorPlugin {
    /// Creates a [`IntegratorPlugin`] with the schedule that is used for running the [`PhysicsSchedule`].
    ///
    /// The default schedule is [`SubstepSchedule`].
    pub fn new(schedule: impl ScheduleLabel) -> Self {
        Self {
            schedule: schedule.intern(),
        }
    }
}

impl Default for IntegratorPlugin {
    fn default() -> Self {
        Self::new(SubstepSchedule)
    }
}

impl Plugin for IntegratorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Gravity>();

        app.configure_sets(
            self.schedule.intern(),
            (IntegrationSet::Velocity, IntegrationSet::Position).chain(),
        );

        app.add_systems(
            self.schedule.intern(),
            (
                integrate_velocities.in_set(IntegrationSet::Velocity),
                integrate_positions.in_set(IntegrationSet::Position),
            ),
        );

        #[cfg(feature = "3d")]
        app.add_systems(
            self.schedule.intern(),
            dynamics::rigid_body::mass_properties::update_global_angular_inertia::<()>
                .in_set(IntegrationSet::Position)
                .after(integrate_positions),
        );

        app.get_schedule_mut(PhysicsSchedule)
            .expect("add PhysicsSchedule first")
            .add_systems(
                (
                    apply_impulses.before(SolverSet::Substep),
                    clear_forces_and_impulses.after(SolverSet::Substep),
                )
                    .in_set(PhysicsStepSet::Solver),
            );
    }
}

/// System sets for position and velocity integration,
/// applying forces and moving bodies based on velocity.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntegrationSet {
    /// Applies gravity and external forces to bodies, updating their velocities.
    Velocity,
    /// Moves bodies based on their current velocities and the physics time step.
    Position,
}

/// A resource for the global gravitational acceleration.
///
/// The default is an acceleration of 9.81 m/s^2 pointing down, which is approximate to the gravitational
/// acceleration near Earth's surface. Note that if you are using pixels as length units in 2D,
/// this gravity will be tiny. You should modify the gravity to fit your application.
///
/// You can also control how gravity affects a specific [rigid body](RigidBody) using the [`GravityScale`]
/// component. The magnitude of the gravity will be multiplied by this scaling factor.
///
/// # Example
///
/// ```no_run
#[cfg_attr(feature = "2d", doc = "use avian2d::prelude::*;")]
#[cfg_attr(feature = "3d", doc = "use avian3d::prelude::*;")]
/// use bevy::prelude::*;
///
/// # #[cfg(feature = "f32")]
/// fn main() {
///     App::new()
///         .add_plugins((DefaultPlugins, PhysicsPlugins::default()))
#[cfg_attr(
    feature = "2d",
    doc = "         .insert_resource(Gravity(Vec2::NEG_Y * 100.0))"
)]
#[cfg_attr(
    feature = "3d",
    doc = "         .insert_resource(Gravity(Vec3::NEG_Y * 19.6))"
)]
///         .run();
/// }
/// # #[cfg(not(feature = "f32"))]
/// # fn main() {} // Doc test needs main
/// ```
///
/// You can also modify gravity while the app is running.
#[derive(Reflect, Resource, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Debug, Resource)]
pub struct Gravity(pub Vector);

impl Default for Gravity {
    fn default() -> Self {
        Self(Vector::Y * -9.81)
    }
}

impl Gravity {
    /// Zero gravity.
    pub const ZERO: Gravity = Gravity(Vector::ZERO);
}

#[derive(Component)]
pub struct ConserveAngularMomentum;

#[cfg(feature = "3d")]
#[derive(Reflect, Clone, Copy, Component, Debug, Default, Deref, DerefMut, PartialEq, From)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Debug, Component, Default, PartialEq)]
pub struct AngularMomentum(Vec3);

#[cfg(feature = "3d")]
#[derive(Reflect, Clone, Copy, Component, Debug, Default, Deref, DerefMut, PartialEq, From)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Debug, Component, Default, PartialEq)]
pub struct PreConstraintEffectiveAngularVelocity(Vec3);

#[derive(QueryData)]
#[query_data(mutable)]
struct VelocityIntegrationQuery {
    rb: &'static RigidBody,
    pos: &'static Position,
    prev_pos: Option<&'static mut PreSolveAccumulatedTranslation>,
    #[cfg(feature = "3d")]
    rot: &'static Rotation,
    lin_vel: &'static mut LinearVelocity,
    ang_vel: &'static mut AngularVelocity,
    pre_constraint_ang_vel: &'static mut PreConstraintEffectiveAngularVelocity,
    ang_mom: &'static mut AngularMomentum,
    force: &'static ExternalForce,
    torque: &'static ExternalTorque,
    mass: &'static ComputedMass,
    angular_inertia: &'static ComputedAngularInertia,
    #[cfg(feature = "3d")]
    global_angular_inertia: &'static GlobalAngularInertia,
    lin_damping: Option<&'static LinearDamping>,
    ang_damping: Option<&'static AngularDamping>,
    max_linear_speed: Option<&'static MaxLinearSpeed>,
    max_angular_speed: Option<&'static MaxAngularSpeed>,
    gravity_scale: Option<&'static GravityScale>,
    locked_axes: Option<&'static LockedAxes>,
    angular_momentum_conserving: Has<ConserveAngularMomentum>,
}

#[allow(clippy::type_complexity)]
fn integrate_velocities(
    mut bodies: Query<VelocityIntegrationQuery, RigidBodyActiveFilter>,
    gravity: Res<Gravity>,
    time: Res<Time>,
) {
    let delta_secs = time.delta_seconds_adjusted();

    bodies.par_iter_mut().for_each(|mut body| {
        if let Some(mut previous_position) = body.prev_pos {
            previous_position.0 = body.pos.0;
        }

        if body.rb.is_static() {
            if *body.lin_vel != LinearVelocity::ZERO {
                *body.lin_vel = LinearVelocity::ZERO;
            }
            if *body.ang_vel != AngularVelocity::ZERO {
                *body.ang_vel = AngularVelocity::ZERO;
            }
            return;
        }

        if body.rb.is_dynamic() {
            let locked_axes = body
                .locked_axes
                .map_or(LockedAxes::default(), |locked_axes| *locked_axes);

            // Apply damping
            if let Some(lin_damping) = body.lin_damping {
                if body.lin_vel.0 != Vector::ZERO && lin_damping.0 != 0.0 {
                    body.lin_vel.0 *= 1.0 / (1.0 + delta_secs * lin_damping.0);
                }
            }
            if let Some(ang_damping) = body.ang_damping {
                if body.ang_vel.0 != AngularVelocity::ZERO.0 && ang_damping.0 != 0.0 {
                    body.ang_vel.0 *= 1.0 / (1.0 + delta_secs * ang_damping.0);
                }
            }

            let external_force = body.force.force();
            let external_torque = body.torque.torque() + body.force.torque();
            let gravity = gravity.0 * body.gravity_scale.map_or(1.0, |scale| scale.0);

            semi_implicit_euler::integrate_linear_velocity(
                body.lin_vel.as_deref_mut(),
                external_force,
                *body.mass,
                locked_axes,
                gravity,
                delta_secs,
            );
            semi_implicit_euler::integrate_angular_velocity(
                body.angular_momentum_conserving,
                body.ang_vel.as_deref_mut(),
                body.ang_mom.as_deref_mut(),
                external_torque,
                body.angular_inertia,
                #[cfg(feature = "3d")]
                body.global_angular_inertia,
                #[cfg(feature = "3d")]
                *body.rot,
                locked_axes,
                delta_secs,
            );
            body.pre_constraint_ang_vel.0 = body.ang_vel.0;
        }

        // Clamp velocities
        if let Some(max_linear_speed) = body.max_linear_speed {
            let linear_speed_squared = body.lin_vel.0.length_squared();
            if linear_speed_squared > max_linear_speed.0.powi(2) {
                body.lin_vel.0 *= max_linear_speed.0 / linear_speed_squared.sqrt();
            }
        }
        if let Some(max_angular_speed) = body.max_angular_speed {
            #[cfg(feature = "2d")]
            if body.ang_vel.abs() > max_angular_speed.0 {
                body.ang_vel.0 = max_angular_speed.copysign(body.ang_vel.0);
            }
            #[cfg(feature = "3d")]
            {
                let angular_speed_squared = body.ang_vel.0.length_squared();
                if angular_speed_squared > max_angular_speed.0.powi(2) {
                    body.ang_vel.0 *= max_angular_speed.0 / angular_speed_squared.sqrt();
                }
            }
        }
    });
}

#[allow(clippy::type_complexity)]
fn integrate_positions(
    mut bodies: Query<
        (
            &RigidBody,
            &Position,
            Option<&mut PreSolveAccumulatedTranslation>,
            &mut AccumulatedTranslation,
            &mut Rotation,
            &LinearVelocity,
            &mut AngularVelocity,
            &mut AngularMomentum,
            &PreConstraintEffectiveAngularVelocity,
            &ComputedAngularInertia,
            &mut GlobalAngularInertia,
            Option<&LockedAxes>,
            Has<ConserveAngularMomentum>,
        ),
        RigidBodyActiveFilter,
    >,
    time: Res<Time>,
) {
    let delta_secs = time.delta_seconds_adjusted();

    bodies.par_iter_mut().for_each(
        |(
            rb,
            pos,
            pre_solve_accumulated_translation,
            mut accumulated_translation,
            mut rot,
            lin_vel,
            mut ang_vel,
            mut ang_mom,
            pre_constraint_angular_velocity,
            body_inertia,
            mut world_inertia,
            locked_axes,
            angular_momentum_conserving,
        )| {
            if let Some(mut previous_position) = pre_solve_accumulated_translation {
                previous_position.0 = pos.0;
            }

            if rb.is_static() || (lin_vel.0 == Vector::ZERO && *ang_vel == AngularVelocity::ZERO) {
                return;
            }

            let locked_axes = locked_axes.map_or(LockedAxes::default(), |locked_axes| *locked_axes);

            semi_implicit_euler::integrate_position(
                accumulated_translation.as_deref_mut(),
                lin_vel.0,
                locked_axes,
                delta_secs,
            );
            semi_implicit_euler::integrate_rotation(
                angular_momentum_conserving,
                rot.reborrow(),
                ang_mom.as_deref_mut(),
                ang_vel.as_deref_mut(),
                pre_constraint_angular_velocity.0,
                world_inertia.reborrow(),
                body_inertia,
                locked_axes,
                delta_secs,
            );
        },
    );
}

#[cfg(feature = "2d")]
type AngularValue = Scalar;
#[cfg(feature = "3d")]
type AngularValue = Vector;
#[cfg(feature = "2d")]
type TorqueValue = Scalar;
#[cfg(feature = "3d")]
type TorqueValue = Vector;

type ImpulseQueryComponents = (
    &'static RigidBody,
    &'static mut ExternalImpulse,
    &'static mut ExternalAngularImpulse,
    &'static mut LinearVelocity,
    &'static mut AngularVelocity,
    &'static Rotation,
    &'static ComputedMass,
    &'static GlobalAngularInertia,
    Option<&'static LockedAxes>,
);

fn apply_impulses(mut bodies: Query<ImpulseQueryComponents, RigidBodyActiveFilter>) {
    for (
        rb,
        impulse,
        ang_impulse,
        mut lin_vel,
        mut ang_vel,
        _rotation,
        mass,
        global_angular_inertia,
        locked_axes,
    ) in &mut bodies
    {
        if !rb.is_dynamic() {
            continue;
        }

        let locked_axes = locked_axes.map_or(LockedAxes::default(), |locked_axes| *locked_axes);

        let effective_inv_mass = locked_axes.apply_to_vec(Vector::splat(mass.inverse()));
        let effective_angular_inertia =
            locked_axes.apply_to_angular_inertia(*global_angular_inertia);

        // Avoid triggering bevy's change detection unnecessarily.
        let delta_lin_vel = impulse.impulse() * effective_inv_mass;
        let delta_ang_vel = effective_angular_inertia.inverse()
            * (ang_impulse.impulse() + impulse.angular_impulse());

        if delta_lin_vel != Vector::ZERO {
            lin_vel.0 += delta_lin_vel;
        }
        if delta_ang_vel != AngularVelocity::ZERO.0 {
            ang_vel.0 += delta_ang_vel;
        }
    }
}

type ForceComponents = (
    &'static mut ExternalForce,
    &'static mut ExternalTorque,
    &'static mut ExternalImpulse,
    &'static mut ExternalAngularImpulse,
);
type ForceComponentsChanged = Or<(
    Changed<ExternalForce>,
    Changed<ExternalTorque>,
    Changed<ExternalImpulse>,
    Changed<ExternalAngularImpulse>,
)>;

/// Responsible for clearing forces and impulses on bodies.
///
/// Runs in [`PhysicsSchedule`], after [`PhysicsStepSet::SpatialQuery`].
pub fn clear_forces_and_impulses(mut forces: Query<ForceComponents, ForceComponentsChanged>) {
    for (mut force, mut torque, mut impulse, mut angular_ímpulse) in &mut forces {
        if !force.persistent {
            force.clear();
        }
        if !torque.persistent {
            torque.clear();
        }
        if !impulse.persistent {
            impulse.clear();
        }
        if !angular_ímpulse.persistent {
            angular_ímpulse.clear();
        }
    }
}
