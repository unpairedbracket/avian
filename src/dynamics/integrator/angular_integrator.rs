//! The *semi-implicit* or *symplectic* Euler [integration](super) scheme.
//!
//! [Semi-implicit Euler](https://en.wikipedia.org/wiki/Semi-implicit_Euler_method)
//! integration is the most common integration scheme because it is simpler and more
//! efficient than implicit Euler integration, has great energy conservation,
//! and provides much better accuracy than explicit Euler integration.
//!
//! Semi-implicit Euler integration evalutes the acceleration at
//! the current timestep and the velocity at the next timestep:
//!
//! ```text
//! v = v_0 + a * Δt (linear velocity)
//! ω = ω_0 + α * Δt (angular velocity)
//! ```
//!
//! and computes the new position:
//!
//! ```text
//! x = x_0 + v * Δt (position)
//! θ = θ_0 + ω * Δt (rotation)
//! ```
//!
//! This order is opposite to explicit Euler integration, which uses the velocity
//! at the current timestep instead of the next timestep. The explicit approach
//! can lead to bodies gaining energy over time, which is why the semi-implicit
//! approach is typically preferred.

use super::*;

use bevy::ecs::world;
use nalgebra::ComplexField;

#[derive(Component)]
pub struct AngularMomentum(pub Vec3);

/// Integrates velocity based on the given forces in order to find
/// the linear and angular velocity after `delta_seconds` have passed.
///
/// This uses [semi-implicit (symplectic) Euler integration](self).
#[cfg(feature = "2d")]
#[allow(clippy::too_many_arguments)]
pub fn integrate_2d(
    rotation: &mut Rotation,
    ang_vel: &mut AngularValue,
    torque: TorqueValue,
    inv_inertia: impl Into<InverseInertia>,
    locked_axes: LockedAxes,
    delta_seconds: Scalar,
) {
    let inv_inertia = inv_inertia.into();

    // Compute angular acceleration.
    let ang_acc = angular_acceleration(torque, inv_inertia.rotated(&rotation).0, locked_axes);

    // Compute angular velocity delta.
    // Δω = α * Δt
    #[allow(unused_mut)]
    let mut delta_ang_vel = ang_acc * delta_seconds;

    if delta_ang_vel != AngularVelocity::ZERO.0 && delta_ang_vel.is_finite() {
        *ang_vel += delta_ang_vel;
    }

    // Effective inverse inertia along each rotational axis
    let ang_vel_locked = locked_axes.apply_to_angular_velocity(*ang_vel);

    // θ = θ_0 + ω * Δt
    {
        let delta_rot = Rotation::radians(ang_vel_locked * delta_seconds);
        if delta_rot != Rotation::IDENTITY && delta_rot.is_finite() {
            *rotation *= delta_rot;
        }
    }
}

#[cfg(feature = "3d")]
#[derive(Default, Clone, Copy, Component, Debug)]
pub enum AngularIntegrator {
    #[default]
    Catto,
    ExplicitCorrected,
    Buss,
    ImplicitMidpoint(usize, &'static str),
    NewtonImplicitMidpoint(usize, &'static str),
    ABC,
    Custom(
        fn(
            rotation: &mut Rotation,
            ang_mom: &mut AngularMomentum,
            ang_vel: &mut AngularValue,
            torque: TorqueValue,
            inv_inertia: InverseInertia,
            locked_axes: LockedAxes,
            delta_seconds: Scalar,
        ),
        &'static str,
    ),
}

impl AngularIntegrator {
    pub fn implicit_midpoint(n_steps: usize) -> AngularIntegrator {
        AngularIntegrator::ImplicitMidpoint(
            n_steps,
            format!("implicit midpoint ({n_steps} steps)").leak(),
        )
    }

    pub fn newton_implicit_midpoint(n_steps: usize) -> AngularIntegrator {
        AngularIntegrator::NewtonImplicitMidpoint(
            n_steps,
            format!("newton midpoint ({n_steps} steps)").leak(),
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            AngularIntegrator::Catto => "catto",
            AngularIntegrator::ExplicitCorrected => "corrected explicit",
            AngularIntegrator::Buss => "buss",
            AngularIntegrator::ABC => "ABC",
            AngularIntegrator::ImplicitMidpoint(_, name)
            | AngularIntegrator::NewtonImplicitMidpoint(_, name)
            | AngularIntegrator::Custom(_, name) => name,
        }
    }

    pub fn integrate(
        &self,
        rotation: &mut Rotation,
        ang_mom: &mut AngularMomentum,
        ang_vel: &mut AngularValue,
        torque: TorqueValue,
        inv_inertia: impl Into<InverseInertia>,
        locked_axes: LockedAxes,
        delta_seconds: Scalar,
    ) {
        let integrator = match self {
            AngularIntegrator::Catto => integrate_catto,
            AngularIntegrator::ExplicitCorrected => integrate_explicit_correction,
            AngularIntegrator::Buss => integrate_buss,
            AngularIntegrator::ABC => integrate_abc,
            AngularIntegrator::ImplicitMidpoint(n_steps, _) => {
                return integrate_implicit_midpoint(
                    rotation,
                    ang_mom,
                    ang_vel,
                    torque,
                    inv_inertia,
                    locked_axes,
                    delta_seconds,
                    *n_steps,
                )
            }
            AngularIntegrator::NewtonImplicitMidpoint(n_steps, _) => {
                return integrate_newton_midpoint(
                    rotation,
                    ang_mom,
                    ang_vel,
                    torque,
                    inv_inertia,
                    locked_axes,
                    delta_seconds,
                    *n_steps,
                )
            }
            AngularIntegrator::Custom(f, _) => *f,
        };
        integrator(
            rotation,
            ang_mom,
            ang_vel,
            torque,
            inv_inertia.into(),
            locked_axes,
            delta_seconds,
        )
    }
}

/// Integrates velocity based on the given forces in order to find
/// the angular velocity and orientation after `delta_seconds` have passed.
#[cfg(feature = "3d")]
#[allow(clippy::too_many_arguments)]
pub fn integrate_catto(
    rotation: &mut Rotation,
    _ang_mom: &mut AngularMomentum,
    ang_vel: &mut AngularValue,
    torque: TorqueValue,
    inv_inertia: impl Into<InverseInertia>,
    locked_axes: LockedAxes,
    delta_seconds: Scalar,
) {
    let inv_inertia = inv_inertia.into();

    // Compute angular acceleration.
    let ang_acc = angular_acceleration(torque, inv_inertia.rotated(&rotation).0, locked_axes);

    // Compute angular velocity delta.
    // Δω = α * Δt
    #[allow(unused_mut)]
    let mut delta_ang_vel = ang_acc * delta_seconds;

    #[cfg(feature = "3d")]
    {
        // In 3D, we should also handle gyroscopic motion, which accounts for
        // non-spherical shapes that may wobble as they spin in the air.
        //
        // Gyroscopic motion happens when the inertia tensor is not uniform, causing
        // the angular momentum to point in a different direction than the angular velocity.
        //
        // The gyroscopic torque is τ = ω x Iω.
        //
        // However, the basic semi-implicit approach can blow up, as semi-implicit Euler
        // extrapolates velocity and the gyroscopic torque is quadratic in the angular velocity.
        // Thus, we use implicit Euler, which is much more accurate and stable, although slightly more expensive.
        let delta_ang_vel_gyro =
            solve_gyroscopic_torque(*ang_vel, rotation.0, inv_inertia.inverse(), delta_seconds);
        delta_ang_vel += locked_axes.apply_to_angular_velocity(delta_ang_vel_gyro);
    }

    if delta_ang_vel != AngularVelocity::ZERO.0 && delta_ang_vel.is_finite() {
        *ang_vel += delta_ang_vel;
    }

    // Effective inverse inertia along each rotational axis
    let ang_vel_locked = locked_axes.apply_to_angular_velocity(*ang_vel);

    // θ = θ_0 + ω * Δt
    // This is a bit more complicated because quaternions are weird.
    // Maybe there's a simpler and more numerically stable way?
    let scaled_axis = ang_vel_locked * delta_seconds;
    if scaled_axis != AngularVelocity::ZERO.0 && scaled_axis.is_finite() {
        let delta_rot = Quaternion::from_scaled_axis(scaled_axis);
        rotation.0 = delta_rot * rotation.0;
        rotation.renormalize();
    }
}

/// Computes angular acceleration based on the current angular velocity, torque, and inertia.
#[cfg_attr(
    feature = "3d",
    doc = "
Note that this does not account for gyroscopic motion. To compute the gyroscopic angular velocity
correction, use `solve_gyroscopic_torque`."
)]
pub fn angular_acceleration(
    torque: TorqueValue,
    world_inv_inertia: InertiaValue,
    locked_axes: LockedAxes,
) -> AngularValue {
    // Effective inverse inertia along each axis
    let effective_inv_inertia = locked_axes.apply_to_rotation(world_inv_inertia);

    if effective_inv_inertia != InverseInertia::ZERO.0 && effective_inv_inertia.is_finite() {
        // Newton's 2nd law for rotational movement:
        //
        // τ = I * α
        // α = τ / I
        //
        // where α (alpha) is the angular acceleration,
        // τ (tau) is the torque, and I is the moment of inertia.
        world_inv_inertia * torque
    } else {
        AngularValue::ZERO
    }
}

/// Computes the angular correction caused by gyroscopic motion,
/// which may cause objects with non-uniform angular inertia to wobble
/// while spinning.
#[cfg(feature = "3d")]
pub fn solve_gyroscopic_torque(
    ang_vel: Vector,
    rotation: Quaternion,
    local_inertia: Inertia,
    delta_seconds: Scalar,
) -> Vector {
    // Based on the "Gyroscopic Motion" section of Erin Catto's GDC 2015 slides on Numerical Methods.
    // https://box2d.org/files/ErinCatto_NumericalMethods_GDC2015.pdf

    // Convert angular velocity to body coordinates so that we can use the local angular inertia
    let local_ang_vel = rotation.inverse() * ang_vel;

    // Compute body-space angular momentum
    let angular_momentum = local_inertia.0 * local_ang_vel;

    // Compute Jacobian
    let jacobian = local_inertia.0
        + delta_seconds
            * (skew_symmetric_mat3(local_ang_vel) * local_inertia.0
                - skew_symmetric_mat3(angular_momentum));

    // Residual vector
    let f = delta_seconds * local_ang_vel.cross(angular_momentum);

    // Do one Newton-Raphson iteration
    let delta_ang_vel = -jacobian.inverse() * f;

    // Convert back to world coordinates
    rotation * delta_ang_vel
}

/// Integrates velocity based on the given forces in order to find
/// the linear and angular velocity after `delta_seconds` have passed.
///
/// This uses [semi-implicit (symplectic) Euler integration](self).
#[allow(clippy::too_many_arguments)]
pub fn integrate_explicit_correction(
    rotation: &mut Rotation,
    ang_mom: &mut AngularMomentum,
    ang_vel: &mut AngularValue,
    torque: TorqueValue,
    inv_inertia: impl Into<InverseInertia>,
    locked_axes: LockedAxes,
    delta_seconds: Scalar,
) {
    let body_inertia = inv_inertia.into();
    let mut inv_inertia = body_inertia.rotated(rotation);

    // Compute angular acceleration.
    let ang_acc = inv_inertia.0 * (torque - ang_vel.cross(ang_mom.0));

    // Approximate average of angular velocity across a timestep
    let mean_ang_vel =
        locked_axes.apply_to_angular_velocity(*ang_vel + ang_acc * delta_seconds / 2.0);

    let predicted_rotation = mean_ang_vel * delta_seconds;

    if predicted_rotation != AngularVelocity::ZERO.0 && predicted_rotation.is_finite() {
        let delta_rot = Quaternion::from_scaled_axis(predicted_rotation);
        rotation.0 = delta_rot * rotation.0;
        rotation.renormalize();
        inv_inertia = body_inertia.rotated(rotation);
    }

    // Apply the torque to the angular momentum
    // No gyro component here, that only affects angular *velocity*
    ang_mom.0 += torque * delta_seconds;

    let ang_vel_new = locked_axes.apply_to_angular_velocity(inv_inertia.0 * ang_mom.0);

    if ang_vel_new != *ang_vel && ang_vel_new.is_finite() {
        *ang_vel = ang_vel_new;
    }
}

/// Integrates velocity based on the given forces in order to find
/// the linear and angular velocity after `delta_seconds` have passed.
///
/// This uses [semi-implicit (symplectic) Euler integration](self).
#[allow(clippy::too_many_arguments)]
pub fn integrate_buss(
    rotation: &mut Rotation,
    ang_mom: &mut AngularMomentum,
    ang_vel: &mut AngularValue,
    torque: TorqueValue,
    inv_inertia: impl Into<InverseInertia>,
    locked_axes: LockedAxes,
    delta_seconds: Scalar,
) {
    let body_inertia = inv_inertia.into();
    let mut inv_inertia = body_inertia.rotated(rotation);

    // Compute angular acceleration.
    let ang_acc = inv_inertia.0 * (torque - ang_vel.cross(ang_mom.0));

    // Approximate average of angular velocity across a timestep
    let mean_ang_vel = locked_axes.apply_to_angular_velocity(
        *ang_vel + delta_seconds / 2.0 * (ang_acc + delta_seconds / 6.0 * ang_acc.cross(*ang_vel)),
    );

    let predicted_rotation = mean_ang_vel * delta_seconds;

    if predicted_rotation != AngularVelocity::ZERO.0 && predicted_rotation.is_finite() {
        let delta_rot = Quaternion::from_scaled_axis(predicted_rotation);
        rotation.0 = delta_rot * rotation.0;
        rotation.renormalize();
        inv_inertia = body_inertia.rotated(rotation);
    }

    // Apply the torque to the angular momentum
    // No gyro component here, that only affects angular *velocity*
    ang_mom.0 += torque * delta_seconds;

    let ang_vel_new = locked_axes.apply_to_angular_velocity(inv_inertia.0 * ang_mom.0);

    if ang_vel_new != *ang_vel && ang_vel_new.is_finite() {
        *ang_vel = ang_vel_new;
    }
}

/// Integrates velocity based on the given forces in order to find
/// the linear and angular velocity after `delta_seconds` have passed.
///
/// This uses [semi-implicit (symplectic) Euler integration](self).
#[allow(clippy::too_many_arguments)]
pub fn integrate_implicit_midpoint(
    rotation: &mut Rotation,
    ang_mom: &mut AngularMomentum,
    ang_vel: &mut AngularValue,
    torque: TorqueValue,
    inv_inertia: impl Into<InverseInertia>,
    locked_axes: LockedAxes,
    delta_seconds: Scalar,
    n_steps: usize,
) {
    let body_inv_inertia = inv_inertia.into();
    let mut world_inv_inertia = body_inv_inertia.rotated(rotation);

    // Leapfrogged torque application: pre-rotate
    ang_mom.0 += torque * delta_seconds / 2.0;

    let mut average_ang_vel = world_inv_inertia.0 * ang_mom.0;
    for _ in 0..n_steps {
        let q = Quaternion::from_scaled_axis(average_ang_vel * delta_seconds);
        let iw = 0.5 * (ang_mom.0 + q.conjugate() * ang_mom.0);
        average_ang_vel = world_inv_inertia.0 * iw;
    }

    // lock the rotation to only allowed axes
    average_ang_vel = locked_axes.apply_to_angular_velocity(average_ang_vel);

    let predicted_rotation = average_ang_vel * delta_seconds;

    if predicted_rotation != AngularVelocity::ZERO.0 && predicted_rotation.is_finite() {
        let delta_rot = Quaternion::from_scaled_axis(predicted_rotation);
        rotation.0 = delta_rot * rotation.0;
        rotation.renormalize();
        world_inv_inertia = body_inv_inertia.rotated(rotation);
    }

    // Leapfrogged torque application: post-rotate
    ang_mom.0 += torque * delta_seconds / 2.0;

    let ang_vel_new = locked_axes.apply_to_angular_velocity(world_inv_inertia.0 * ang_mom.0);

    if ang_vel_new != *ang_vel && ang_vel_new.is_finite() {
        *ang_vel = ang_vel_new;
    }
}

/// Integrates velocity based on the given forces in order to find
/// the linear and angular velocity after `delta_seconds` have passed.
///
/// This uses [semi-implicit (symplectic) Euler integration](self).
#[allow(clippy::too_many_arguments)]
pub fn integrate_newton_midpoint(
    rotation: &mut Rotation,
    ang_mom: &mut AngularMomentum,
    ang_vel: &mut AngularValue,
    torque: TorqueValue,
    inv_inertia: impl Into<InverseInertia>,
    locked_axes: LockedAxes,
    delta_seconds: Scalar,
    n_steps: usize,
) {
    let body_inv_inertia: InverseInertia = inv_inertia.into();
    let mut world_inv_inertia = body_inv_inertia.rotated(rotation);

    // Leapfrogged torque application: pre-rotate
    ang_mom.0 += torque * delta_seconds / 2.0;

    let mut iw = ang_mom.0 + delta_seconds / 2.0 * ang_mom.0.cross(world_inv_inertia.0 * ang_mom.0);
    let mut average_ang_vel = world_inv_inertia.0 * iw;

    for _ in 0..n_steps {
        let abs_w = average_ang_vel.length();
        let n = average_ang_vel.normalize_or_zero();
        let q_half = Mat3::from_axis_angle(n, abs_w * delta_seconds / 2.0);
        let qt = Mat3::from_axis_angle(n, -abs_w * delta_seconds);
        let gamma = (abs_w * delta_seconds / 2.0).sinc();
        let a =
            gamma * Mat3::IDENTITY + (1.0 - gamma) * Mat3::from_cols(n[0] * n, n[1] * n, n[2] * n);

        let f: Vec3 = iw - 0.5 * (ang_mom.0 + qt * ang_mom.0);
        let jacobian = Mat3::IDENTITY
            - delta_seconds / 2.0
                * qt
                * skew_symmetric_mat3(ang_mom.0)
                * a
                * q_half
                * world_inv_inertia.0;
        iw -= jacobian.inverse() * f;
        average_ang_vel = world_inv_inertia.0 * iw;
    }

    // lock the rotation to only allowed axes
    average_ang_vel = locked_axes.apply_to_angular_velocity(average_ang_vel);

    let predicted_rotation = average_ang_vel * delta_seconds;

    if predicted_rotation != AngularVelocity::ZERO.0 && predicted_rotation.is_finite() {
        let delta_rot = Quaternion::from_scaled_axis(predicted_rotation);
        rotation.0 = delta_rot * rotation.0;
        rotation.renormalize();
        world_inv_inertia = body_inv_inertia.rotated(rotation);
    }

    // Leapfrogged torque application: post-rotate
    ang_mom.0 += torque * delta_seconds / 2.0;

    let ang_vel_new = locked_axes.apply_to_angular_velocity(world_inv_inertia.0 * ang_mom.0);

    if ang_vel_new != *ang_vel && ang_vel_new.is_finite() {
        *ang_vel = ang_vel_new;
    }
}

pub fn integrate_abc(
    rotation: &mut Rotation,
    ang_mom: &mut AngularMomentum,
    ang_vel: &mut AngularValue,
    torque: TorqueValue,
    inv_inertia: impl Into<InverseInertia>,
    locked_axes: LockedAxes,
    delta_seconds: Scalar,
) {
    let body_inv_inertia: InverseInertia = inv_inertia.into();

    ang_mom.0 += torque * delta_seconds / 2.0;

    let body_am = rotation.0.inverse() * ang_mom.0;
    let body_omega = body_inv_inertia.0 * body_am;
    rotation.0 = rotation.0 * Quat::from_scaled_axis(Vec3::Y * body_omega.y * delta_seconds / 2.0);

    let body_am = rotation.0.inverse() * ang_mom.0;
    let body_omega = body_inv_inertia.0 * body_am;
    rotation.0 = rotation.0 * Quat::from_scaled_axis(Vec3::X * body_omega.x * delta_seconds / 2.0);

    let body_am = rotation.0.inverse() * ang_mom.0;
    let body_omega = body_inv_inertia.0 * body_am;
    rotation.0 = rotation.0 * Quat::from_scaled_axis(Vec3::Z * body_omega.z * delta_seconds);

    let body_am = rotation.0.inverse() * ang_mom.0;
    let body_omega = body_inv_inertia.0 * body_am;
    rotation.0 = rotation.0 * Quat::from_scaled_axis(Vec3::X * body_omega.x * delta_seconds / 2.0);

    let body_am = rotation.0.inverse() * ang_mom.0;
    let body_omega = body_inv_inertia.0 * body_am;
    rotation.0 = rotation.0 * Quat::from_scaled_axis(Vec3::Y * body_omega.y * delta_seconds / 2.0);

    ang_mom.0 += torque * delta_seconds / 2.0;

    *ang_vel = body_inv_inertia.rotated(rotation).0 * ang_mom.0;

    // R1 = Q(omega_w) R0;
    //    = R0 Q(omega_b);
}

// #[cfg(test)]
// mod tests {
//     use approx::assert_relative_eq;

//     use super::*;

//     #[test]
//     fn semi_implicit_euler() {
//         let mut rotation = Rotation::default();

//         #[cfg(feature = "2d")]
//         let mut angular_velocity = 2.0;
//         #[cfg(feature = "3d")]
//         let mut angular_velocity = Vector::Z * 2.0;

//         #[cfg(feature = "2d")]
//         let inv_inertia = 1.0;
//         #[cfg(feature = "3d")]
//         let inv_inertia = Matrix3::IDENTITY;

//         // Step by 100 steps of 0.1 seconds
//         for _ in 0..100 {
//             integrate_catto(
//                 &mut rotation,
//                 &mut angular_velocity,
//                 default(),
//                 inv_inertia,
//                 default(),
//                 1.0 / 10.0,
//             );
//         }

//         #[cfg(feature = "2d")]
//         assert_relative_eq!(
//             rotation.as_radians(),
//             Rotation::radians(20.0).as_radians(),
//             epsilon = 0.00001
//         );
//         #[cfg(feature = "3d")]
//         assert_relative_eq!(
//             rotation.0,
//             Quaternion::from_rotation_z(20.0),
//             epsilon = 0.01
//         );

//         #[cfg(feature = "2d")]
//         assert_relative_eq!(angular_velocity, 2.0, epsilon = 0.00001);
//         #[cfg(feature = "3d")]
//         assert_relative_eq!(angular_velocity, Vector::Z * 2.0, epsilon = 0.00001);
//     }
// }
