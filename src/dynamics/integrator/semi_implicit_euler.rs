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

/// Integrates velocity based on the given forces in order to find
/// the linear and angular velocity after `delta_seconds` have passed.
///
/// This uses [semi-implicit (symplectic) Euler integration](self).
#[allow(clippy::too_many_arguments)]
pub fn integrate_velocity(
    lin_vel: &mut Vector,
    force: Vector,
    inv_mass: Scalar,
    locked_axes: LockedAxes,
    gravity: Vector,
    delta_seconds: Scalar,
) {
    // Compute linear acceleration.
    let lin_acc = linear_acceleration(force, inv_mass, locked_axes, gravity);

    // Compute next linear velocity.
    // v = v_0 + a * Δt
    let next_lin_vel = *lin_vel + lin_acc * delta_seconds;
    if next_lin_vel != *lin_vel {
        *lin_vel = next_lin_vel;
    }
}

/// Integrates position and rotation based on the given velocities in order to
/// find the position and rotation after `delta_seconds` have passed.
///
/// This uses [semi-implicit (symplectic) Euler integration](self).
pub fn integrate_position(
    pos: &mut Vector,
    lin_vel: Vector,
    locked_axes: LockedAxes,
    delta_seconds: Scalar,
) {
    let lin_vel = locked_axes.apply_to_vec(lin_vel);

    // x = x_0 + v * Δt
    let next_pos = *pos + lin_vel * delta_seconds;

    if next_pos != *pos && next_pos.is_finite() {
        *pos = next_pos;
    }
}

/// Computes linear acceleration based on the given forces and mass.
pub fn linear_acceleration(
    force: Vector,
    inv_mass: Scalar,
    locked_axes: LockedAxes,
    gravity: Vector,
) -> Vector {
    // Effective inverse mass along each axis
    let inv_mass = locked_axes.apply_to_vec(Vector::splat(inv_mass));

    if inv_mass != Vector::ZERO && inv_mass.is_finite() {
        // Newton's 2nd law for translational movement:
        //
        // F = m * a
        // a = F / m
        //
        // where a is the acceleration, F is the force, and m is the mass.
        //
        // `gravity` below is the gravitational acceleration,
        // so it doesn't need to be divided by mass.
        force * inv_mass + locked_axes.apply_to_vec(gravity)
    } else {
        Vector::ZERO
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn semi_implicit_euler() {
        let mut position = Vector::ZERO;

        let mut linear_velocity = Vector::ZERO;

        let inv_mass = 1.0;
        let gravity = Vector::NEG_Y * 9.81;

        // Step by 100 steps of 0.1 seconds
        for _ in 0..100 {
            integrate_velocity(
                &mut linear_velocity,
                default(),
                inv_mass,
                default(),
                gravity,
                1.0 / 10.0,
            );
            integrate_position(&mut position, linear_velocity, default(), 1.0 / 10.0);
        }

        // Euler methods have some precision issues, but this seems weirdly inaccurate.
        assert_relative_eq!(position, Vector::NEG_Y * 490.5, epsilon = 10.0);
        assert_relative_eq!(linear_velocity, Vector::NEG_Y * 98.1, epsilon = 0.0001);
    }
}
