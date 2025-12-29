use bevy::log::debug;

use crate::prelude::*;

/// Needed to improve stability when `n.dot(dir)` happens to be very close to zero.
const DOT_EPSILON: Scalar = 0.005;

/// Projects input velocity `v` onto the planes defined by the given `normals`.
/// This ensures that `velocity` does not point into any of the planes, but along them.
///
/// This is often used after [`MoveAndSlide::cast_move`] to ensure a character moved that way
/// does not try to continue moving into colliding geometry.
#[must_use]
pub fn project_velocity_bruteforce(v: Vector, normals: &[Dir]) -> Vector {
    if normals.is_empty() {
        return v;
    }

    // The halfspaces defined by the contact normals form a polyhedral cone.
    // We want to find the closest point to v that lies inside this cone.
    //
    // There are three cases to consider:
    // 1. v is already inside the cone -> return v
    // 2. v is outside the cone
    //    a. Project v onto each plane and check if the projection is inside the cone
    //    b. Project v onto each edge (intersection of two planes) and check if the projection is inside the cone
    // 3. If no valid projection is found, return the apex of the cone (the origin)

    // Case 1: Check if v is inside the cone
    if normals
        .iter()
        .all(|normal| normal.adjust_precision().dot(v) >= -DOT_EPSILON)
    {
        return v;
    }

    // Best candidate so far
    let mut best_projection = Vector::ZERO;
    let mut best_distance_sq = Scalar::INFINITY;

    // Helper to test halfspace validity
    let is_valid = |projection: Vector| {
        normals
            .iter()
            .all(|n| projection.dot(n.adjust_precision()) >= -DOT_EPSILON)
    };

    // Case 2a: Face projections (single-plane active set)
    for n in normals {
        let n = n.adjust_precision();
        let n_dot_v = n.dot(v);
        if n_dot_v < -DOT_EPSILON {
            // Project v onto the plane defined by n:
            // projection = v - (v · n) n
            let projection = v - n_dot_v * n;

            // Check if better than previous best and valid
            let distance_sq = v.distance_squared(projection);
            if distance_sq < best_distance_sq && is_valid(projection) {
                best_distance_sq = distance_sq;
                best_projection = projection;
            }
        }
    }

    // Case 2b: Edge projections (two-plane active set)
    // TODO: Can we optimize this from O(n^3) to O(n^2)?
    #[cfg(feature = "3d")]
    {
        let n = normals.len();
        for i in 0..n {
            let ni = normals[i].adjust_precision();
            for nj in normals
                .iter()
                .take(n)
                .skip(i + 1)
                .map(|n| n.adjust_precision())
            {
                // Compute edge direction e = ni x nj
                let e = ni.cross(nj);
                let e_length_sq = e.length_squared();
                if e_length_sq < DOT_EPSILON {
                    // Nearly parallel edge
                    continue;
                }

                // Project v onto the line spanned by e:
                // projection = ((v · e) / |e|²) e
                let projection = e * (v.dot(e) / e_length_sq);

                // Check if better than previous best and valid
                let distance_sq = v.distance_squared(projection);
                if distance_sq < best_distance_sq && is_valid(projection) {
                    best_distance_sq = distance_sq;
                    best_projection = projection;
                }
            }
        }
    }

    // Case 3: If no candidate is found, the projection is at the apex (the origin)
    if best_distance_sq.is_infinite() {
        Vector::ZERO
    } else {
        best_projection
    }
}

/// Projects input velocity `v` onto the planes defined by the given `normals`.
/// This ensures that `velocity` does not point into any of the planes, but along them.
///
/// This is often used after [`MoveAndSlide::cast_move`] to ensure a character moved that way
/// does not try to continue moving into colliding geometry.
///
/// This function uses a GJK-like algorithm to solve the dual problem to velocity projection,
/// from which obtaining the solution of the primal problem is simple.
/// See <https://benspiers.co.uk/Games/Velocity-Projection> for the full mathematical details.
#[must_use]
pub fn project_velocity(v: Vector, normals: &[Dir]) -> Vector {
    -project_onto_conical_hull(-v, normals)
}

/// This type represents a particular geometric object similar to a simplex,
/// except that the first point of the simplex is the origin and the others
/// are all an arbitrarily large distance away from it (informally, "at infinity").
///
/// The relevant property of the "arbitrarily large" distances is that the closest
/// point to any test point is the origin, and the closest line segment, triangle
/// or higher simplex to any test point always includes the origin.
///
/// The non-origin points are therefore all represented by [`Dir`] unit vectors
/// rather than actual points. When running a GJK-like algorithm for this shape,
/// the relevant simplices are:
///  - Point: the origin
///  - Line segment: a ray extending in some given direction from the origin
///  - Triangle: a wedge spanning the area between two of the above rays
#[cfg_attr(
    feature = "3d",
    doc = " - Tetrahedron: a 'solid wedge' spanning the volume between three rays"
)]
///
/// The last of these is not directly represented as a variant of [`SimplicialCone`],
/// as the projection method always prunes its output down to a simplex of non-full
/// dimensionality.
#[derive(Debug)]
enum SimplicialCone {
    /// A simplicial cone consisting only of a single point, the origin
    Origin,
    /// A simplicial cone consisting of two points, the origin and a point at infinity
    /// which together form a semi-infinite ray, the analog to a line segment
    Ray(#[cfg(feature = "3d")] Dir),
    #[cfg(feature = "3d")]
    /// A simplicial cone consisting of three points: the origin and two points at infinity.
    /// It forms a wedge between two rays, the analog to a triangle
    Wedge(Dir, Dir),
}

/// Project the input point `x0` onto the convex cone defined by the given `normals`.
/// This runs a variant of GJK, specialised for point vs. convex cone computation.
#[must_use]
fn project_onto_conical_hull(x0: Vector, normals: &[Dir]) -> Vector {
    let mut maybe_cone = Some(SimplicialCone::Origin);
    // Search vector is the vector pointing from the
    // closest point on the current `maybe_cone` to x0
    let mut search_vector = x0;
    let mut n_iters = 0;

    while let Some(cone) = maybe_cone {
        // If the search vector is very short, that means the current
        // optimal point is very close to the input, so terminate
        if search_vector.length_squared() < DOT_EPSILON * DOT_EPSILON {
            break;
        }

        let betas = normals
            .iter()
            .map(|n| n.adjust_precision().dot(search_vector));

        let Some((best_idx, best_beta)) = betas
            .enumerate()
            .max_by(|(_, beta1), (_, beta2)| Scalar::total_cmp(beta1, beta2))
        else {
            // This only happens if normals is an empty slice
            break;
        };

        // None of the possible directions can improve on
        // the current optimal point, so terminate
        if best_beta <= DOT_EPSILON {
            break;
        }

        let best_normal = normals[best_idx];

        (maybe_cone, search_vector) = cone.project_point(x0, best_normal);
        n_iters += 1;
        if n_iters >= 10 {
            break;
        }
    }
    if n_iters > 4 {
        debug!(
            "Slow to converge with {} normals, took {n_iters} iterations",
            normals.len()
        );
        debug!("Input -v = {x0}");
        debug!("Normals = {normals:?}");
    }
    search_vector
}

impl SimplicialCone {
    /// Update the simplicial cone by adding a new direction vector to it.
    /// Returns the updated simplicial cone, and indicates `x` - the closest point on the
    /// new simplex to `x0` - by returning a new "search vector" `x0 - x`.
    /// This is returned rather than `x` itself because it's more useful for calling code,
    /// and in some cases more direct to compute
    ///
    /// This methods adds the candidate `new_direction` to the current n-point simplicial cone.
    /// It then finds the closest point on the resulting n+1-point simplicial cone to the
    /// target point `x0`.
    ///
    /// If possible (i.e. if the closest point falls on one of the n-simplices on the boundary
    /// of the n+1 simplex), the n+1-simplex is then pruned back down to a n-simplex.
    /// Otherwise, the n+1-simplex is returned.
    ///
    /// If `x0` is found to be contained within a maximum-dimensionality simplicial cone,
    /// [`None`] is returned in place of the updated simplex and the accompanying search vector
    /// is zero. This avoids having to maintain and handle a variant of [`SimplicialCone`] to
    /// represent a full-dimensional simplicial cone.
    ///
    /// This method makes various assumptions about its inputs that match the implementation of
    /// [`project_onto_conical_hull`] and may not return correct results if provided with `x0`
    /// and `new_direction` that do not follow these assumptions.
    /// For example, the dot product of `new_direction` and the search direction from `self`
    /// to `x0` should not be negative.
    fn project_point(self, x0: Vector, new_direction: Dir) -> (Option<SimplicialCone>, Vector) {
        let new_direction_vec = new_direction.adjust_precision();
        match self {
            SimplicialCone::Origin => {
                // New simplex is a ray between the origin and new_direction,
                // closest point is on the resulting ray.

                // The preconditions of this method imply that alpha >= 0.0
                let alpha = new_direction_vec.dot(x0);
                #[cfg(feature = "2d")]
                let ray = Self::Ray();
                #[cfg(feature = "3d")]
                let ray = Self::Ray(new_direction);
                (Some(ray), x0 - alpha * new_direction_vec)
            }
            #[cfg(feature = "2d")]
            // Preconditions imply that:
            // - dot product between new_direction and old search_direction is > 0.0
            //   - i.e. x0 and new_direction in the same half-disk relative to current ray
            // - dot product between new_direction and x0 is smaller than between the current ray and x0
            //   - i.e. new_direction can't fall _between_ current ray and x0
            //   - the two rays must be on either side of x0
            // therefore can deduce that x0 falls within the wedge, so we can just return (None, ZERO)
            // without any more checking
            SimplicialCone::Ray() => (None, Vector::ZERO),
            #[cfg(feature = "3d")]
            SimplicialCone::Ray(previous_direction) => {
                let cross = new_direction_vec.cross(previous_direction.adjust_precision());
                let alpha = x0.dot(cross);
                let new_search_vector = alpha * cross / cross.length_squared();
                // Orient the Wedge(n1,n2) to guarantee that x0 . (n1 x n2) is nonnegative
                let new_cone = if alpha > 0.0 {
                    Self::Wedge(new_direction, previous_direction)
                } else {
                    Self::Wedge(previous_direction, new_direction)
                };

                (Some(new_cone), new_search_vector)
            }
            #[cfg(feature = "3d")]
            SimplicialCone::Wedge(n1, n2) => {
                // According to precondition, `new_direction`` has positive dot-product with old search direction
                // - therefore `new_direction` falls on the same side of the plane spanned by `n1`, `n2` as `x0` does
                // - Wedge(n1,n2) is oriented such that x0 . (n1 x n2) is nonnegative, so on the new simplicial cone
                //   (n1 x n2) points towards new_direction as well, towards the interior of (n1, n2, new_direction)
                // - The cross products below are in the opposite order to the cycle (n1, n2, new_direction),
                //   so they point away from the interior of the solid wedge

                // cross1 points away from n2
                let cross1 = n1.adjust_precision().cross(new_direction_vec);
                let c1sq = cross1.length_squared();
                // distance of x0 from the wedge facet (n1, new_direction)
                // positive if away from n2, negative if towards n2
                let d1 = x0.dot(cross1);

                let inside1 = d1 <= 0.0;

                // cross2 points away from n1
                let cross2 = new_direction_vec.cross(n2.adjust_precision());
                let c2sq = cross2.length_squared();
                // distance of x0 away from the wedge facet (new_direction, n2)
                // positive if away from n1, negative if towards n1
                let d2 = x0.dot(cross2);
                let inside2 = d2 <= 0.0;

                if inside1 & inside2 {
                    // x0 is on the "inside" side of (n1, new_direction) and of (new_direction, n2)
                    // The previous iteration of this method will have established that it's on
                    // the current "inside" side of (n2, n1) as well, so x0 is inside the overall
                    // solid wedge, and our work here is done.
                    (None, Vector::ZERO)
                } else if d1 * d1.abs() * c2sq > d2 * d2.abs() * c1sq {
                    // the above is `if d1 / cross1.length() > d2 / cross2.length()`
                    // but avoiding that couple of square roots

                    // Again careful here to orient the Wedges such that x0 . (n1 x n2) is non-negative.
                    // i.e. (n1 x n2) faces outwards from the current solid wedge
                    // (which will be inwards next iteration, because the next new_direction
                    // will be on that side!)
                    (Some(Self::Wedge(n1, new_direction)), d1 * cross1 / c1sq)
                } else {
                    (Some(Self::Wedge(new_direction, n2)), d2 * cross2 / c2sq)
                }
            }
        }
    }
}

#[cfg(test)]
pub mod test {
    //! Tests for velocity projection, notably the [`QuasiRandomDirection`] type
    //! used in testing and benchmarking functions on uniformly distributed
    //! input directions
    //!
    //! This is used because the velocity projection edge cases (both in terms
    //! of correctness and performance) may show up for relatively small subsets
    //! of input directions

    use super::DOT_EPSILON;
    use crate::prelude::*;

    #[test]
    fn check_agreement() {
        #[cfg(feature = "2d")]
        let normals = &[
            Dir::Y,
            Dir::from_xy(2.0, 1.0).unwrap(),
            Dir::from_xy(-2.0, 1.0).unwrap(),
            Dir::from_xy(1.0, 1.0).unwrap(),
            Dir::from_xy(-1.0, 1.0).unwrap(),
        ];

        #[cfg(feature = "3d")]
        let normals = &[
            Dir::Z,
            Dir::from_xyz(2.0, 0.0, 1.0).unwrap(),
            Dir::from_xyz(-2.0, 0.0, 1.0).unwrap(),
            Dir::from_xyz(0.0, 2.0, 1.0).unwrap(),
            Dir::from_xyz(0.0, -2.0, 1.0).unwrap(),
            Dir::from_xyz(1.5, 1.5, 1.0).unwrap(),
            Dir::from_xyz(1.5, -1.5, 1.0).unwrap(),
            Dir::from_xyz(-1.5, 1.5, 1.0).unwrap(),
            Dir::from_xyz(-1.5, -1.5, 1.0).unwrap(),
            Dir::from_xyz(1.0, 1.75, 1.0).unwrap(),
            Dir::from_xyz(1.0, -1.75, 1.0).unwrap(),
            Dir::from_xyz(-1.0, 1.75, 1.0).unwrap(),
            Dir::from_xyz(-1.0, -1.75, 1.0).unwrap(),
            Dir::from_xyz(1.75, 1.0, 1.0).unwrap(),
            Dir::from_xyz(1.75, -1.0, 1.0).unwrap(),
            Dir::from_xyz(-1.75, 1.0, 1.0).unwrap(),
            Dir::from_xyz(-1.75, -1.0, 1.0).unwrap(),
        ];

        for n in 1..=normals.len() {
            let selected_normals = &normals[..n];
            let velocities = QuasiRandomDirection::default();
            let mut worst_result = (Scalar::NEG_INFINITY, Vector::ZERO);
            for vel in velocities.take(1000) {
                let new_result = super::project_velocity(vel, selected_normals);
                for (k, normal) in selected_normals.iter().enumerate() {
                    let intrusion = -new_result.dot(normal.adjust_precision());
                    assert!(
                        intrusion <= DOT_EPSILON,
                        "velocity still points into constraint plane after projection: \
                        input {vel} against {n} normals has intrusion \
                        {intrusion} against normal {k}"
                    );
                }
                let old_result = super::project_velocity_bruteforce(vel, selected_normals);

                // Measure of quality of output: we're trying to minimise |output - input|
                // How much worse than the brute-force method do we do?
                let amount_worse = (new_result - vel).length() - (old_result - vel).length();
                assert!(
                    amount_worse <= DOT_EPSILON,
                    "velocity projection result was {amount_worse} > {DOT_EPSILON} worse than \
                    result from brute-force method when projecting input {vel} against {n} normals"
                );

                if amount_worse >= worst_result.0 {
                    worst_result = (amount_worse, vel);
                }
            }
            let (worst_error, bad_input) = worst_result;

            eprintln!(
                "For {n} normals, worst case new method is {worst_error} worse than brute-force for input {bad_input}",
            );
        }
    }

    /// Iterator that produces a fixed sequence of unit vectors,
    /// uniformly distributed around the unit sphere.
    ///
    /// The produced sequence of directions is open, meaning it
    /// continues indefinitely without repeating, at least up to
    /// the limits of floating-point precision.
    #[derive(Default)]
    pub struct QuasiRandomDirection {
        #[cfg(feature = "3d")]
        i: Scalar,
        j: Scalar,
    }

    #[cfg(feature = "3d")]
    impl QuasiRandomDirection {
        #[allow(clippy::excessive_precision)]
        const PLASTIC: Scalar = 1.32471795724475;
        const INV_PLASTIC: Scalar = 1.0 / Self::PLASTIC;
        const INV_PLASTIC_SQ: Scalar = Self::INV_PLASTIC * Self::INV_PLASTIC;
    }

    #[cfg(feature = "2d")]
    impl QuasiRandomDirection {
        #[allow(clippy::excessive_precision)]
        const GOLDEN: Scalar = 1.61803398875;
        const INV_GOLDEN: Scalar = 1.0 / Self::GOLDEN;
    }

    impl QuasiRandomDirection {
        /// Reset the internal state of the direction generator
        /// Directions returned by [`Self::next()`] after this
        /// method is called will match the sequence produced
        /// by a new instance
        pub fn reset(&mut self) {
            *self = Self::default();
        }
    }

    impl Iterator for QuasiRandomDirection {
        type Item = Vector;

        fn next(&mut self) -> Option<Self::Item> {
            let phi = 2.0 * PI * self.j;
            let x = phi.cos();
            let y = phi.sin();
            #[cfg(feature = "3d")]
            {
                let z = 2.0 * self.i - 1.0;
                let rho = (1.0 - z * z).sqrt();
                self.i = (self.i + Self::INV_PLASTIC) % 1.0;
                self.j = (self.j + Self::INV_PLASTIC_SQ) % 1.0;
                Some(Vector::new(rho * x, rho * y, z))
            }
            #[cfg(feature = "2d")]
            {
                self.j = (self.j + Self::INV_GOLDEN) % 1.0;
                Some(Vector::new(x, y))
            }
        }
    }
}
