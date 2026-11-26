//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/woodcock/detail/WoodcockExecutor.hh
//! \sa celeritas/woodcock/WoodcockTracking.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/math/Quantity.hh"
#include "corecel/random/distribution/ExponentialDistribution.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "geocel/detail/LengthUnits.hh"
#include "celeritas/alongstep/detail/StepCounter.hh"
#include "celeritas/geo/GeoMaterialView.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/PhysicsStepUtils.hh"

#include "../WoodcockView.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/Logger.hh"
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Transport photons inside an enclosing volume with Woodcock tracking.
 *
 * Woodcock or \em delta tracking (section 9.2 of
 * \cite{woodcock-techniques-1965} ) adds
 * a fictitious "null" collision, which does not change the particle state, to
 * every material in the enclosing volume so that the total cross section
 * \f$ \Sigma^* \f$ is the same everywhere. Distances to collision can then be
 * sampled without stopping at internal boundaries:
 * - A straight-line flight is sampled from the majorant cross section: \f[
   s = \frac{-\ln \xi}{\Sigma^*(E)}.
   \f]
 *   This is the free path distribution (Eq. 7 of
 *   \cite{galtier-integralformulation-2013} ) for a uniform \f$ \Sigma^*\f$.
 * - If the flight reaches the boundary of the enclosing volume, the track stops
 *   there.
 * - Otherwise the track moves to the end of the flight and is located inside
 *   the enclosing volume. Given the total macroscopic cross section
 *   \f$ \Sigma(E) \f$ of the material at that point, the collision is
 *   real with probability \f[
   p = \frac{\Sigma(E)}{\Sigma^*(E)}
   \f]
 *   (section 1 and Fig. 2 of \cite{galtier-integralformulation-2013} ).
 *   For a null collision, sampling continues with the next flight.
 *
 * The resulting collision sites have the same distribution as they would when
 * stopping at every boundary as long as \f$ \Sigma^* \ge \Sigma \f$ everywhere
 * along the flight. See Eqs. 5 to 11 of
 * \cite{galtier-integralformulation-2013} for a derivation (with
 * \f$ \hat k = \Sigma^* \f$), and \cite{coleman-mathematicalverification-1968}
 * for a proof and its application to piecewise constant cross sections. A
 * scheme for cross sections that vary arbitrarily along the flight is described
 * in \cite{carter-montecarlo-1972} .
 *
 * The distance to collision in mean free paths is stored as the
 * physics track's "interaction MFP". Since it is exponentially distributed, the
 * distance remaining when stopping at a boundary has the same distribution
 * and is kept when changing to another majorant or to the local
 * cross section outside the enclosing volume.
 *
 * A track that isn't inside of a tracking volume (because the same logical
 * volume is also placed elsewhere) or that is about to leave it takes a normal
 * step.
 *
 * \pre The track's along-step action is the Woodcock action
 */
struct WoodcockExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView& track);

  private:
    //! Result of one or more flights
    struct Flights
    {
        real_type distance{0};
        ActionId action{};
    };

    // Move inside the root until something happens
    inline CELER_FUNCTION Flights fly(CoreTrackView& track,
                                      GeoTrackView& geo,
                                      WoodcockView::Root const& root);

    // Calculate cross sections and set the local material
    inline CELER_FUNCTION bool update_material(CoreTrackView& track,
                                               GeoTrackView const& geo);

    // Use a normal step
    inline CELER_FUNCTION void revert(CoreTrackView& track);

    // Smallest distance to exit for Woodcock tracking
    static CELER_CONSTEXPR_FUNCTION real_type min_exit_distance()
    {
        return real_type(CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
                             ? 1e-7
                             : 1e-3)
               * lengthunits::centimeter;
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
CELER_FUNCTION void WoodcockExecutor::operator()(CoreTrackView& track)
{
    CELER_EXPECT(track.sim().status() == TrackStatus::alive);

    auto geo = track.geometry();
    auto const root = track.woodcock().find_root(geo, track.volumes());

    Flights flights;
    if (root)
    {
        flights = this->fly(track, geo, root);
    }

    auto sim = track.sim();
    if (CELER_UNLIKELY(sim.status() == TrackStatus::errored))
    {
        return;
    }
    if (!flights.action)
    {
        // Not in a root, or about to leave
        CELER_ASSERT(flights.distance == 0);
        return this->revert(track);
    }

    CELER_ASSERT(flights.distance > 0);
    sim.step_length(flights.distance);
    sim.post_step_action(flights.action);
    sim.add_time(
        flights.distance / native_value_from(track.particle().speed()));
    StepCounter{}(track);
}

//---------------------------------------------------------------------------//
/*!
 * Move inside the root until colliding or reaching the boundary.
 *
 * The action will be null if the track can't move.
 */
CELER_FUNCTION auto WoodcockExecutor::fly(
    CoreTrackView& track, GeoTrackView& geo, WoodcockView::Root const& root)
    -> Flights
{
    auto const woodcock = track.woodcock();
    auto const particle = track.particle();
    real_type const majorant
        = woodcock.calc_majorant(root.id, particle.energy());
    CELER_ASSERT(majorant > 0);

    auto fail = [&track] {
        track.apply_errored();
        return Flights{};
    };

    Flights result;
    for (size_type i = 0; i < woodcock.max_flights(); ++i)
    {
        real_type const mfp = track.physics().interaction_mfp();
        // Sample distance to next collision, real or null (Galtier Eq. 7)
        real_type const flight = mfp / majorant;
        auto const exit = geo.find_exit_step(root.level, flight);
        if (exit.boundary)
        {
            if (exit.distance < min_exit_distance())
            {
                // Finish with a normal step, now or in the next step
                break;
            }
            geo.move_to_exit();
            if (CELER_UNLIKELY(geo.failed())
                || !this->update_material(track, geo))
            {
                return fail();
            }
            track.physics().interaction_mfp(
                max(mfp - exit.distance * majorant,
                    numeric_limits<real_type>::min()));
            result.distance += exit.distance;
            result.action = track.boundary_action();
            return result;
        }

        geo.move_relocate(root.level, flight);
        if (CELER_UNLIKELY(geo.failed()) || !this->update_material(track, geo))
        {
            return fail();
        }
        result.distance += flight;

        // Decide whether the collision is real (Galtier section 1, Fig. 2).
        // Note that the physics view depends on the updated material.
        auto phys = track.physics();
        auto pstep = track.physics_step();
        real_type const xs
            = calc_macro_xs(track.material(), particle, phys, pstep);
        CELER_ASSERT(xs <= majorant);
        auto rng = track.rng();
        if (generate_canonical(rng) * majorant < xs)
        {
            result.action = phys.scalars().discrete_action();
            return result;
        }

        // Sample the next flight
        ExponentialDistribution<real_type> sample_exponential;
        phys.interaction_mfp(sample_exponential(rng));
    }

    if (result.distance > 0)
    {
        result.action = track.core_scalars().woodcock_null_action;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Set the local material, returning false on failure.
 */
CELER_FUNCTION bool WoodcockExecutor::update_material(CoreTrackView& track,
                                                      GeoTrackView const& geo)
{
    auto matid = track.geo_material().material_id(geo.impl_volume_id());
    if (CELER_UNLIKELY(!matid))
    {
#if !CELER_DEVICE_COMPILE
        CELER_LOG_LOCAL(error)
            << R"(Track moved to a volume without an associated material)";
#endif
        return false;
    }
    auto mat = track.material();
    mat = {matid};
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Revert to a normal step.
 */
CELER_FUNCTION void WoodcockExecutor::revert(CoreTrackView& track)
{
    auto phys = track.physics();
    auto pstep = track.physics_step();
    auto sim = track.sim();
    sim.reset_step_limit(calc_physics_step_limit(
        track.material(), track.particle(), phys, pstep));
    sim.along_step_action(track.core_scalars().along_step_neutral_action);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
