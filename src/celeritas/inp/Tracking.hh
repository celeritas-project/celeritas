//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Tracking.hh
//---------------------------------------------------------------------------//
#pragma once

#include <limits>
#include <string>
#include <vector>

#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Hard cutoffs for counters.
 */
struct TrackingLimits
{
    //! Don't limit the number of steps
    static inline constexpr size_type unlimited
        = std::numeric_limits<size_type>::max();

    //! Steps per track before killing it
    size_type steps{unlimited};
    //! Step iterations before aborting a run
    size_type step_iters{unlimited};
};

//---------------------------------------------------------------------------//
/*!
 * Tracking limits for the main stepping loop.
 */
struct CoreTrackingLimits : TrackingLimits
{
    //! Integration substeps during field propagation before ending the step
    size_type field_substeps{10};

    //! Stop electron/positron below this energy
    // TODO: Energy electron_energy = Energy{0.001};
};

//---------------------------------------------------------------------------//
/*!
 * Tracking limits for the optical stepping loop.
 */
struct OpticalTrackingLimits : TrackingLimits
{
    //! Surface physics iterations per step for tracks crossing a surface
    size_type surface_substeps{1};
};

//---------------------------------------------------------------------------//
/*!
 * Use Woodcock (delta) tracking for photons in the given regions.
 *
 * Instead of stopping at every internal volume boundary, photons sample their
 * flights with a cross section that bounds the ones in all the region's
 * materials, and collisions are then accepted or rejected based on the local
 * cross section (section 9.2 of \cite{woodcock-techniques-1965} ). This can
 * reduce the number of photon steps in granular regions such as sampling
 * calorimeters. It does not change the distribution of physical interactions.
 *
 * - Region names correspond to the labels in \c inp::Regions (i.e., the Geant4
 *   region names). Everything inside a selected region's volumes is included
 *   even if it belongs to a different region.
 * - Photons below \c min_energy stop at every boundary. At low energy,
 *   photoelectric absorption increases the ratio of cross sections between
 *   heavy and light materials, so most of the sampled collisions would be
 *   rejected. This inefficiency due to materials with a high cross section is
 *   noted in the discussion (page 578) of \cite{woodcock-techniques-1965} . See
 *   also \cite{leppanen-performancewoodcock-2010} for advantages and
 *   limitations of the method.
 * - A step pauses after \c max_flights rejected collisions. For calorimeter
 *   showers, performance is best with one or two on GPU and is insensitive to
 *   larger values on CPU.
 *
 * \note Because a Woodcock step can cross any number of volumes, its pre-step
 *   point for scoring is at the \em end of the flight, coincident with the
 *   pre-interaction post-step point. Steps that are not limited by an
 *   interaction do not correspond to the volumes that were traversed.
 */
struct WoodcockTracking
{
    //! Region names: no Woodcock tracking if empty
    std::vector<std::string> regions;
    //! Lowest photon energy for Woodcock tracking
    units::MevEnergy min_energy{0.2};
    //! Maximum number of flights before ending the step
    size_type max_flights{2};

    //! True if any region is selected
    explicit operator bool() const { return !regions.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Specify non-physical parameters which can affect the physics.
 */
struct Tracking
{
    //! Hard-coded cutoffs before giving up
    CoreTrackingLimits limits;
    //! Limits for the optical stepping loop
    OpticalTrackingLimits optical_limits;

    //! Hardcoded maximum step for debugging charged particles (none if zero)
    real_type force_step_limit{};
    //! Optional photon tracking optimization in granular regions
    WoodcockTracking woodcock;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
