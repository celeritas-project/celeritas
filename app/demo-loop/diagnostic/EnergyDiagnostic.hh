//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EnergyDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "celeritas_config.h"
#include "base/Atomics.hh"
#include "base/CollectionAlgorithms.hh"
#include "base/CollectionBuilder.hh"
#include "base/CollectionMirror.hh"
#include "base/Macros.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "orange/Types.hh"
#include "physics/grid/NonuniformGrid.hh"
#include "sim/CoreTrackData.hh"
#include "sim/SimTrackView.hh"

#include "Diagnostic.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Diagnostic class for binning energy deposition along an axis.
 */
template<MemSpace M>
class EnergyDiagnostic : public Diagnostic<M>
{
  public:
    //!@{
    //! Types
    using real_type    = celeritas::real_type;
    using Axis         = celeritas::Axis;
    using Items        = celeritas::Collection<real_type, Ownership::value, M>;
    using StateRef     = celeritas::CoreStateData<Ownership::reference, M>;
    using TransporterResult = celeritas::TransporterResult;
    //!@}

  public:
    // Construct with grid parameters
    explicit EnergyDiagnostic(const std::vector<real_type>& bounds, Axis axis);

    // Number of alive tracks determined at the end of a step.
    void mid_step(const StateRef& states) final;

    // Collect diagnostic results
    void get_result(TransporterResult* result) final;

    // Get vector of binned energy deposition
    std::vector<real_type> energy_deposition();

  private:
    Items bounds_;
    Items energy_per_bin_;
    Axis  axis_;
};

//---------------------------------------------------------------------------//
/*!
 * Holds pointers to grid and binned energy values to pass to kernel
 */
template<MemSpace M>
struct EnergyBinPointers
{
    using Axis = celeritas::Axis;
    template<Ownership W>
    using Items = celeritas::Collection<celeritas::real_type, W, M>;

    Axis                              axis = Axis::size_;
    Items<Ownership::const_reference> bounds;
    Items<Ownership::reference>       edep;

    //! Whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return axis != Axis::size_ && !bounds.empty() && !edep.empty();
    }
};

//---------------------------------------------------------------------------//
// KERNEL LAUNCHER(S)
//---------------------------------------------------------------------------//
/*!
 * Diagnostic kernel launcher
 */
template<MemSpace M>
class EnergyDiagnosticLauncher
{
  public:
    //!@{
    //! Type aliases
    using real_type    = celeritas::real_type;
    using ThreadId     = celeritas::ThreadId;
    using Pointers     = EnergyBinPointers<M>;
    using StateRef     = celeritas::CoreStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION
    EnergyDiagnosticLauncher(const StateRef& states, const Pointers& pointers);

    // Perform energy binning by position
    inline CELER_FUNCTION void operator()(ThreadId tid) const;

  private:
    const StateRef&     states_;
    const Pointers&     pointers_;
};

using PointersDevice = EnergyBinPointers<MemSpace::device>;
using PointersHost   = EnergyBinPointers<MemSpace::host>;

void bin_energy(const celeritas::CoreStateDeviceRef& states,
                PointersDevice&                      pointers);
void bin_energy(const celeritas::CoreStateHostRef& states,
                PointersHost&                      pointers);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
// EnergyDiagnostic implementation
//---------------------------------------------------------------------------//
/*!
 * Construct with grid bounds and axis.
 */
template<MemSpace M>
EnergyDiagnostic<M>::EnergyDiagnostic(const std::vector<real_type>& bounds,
                                      Axis                          axis)
    : Diagnostic<M>(), axis_(axis)
{
    CELER_EXPECT(axis != Axis::size_);
    using HostItems
        = celeritas::Collection<real_type, Ownership::value, MemSpace::host>;

    // Create collection on host and copy to device
    HostItems bounds_host;
    make_builder(&bounds_host).insert_back(bounds.cbegin(), bounds.cend());
    bounds_ = bounds_host;

    // Resize bin data
    resize(&energy_per_bin_, bounds_.size() - 1);
    celeritas::fill(real_type(0), &energy_per_bin_);
}

//---------------------------------------------------------------------------//
/*!
 * Accumulate energy deposition in diagnostic.
 */
template<MemSpace M>
void EnergyDiagnostic<M>::mid_step(const StateRef& states)
{
    // Set up pointers to pass to device
    EnergyBinPointers<M> pointers;
    pointers.axis           = axis_;
    pointers.bounds         = bounds_;
    pointers.edep           = energy_per_bin_;

    // Invoke kernel for binning energies
    demo_loop::bin_energy(states, pointers);
}

//---------------------------------------------------------------------------//
/*!
 * Collect the diagnostic results.
 */
template<MemSpace M>
void EnergyDiagnostic<M>::get_result(TransporterResult* result)
{
    result->edep = this->energy_deposition();
}

//---------------------------------------------------------------------------//
/*!
 * Get vector of binned energy deposition.
 */
template<MemSpace M>
std::vector<celeritas::real_type> EnergyDiagnostic<M>::energy_deposition()
{
    // Copy binned energy deposition to host
    std::vector<real_type> edep(energy_per_bin_.size());
    celeritas::copy_to_host(energy_per_bin_, celeritas::make_span(edep));
    return edep;
}

//---------------------------------------------------------------------------//
// EnergyDiagnosticLauncher implementation
//---------------------------------------------------------------------------//
template<MemSpace M>
CELER_FUNCTION
EnergyDiagnosticLauncher<M>::EnergyDiagnosticLauncher(const StateRef& states,
                                                      const Pointers& pointers)
    : states_(states), pointers_(pointers)
{
    CELER_EXPECT(states_);
    CELER_EXPECT(pointers_);
}

//---------------------------------------------------------------------------//
template<MemSpace M>
CELER_FUNCTION void EnergyDiagnosticLauncher<M>::operator()(ThreadId tid) const
{
    celeritas::SimTrackView sim(states_.sim, tid);
    if (sim.status() == celeritas::TrackStatus::inactive)
    {
        // Only apply to active and dying tracks
        return;
    }

    // Create grid from EnergyBinPointers
    celeritas::NonuniformGrid<real_type> grid(pointers_.bounds);

    real_type pos = states_.geometry.pos[tid][static_cast<int>(pointers_.axis)];
    {
        // Bump particle to mid-step point to avoid grid edges coincident with
        // geometry boundaries
        // XXX this is not right if multiple scattering is on or for magnetic
        // fields!!! The only way we can be really sure to deposit energy in
        // the correct grid cell is to have the same boundary treatment as the
        // main geometry, so that the magnetic field and multiple scattering
        // take care to stop at the edge.
        // Until then, this heuristic will have to do.
        // XXX at the time being the "step" we've hacked into here may not be
        // the same as the geometry step or the true step.
        real_type dir
            = states_.geometry.dir[tid][static_cast<int>(pointers_.axis)];

        pos -= real_type(0.5) * states_.sim.state[tid].step_limit.step * dir;
    }

    using BinId = celeritas::ItemId<real_type>;
    if (pos > grid.front() && pos < grid.back())
    {
        real_type energy_deposition
            = states_.physics.state[tid].energy_deposition;
        if (energy_deposition > 0)
        {
            // Particle might not have deposited energy (geometry step for
            // photon, not-alive track, etc.): avoid the atomic if so
            auto bin = grid.find(pos);
            celeritas::atomic_add(&pointers_.edep[BinId{bin}],
                                  energy_deposition);
        }
    }
}

#if !CELER_USE_DEVICE
inline void bin_energy(const celeritas::CoreStateDeviceRef&, PointersDevice&)
{
    CELER_NOT_CONFIGURED("CUDA/HIP");
}
#endif

//---------------------------------------------------------------------------//
} // namespace demo_loop
