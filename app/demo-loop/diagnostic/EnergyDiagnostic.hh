//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EnergyDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Diagnostic.hh"

#include <vector>
#include "celeritas_config.h"
#include "base/Atomics.hh"
#include "base/CollectionAlgorithms.hh"
#include "base/CollectionBuilder.hh"
#include "base/CollectionMirror.hh"
#include "base/Macros.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/grid/NonuniformGrid.hh"
#include "sim/TrackData.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Diagnostic class for collecting energy deposition by z stored in device
 * memory.
 */
template<MemSpace M>
class EnergyDiagnostic : public Diagnostic<M>
{
  public:
    using real_type    = celeritas::real_type;
    using Items        = celeritas::Collection<real_type, Ownership::value, M>;
    using StateDataRef = celeritas::StateData<Ownership::reference, M>;
    using TransporterResult = celeritas::TransporterResult;

    EnergyDiagnostic(const std::vector<real_type>& z_bounds);

    // Number of alive tracks determined at the end of a step.
    void end_step(const StateDataRef& states) final;

    // Collect diagnostic results
    void get_result(TransporterResult* result) final;

    // Get vector of binned energy deposition
    std::vector<real_type> energy_deposition();

  private:
    Items z_bounds_;
    Items energy_by_z_;
};

//---------------------------------------------------------------------------//
/*!
 * Holds pointers to z grid and binned energy values to pass to kernel
 */
template<MemSpace M>
struct EnergyBinPointers
{
    template<Ownership W>
    using Items = celeritas::Collection<celeritas::real_type, W, M>;

    Items<Ownership::const_reference> z_bounds; //!< z bounds
    Items<Ownership::reference> energy_by_z; //!< Binned energy values for each
                                             //!< z interval

    //! Whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return !z_bounds.empty() && !energy_by_z.empty();
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
    using StateDataRef = celeritas::StateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION EnergyDiagnosticLauncher(const StateDataRef& states,
                                            const Pointers&     pointers);

    //! Perform energy binning by z position
    inline CELER_FUNCTION void operator()(ThreadId tid) const;

  private:
    const StateDataRef& states_;
    const Pointers&     pointers_;
};

using PointersDevice = EnergyBinPointers<MemSpace::device>;
using PointersHost   = EnergyBinPointers<MemSpace::host>;

void bin_energy(const celeritas::StateDeviceRef& states,
                PointersDevice&                  pointers);
void bin_energy(const celeritas::StateHostRef& states, PointersHost& pointers);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
// EnergyDiagnostic implementation
//---------------------------------------------------------------------------//
template<MemSpace M>
EnergyDiagnostic<M>::EnergyDiagnostic(const std::vector<real_type>& z_bounds)
    : Diagnostic<M>()
{
    using HostItems
        = celeritas::Collection<real_type, Ownership::value, MemSpace::host>;

    // Create collection on host and copy to device
    HostItems z_bounds_host;
    make_builder(&z_bounds_host).insert_back(z_bounds.cbegin(), z_bounds.cend());
    z_bounds_ = z_bounds_host;

    // Resize bin data
    resize(&energy_by_z_, z_bounds_.size() - 1);
}

//---------------------------------------------------------------------------//
/*!
 * Accumulate energy deposition in diagnostic
 */
template<MemSpace M>
void EnergyDiagnostic<M>::end_step(const StateDataRef& states)
{
    // Set up pointers to pass to device
    EnergyBinPointers<M> pointers;
    pointers.z_bounds    = z_bounds_;
    pointers.energy_by_z = energy_by_z_;

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
 * Get vector of binned energy deposition
 */
template<MemSpace M>
std::vector<celeritas::real_type> EnergyDiagnostic<M>::energy_deposition()
{
    // Copy binned energy deposition to host
    std::vector<real_type> edep(energy_by_z_.size());
    celeritas::copy_to_host(energy_by_z_, celeritas::make_span(edep));
    return edep;
}

//---------------------------------------------------------------------------//
// EnergyDiagnosticLauncher implementation
//---------------------------------------------------------------------------//
template<MemSpace M>
CELER_FUNCTION EnergyDiagnosticLauncher<M>::EnergyDiagnosticLauncher(
    const StateDataRef& states, const Pointers& pointers)
    : states_(states), pointers_(pointers)
{
    CELER_EXPECT(states_);
    CELER_EXPECT(pointers_);
}

//---------------------------------------------------------------------------//
template<MemSpace M>
CELER_FUNCTION void EnergyDiagnosticLauncher<M>::operator()(ThreadId tid) const
{
    // Create grid from EnergyBinPointers
    celeritas::NonuniformGrid<real_type> grid(pointers_.z_bounds);

    real_type z_pos             = states_.geometry.pos[tid][2];
    real_type energy_deposition = states_.energy_deposition[tid];

    using BinId = celeritas::ItemId<real_type>;
    if (z_pos > grid.front() && z_pos < grid.back())
    {
        auto bin = grid.find(z_pos);
        celeritas::atomic_add(&pointers_.energy_by_z[BinId{bin}],
                              energy_deposition);
    }
}

#if !CELERITAS_USE_CUDA
inline void bin_energy(const celeritas::StateDeviceRef&, PointersDevice&)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif

//---------------------------------------------------------------------------//
} // namespace demo_loop
