//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EnergyDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Diagnostic.hh"

#include <vector>
#include "base/CollectionMirror.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "sim/TrackData.hh"

using celeritas::Collection;
using celeritas::real_type;

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
    using StateDataRef = StateData<Ownership::reference, M>;

    EnergyDiagnostic(const std::vector<real_type>& z_bounds);

    // Number of alive tracks determined at the end of a step.
    void end_step(const StateDataRef& states) final;

    // Get vector of binned energy deposition
    std::vector<real_type> energy_deposition();

  private:
    Collection<real_type, Ownership::value, M> z_bounds_;
    Collection<real_type, Ownership::value, M> energy_by_z_;
};

//---------------------------------------------------------------------------//
/*!
 * Holds pointers to z grid and binned energy values to pass to kernel
 */
template<MemSpace M>
struct EnergyBinPointers
{
    template<Ownership W>
    using Items = Collection<real_type, W, M>;

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
    using StateDataRef = StateData<Ownership::reference, M>;
    using Pointers     = EnergyBinPointers<M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION EnergyDiagnosticLauncher(const StateDataRef& states,
                                            const Pointers&     pointers);

    //! Perform energy binning by z position
    inline CELER_FUNCTION void operator()(celeritas::ThreadId tid) const;

  private:
    const StateDataRef& states_;
    const Pointers&     pointers_;
};

using StateDataRefDevice = StateData<Ownership::reference, MemSpace::device>;
using StateDataRefHost   = StateData<Ownership::reference, MemSpace::host>;
using PointersDevice     = EnergyBinPointers<MemSpace::device>;
using PointersHost       = EnergyBinPointers<MemSpace::host>;

void bin_energy(const StateDataRefDevice& states, PointersDevice& pointers);
void bin_energy(const StateDataRefHost& states, PointersHost& pointers);

//---------------------------------------------------------------------------//
} // namespace demo_loop

#include "EnergyDiagnostic.i.hh"
