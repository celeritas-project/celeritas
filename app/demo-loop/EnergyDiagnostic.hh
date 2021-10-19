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
#include "sim/TrackInterface.hh"

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
    void end_step(const StateDataRef& data) final;

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
struct EnergyBinPointers
{
    Collection<real_type, Ownership::const_reference, MemSpace::device>
        z_bounds; //!< z bounds
    Collection<real_type, Ownership::reference, MemSpace::device>
        energy_by_z; //!< Binned energy values for each z interval

    //! Whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return !z_bounds.empty() && !energy_by_z.empty();
    }
};

//---------------------------------------------------------------------------//
// KERNEL LAUNCHER(S)
//---------------------------------------------------------------------------//

using StateDataRefDevice = StateData<Ownership::reference, MemSpace::device>;

void bin_energy(const StateDataRefDevice& states, EnergyBinPointers& pointers);

//---------------------------------------------------------------------------//
} // namespace demo_loop

#include "EnergyDiagnostic.i.hh"
