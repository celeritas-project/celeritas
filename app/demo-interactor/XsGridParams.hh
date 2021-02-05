//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file XsGridParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "base/DeviceVector.hh"
#include "physics/grid/XsGridInterface.hh"

namespace demo_interactor
{
//---------------------------------------------------------------------------//
/*!
 * Manage 1D arrays of energy-depenent data used by physics classes.
 *
 * For EM physics, above \c prime_energy, the input cross section is provided
 * as a multiple of the particle's kinetic energy. During transport, the cross
 * sections are interpolated and then *if above this energy, the cross section
 * is divided by the particle's energy*.
 *
 * TODO: for the purposes of the demo app, this only holds a single array which
 * must be uniformly log-spaced.
 */
class XsGridParams
{
  public:
    using real_type      = celeritas::real_type;
    using XsGridPointers = celeritas::XsGridPointers;

    struct Input
    {
        std::vector<real_type> energy;       // MeV
        std::vector<real_type> xs;           // 1/cm
        real_type              prime_energy; // See class documentation
    };

  public:
    // Construct with input data
    explicit XsGridParams(const Input& input);

    // Access on-device data
    XsGridPointers device_pointers() const;

    // Get host-side data
    XsGridPointers host_pointers() const;

  private:
    celeritas::UniformGridData         log_energy_;
    celeritas::DeviceVector<real_type> xs_;
    celeritas::size_type               prime_index_;

    // Host side xs data
    std::vector<real_type> host_xs_;
};

//---------------------------------------------------------------------------//
} // namespace demo_interactor
