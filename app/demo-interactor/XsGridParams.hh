//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file XsGridParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "base/CollectionMirror.hh"
#include "physics/grid/XsGridInterface.hh"
#include "KNDemoKernel.hh"

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
    using real_type = celeritas::real_type;
    using HostRef   = TableData<celeritas::Ownership::const_reference,
                              celeritas::MemSpace::host>;
    using DeviceRef = TableData<celeritas::Ownership::const_reference,
                                celeritas::MemSpace::device>;

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
    const DeviceRef& device_pointers() const { return data_.device(); }

    // Get host-side data
    const HostRef& host_pointers() const { return data_.host(); }

  private:
    celeritas::CollectionMirror<TableData> data_;
};

//---------------------------------------------------------------------------//
} // namespace demo_interactor
