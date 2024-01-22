//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/XsGridParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/grid/XsGridData.hh"

#include "KNDemoKernel.hh"

namespace celeritas
{
namespace app
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
    using HostRef = TableData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef = TableData<Ownership::const_reference, MemSpace::device>;

    struct Input
    {
        std::vector<real_type> energy;  // MeV
        std::vector<real_type> xs;  // 1/cm
        real_type prime_energy;  // See class documentation
    };

  public:
    // Construct with input data
    explicit XsGridParams(Input const& input);

    // Access on-device data
    DeviceRef const& device_ref() const { return data_.device_ref(); }

    // Get host-side data
    HostRef const& host_ref() const { return data_.host_ref(); }

  private:
    CollectionMirror<TableData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
