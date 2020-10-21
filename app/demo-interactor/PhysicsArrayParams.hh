//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsArrayParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "base/DeviceVector.hh"
#include "PhysicsArrayPointers.hh"

namespace celeritas
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
class PhysicsArrayParams
{
  public:
    struct Input
    {
        std::vector<real_type> energy;
        std::vector<real_type> xs;
        real_type              prime_energy; // See class documentation
    };

  public:
    // Construct with input data
    explicit PhysicsArrayParams(const Input& input);

    // Access on-device data
    PhysicsArrayPointers device_pointers() const;

    // Get host-side data
    PhysicsArrayPointers host_pointers() const;

  private:
    UniformGrid::Params     log_energy_;
    DeviceVector<real_type> xs_;
    real_type               prime_energy_;

    // Host side xs data
    std::vector<real_type> host_xs_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
