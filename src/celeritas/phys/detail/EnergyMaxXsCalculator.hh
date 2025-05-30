//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/EnergyMaxXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/inp/Grid.hh"
#include "celeritas/phys/PhysicsOptions.hh"
#include "celeritas/phys/Process.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Find the energy where the macroscopic cross section is largest.
 *
 * This is used in the integral approach of sampling a discrete interaction
 * length when a particle loses energy along a step.
 */
class EnergyMaxXsCalculator
{
  public:
    // Construct with physics options and process
    EnergyMaxXsCalculator(PhysicsOptions const&, Process const&);

    // Calculate the energy of the maximum cross section in the grid
    real_type operator()(inp::XsGrid const&) const;

    //! Whether the integral approach is used
    explicit operator bool() const { return use_integral_xs_; }

  private:
    bool use_integral_xs_;
    bool is_annihilation_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
