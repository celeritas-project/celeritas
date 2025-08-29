//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/EmStandardPhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/g4/SupportedEmStandardPhysics.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct full EmStandardPhysics.
 *
 * These are both Celeritas supported physics and additional muon/hadronic
 * EM processes.
 */
class EmStandardPhysics : public SupportedEmStandardPhysics
{
  public:
    using SupportedEmStandardPhysics::SupportedEmStandardPhysics;

    // Set up minimal EM particle list
    void ConstructParticle() override;
    // Set up process list
    void ConstructProcess() override;

  private:
    void construct_particle();
    void construct_process();
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
