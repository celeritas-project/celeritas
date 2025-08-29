//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/MuHadEmStandardPhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/g4/SupportedEmStandardPhysics.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct EM standard physics, including those not implemented in Celeritas.
 */
class MuHadEmStandardPhysics : public SupportedEmStandardPhysics
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
