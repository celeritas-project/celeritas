//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/GeantSimpleCaloSD.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4VSensitiveDetector.hh>

#include "corecel/Types.hh"

namespace celeritas
{
namespace detail
{
struct GeantSimpleCaloStorage;
//---------------------------------------------------------------------------//
/*!
 * Accumulate energy deposition in volumes.
 *
 * This SD is returned by \c GeantSimpleCalo.
 */
class GeantSimpleCaloSD : public G4VSensitiveDetector
{
  public:
    //!@{
    //! \name Type aliases
    using SPStorage = std::shared_ptr<detail::GeantSimpleCaloStorage>;
    //!@}

  public:
    // Construct with name and shared params
    GeantSimpleCaloSD(SPStorage storage, size_type thread_id);

  protected:
    void Initialize(G4HCofThisEvent*) final {}
    bool ProcessHits(G4Step*, G4TouchableHistory*) final;

  private:
    SPStorage storage_;
    size_type thread_id_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
